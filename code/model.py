
from collections import OrderedDict
import torch
import torch.nn as nn
from config import cfg
from GlobalAttention import GlobalAttentionGeneral as ATT_NET
from base_function import *
import torch.nn.functional as F
from torch.autograd import Variable

class StructureNet(nn.Module):
    def __init__(self, in_c=3, out_c=32):
        super().__init__()
        self.first = nn.Sequential(sameBlock(in_c, 64), sameBlock(64, 64), Down_unet(64, 64))
        # 128*64*64
        self.down1 = Down_unet(64, 128)
        # 256*32*32
        self.down2 = Down_unet(128, 256)
        # 512*16*16
        self.down3 = Down_unet(256, 512)
        # 512*8*8
        self.down4 = Down_unet(512, 512)

        # 256*16*16
        self.up1 = Up_unet(512, 256)
        # 512*32*32
        self.up2 = Up_unet(256 + 512, 256)
        # 256*64*64
        self.up3 = Up_unet(256 + 256, 128)
        # 128*128*128
        self.up4 = Up_unet(128 + 128, 64)
        self.last = nn.Sequential(Up_unet(64, out_c), ResBlock(out_c), ResBlock(out_c))
        self.get_mask = GET_MASK(out_c)

    def forward(self, img):
        x0 = self.first(img)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x = self.up1(x4)
        x = self.up2(torch.cat([x, x3], dim=1))
        x = self.up3(torch.cat([x, x2], dim=1))
        x = self.up4(torch.cat([x, x1], dim=1))
        feat = self.last(x)
        mask = self.get_mask(feat)
        mask128 = F.interpolate(mask, [128, 128], mode='nearest')
        # 64 x 128 x 128
        # 128 x 64 x 64
        # 512 x 16 x 16
        return [x0, x1, x3], mask, mask128


# reference:https://github.com/tobran/DF-GAN/blob/master/code/model.py
class affine(nn.Module):

    def __init__(self, feat_dim, cond_dim):
        super(affine, self).__init__()

        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(cond_dim, 256)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(256, feat_dim)),
            ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(cond_dim, 256)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(256, feat_dim)),
            ]))

    def forward(self, x, y=None):

        weight = self.fc_gamma(y)
        bias = self.fc_beta(y)

        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)

        size = x.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * x + bias


class SFFB_fuse(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super(SFFB_fuse, self).__init__()

        self.learnable_sc = in_ch != out_ch
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.affine0 = affine(in_ch, cond_dim=cond_dim)
        self.affine1 = affine(in_ch, cond_dim=cond_dim)
        self.gamma = nn.Parameter(torch.zeros(1))
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, x, y=None):
        h = self.affine0(x, y)
        h = nn.LeakyReLU(0.2, inplace=True)(h)
        h = self.affine1(h, y)
        h = nn.LeakyReLU(0.2, inplace=True)(h)
        h = self.c1(h)
        return h

    def forward(self, x, y=None):
        return self.shortcut(x) + self.gamma * self.residual(x, y)


class CFFB_fuse(nn.Module):
    def __init__(self, x_dim, feat_dim, attn_dim):
        super(CFFB_fuse, self).__init__()
        # attn feat
        self.attn_conv = conv3x3(attn_dim, 128)
        self.attn_weight = conv3x3(128, x_dim)
        self.attn_bias = conv3x3(128, x_dim)
        # image feat
        self.feat_conv = conv3x3(feat_dim, 128)
        self.feat_weight = conv3x3(128, x_dim)

    def forward(self, x, feat_x, attn_x):
        # x and cond is same width and height
        feat = self.feat_conv(feat_x)
        feat_weight = self.feat_weight(feat)

        attn = self.attn_conv(attn_x)
        attn_weight = self.attn_weight(attn)
        attn_bias = self.attn_bias(attn)
        return x * (feat_weight+attn_weight) + attn_bias


class CA_NET(nn.Module):
    def __init__(self, emb_dim, c_dim):
        super().__init__()
        self.c_dim = c_dim
        self.emb_dim = emb_dim
        self.fc = nn.Linear(self.emb_dim, self.c_dim * 4, bias=True)
        self.c_dim = cfg.GAN.Z_DIM
        self.relu = GLU()

    def encode(self, code):
        code = self.relu(self.fc(code))
        mu = code[:, :self.c_dim]
        logvar = code[:, self.c_dim:]
        return mu, logvar

    def reparameterize(self, mu, logvar, device):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = eps.to(device)
        eps = Variable(eps, requires_grad=True)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding, device):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparameterize(mu, logvar, device)
        return c_code, mu, logvar


class RNN_CA(nn.Module):
    def __init__(self, text_embs=cfg.TEXT.EMBEDDING_DIM, ca_dim=cfg.GAN.Z_DIM, out_dim=cfg.TRAIN.CLASS_NUM):
        super().__init__()
        # 100
        self.ca_dim = ca_dim
        # 256
        self.emb_dim = text_embs
        self.ca = CA_NET(emb_dim=text_embs, c_dim=self.ca_dim)

        # 200
        self.fc = nn.Sequential(
            nn.Linear(self.ca_dim * 2, out_dim * 2, bias=False),
            nn.BatchNorm1d(out_dim * 2),
            GLU())

    def forward(self, z_code, sent_embs, device):
        c_code, mu, logvar = self.ca(sent_embs, device)
        c_z_code = torch.cat((c_code, z_code), 1)
        out_code = self.fc(c_z_code)
        return out_code, mu, logvar


class ModificationGen(nn.Module):

    def __init__(self, ngf=64, num_residual=2, emb_dim=cfg.TRAIN.Z_DIM, feat_dim1=128, feat_dim2=64):
        super().__init__()

        self.ngf = ngf
        self.code_len = emb_dim
        self.text_emb_dim = cfg.TEXT.EMBEDDING_DIM
        self.cond_emb_dim = emb_dim
        self.num_residual = num_residual
        self.define_module()

        self.ca_net = RNN_CA()

        self.sent_fuse_block = nn.ModuleList([
            SFFB_fuse(self.ngf*8, self.ngf*8, self.code_len),
            upBlock(self.ngf*8, self.ngf*4),
            SFFB_fuse(self.ngf*4, self.ngf*2, self.code_len),
            upBlock(self.ngf*2, self.ngf)
        ])

        self.att1 = ATT_NET(self.ngf, self.text_emb_dim)
        self.cross_fuse1 = CFFB_fuse(self.ngf, feat_dim=feat_dim1, attn_dim=self.ngf)
        self.upBlock1 = nn.Sequential(self._make_layer(self.ngf, num_residual=2),
                                      upBlock(self.ngf, self.ngf))

        self.att2 = ATT_NET(self.ngf, self.text_emb_dim)
        self.cross_fuse2 = CFFB_fuse(self.ngf, feat_dim=feat_dim2, attn_dim=self.ngf)
        self.upBlock2 = nn.Sequential(self._make_layer(self.ngf, num_residual=2),
                                      upBlock(self.ngf, self.ngf))

        self.get_image_128 = GET_IMAGE(self.ngf)
        self.get_mask_128 = GET_MASK(self.ngf)

        self.get_image_256 = GET_IMAGE(self.ngf)
        self.get_mask_256 = GET_MASK(self.ngf)

    def _make_layer(self, channel_num, num_residual):
        layers = []
        for _ in range(num_residual):
            layers.append(ResBlock(channel_num=channel_num, norm=True))
        return nn.Sequential(*layers)

    def sent_injuct(self, h_feat, c_code):
        h_c_code = self.sent_fuse_block[0](h_feat, c_code)
        h_c_code = self.sent_fuse_block[1](h_c_code)
        h_c_code = self.sent_fuse_block[2](h_c_code, c_code)
        h_c_code = self.sent_fuse_block[3](h_c_code)
        return h_c_code

    def word_inject1(self, h_code, cond_feat, word_embs, mask):
        self.att1.applyMask(mask)
        attn_code, _ = self.att1(h_code, word_embs)
        h_c_code = self.cross_fuse1(h_code, cond_feat, attn_code)
        h_c_code = self.upBlock1(h_c_code)
        return h_c_code

    def word_inject2(self, h_code, cond_feat, word_embs, mask):
        self.att2.applyMask(mask)
        attn_code, attn_map = self.att2(h_code, word_embs)
        h_c_code = self.cross_fuse2(h_code, cond_feat, attn_code)
        h_c_code = self.upBlock2(h_c_code)
        return h_c_code, attn_map

    def forward(self, h_feat, noise, sent_emb, word_embs, mask, cond_feat, cond_feat2, device):
        """
        :param h_feat: bs x 512 x 16 x 16
        :param c_code: bs x 200
        :param word_embs: bs x 256 x L
        :param mask: bs x L
        :param cond_feat: bs x feat_dim1 x 64 x 64, feat_dim2 x 128 x 128
        :return: img, mask
        """
        # step1. get the condition code
        c_code, mu, logvar = self.ca_net(noise, sent_emb, device)

        # step2. sentence-level fusion
        h_c_code = self.sent_injuct(h_feat, c_code)

        # step3. word-level fusion
        h_c_code = self.word_inject1(h_c_code, cond_feat, word_embs, mask)
        h_c_code2, _ = self.word_inject2(h_c_code, cond_feat2, word_embs, mask)

        img128 = self.get_image_128(h_c_code)
        mask128 = self.get_mask_128(h_c_code)
        img256 = self.get_image_256(h_c_code2)
        mask256 = self.get_mask_256(h_c_code2)

        return img256, img128, mask256, mask128, mu, logvar


class D_NET_256(nn.Module):
    def __init__(self, ndf=64):
        super(D_NET_256, self).__init__()
        self.ndf = ndf
        self.code_len = cfg.TEXT.EMBEDDING_DIM

        self.image_layer = nn.Sequential(
            encode_image_by_16times(self.ndf),
            # 512 x 16 x 16
            downBlock(self.ndf * 8, self.ndf * 16),
            downBlock(self.ndf * 16, self.ndf * 16),
            # 1024 x 4 x 4
            Block3x3_leakRelu(self.ndf * 16, self.ndf * 8),
            Block3x3_leakRelu(self.ndf * 8, self.ndf * 8)
            # 512 x 4 x 4
        )

        self.rf_logits = nn.Sequential(nn.Conv2d(self.ndf*8, 1, kernel_size=4, stride=4), nn.Sigmoid())

        self.sent_joint = nn.Sequential(Block3x3_leakRelu(self.ndf*8+self.code_len, self.ndf*8),
                                        nn.Conv2d(self.ndf*8, 1, kernel_size=4, stride=4),
                                        nn.Sigmoid())

    def get_feat(self, image):
        x_code = self.image_layer(image)
        return x_code

    def forward(self, x_code, c_code=None):
        if c_code is None:
            return self.rf_logits(x_code)
        else:
            h, w = x_code.size(2), x_code.size(3)
            c_code = c_code.view(-1, self.code_len, 1, 1).repeat(1, 1, h, w)
            x_c_code = torch.cat((x_code, c_code), 1)
            return self.rf_logits(x_code), self.sent_joint(x_c_code)


class D_NET_128(nn.Module):
    def __init__(self):
        super(D_NET_128).__init__()
        self.ndf = cfg.GAN.DF_DIM # 64
        self.code_len = cfg.TEXT.EMBEDDING_DIM

        self.image_layer = nn.Sequential(
            encode_image_by_16times(self.ndf),
            downBlock(self.ndf * 8, self.ndf * 16),
            Block3x3_leakRelu(self.ndf * 16, self.ndf * 8)
        )

        self.rf_logits = nn.Sequential(nn.Conv2d(self.ndf*8, 1, kernel_size=4, stride=4), nn.Sigmoid())

        self.sent_joint = nn.Sequential(Block3x3_leakRelu(self.ndf*8+self.code_len, self.ndf*8),
                                        nn.Conv2d(self.ndf*8, 1, kernel_size=4, stride=4),
                                        nn.Sigmoid())

    def get_feat(self, image):
        x_code = self.image_layer(image)
        return x_code

    def forward(self, x_code, c_code=None):
        if c_code is None:
            return self.rf_logits(x_code)
        else:
            h, w = x_code.size(2), x_code.size(3)
            c_code = c_code.view(-1, self.code_len, 1, 1).repeat(1, 1, h, w)
            x_c_code = torch.cat((x_code, c_code), 1)
            return self.rf_logits(x_code), self.sent_joint(x_c_code)


