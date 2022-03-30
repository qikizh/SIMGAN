import torch
import torch.nn as nn


def conv1x1(in_planes, out_planes, bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=bias)


def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)


def sameBlock(in_planes, out_planes):
    block = nn.Sequential(conv3x3(in_planes, out_planes * 2),
                          nn.BatchNorm2d(out_planes * 2),
                          GLU())
    return block


def Up_unet(in_c, out_c):
    return nn.Sequential(nn.ConvTranspose2d(in_c, out_c*2, 4, 2, 1), nn.BatchNorm2d(out_c*2), GLU())


def BottleNeck(in_c, out_c): # 瓶颈层
    return nn.Sequential(nn.Conv2d(in_c, out_c*2, 4, 4), nn.BatchNorm2d(out_c*2), GLU())


def Down_unet(in_c, out_c):
    return nn.Sequential(nn.Conv2d(in_c, out_c*2, 4, 2, 1), nn.BatchNorm2d(out_c*2), GLU())

###### For generator


class ResBlock(nn.Module):
    def __init__(self, channel_num, norm=False, norm_type='instance'):
        super(ResBlock, self).__init__()

        block = [conv3x3(channel_num, channel_num * 2),
                 # nn.BatchNorm2d(channel_num * 2),
                 GLU(),
                 conv3x3(channel_num, channel_num),
                 # nn.BatchNorm2d(channel_num)
                 ]
        if norm:
            if norm_type == 'instance':
                block.insert(1, nn.InstanceNorm2d(channel_num * 2))
                block.insert(4, nn.InstanceNorm2d(channel_num))
            elif norm_type == 'batch':
                block.insert(1, nn.BatchNorm2d(channel_num * 2))
                block.insert(4, nn.BatchNorm2d(channel_num))

        self.model = nn.Sequential(*block)

    def forward(self, x):
        return x + self.model(x)


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc / 2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class GET_IMAGE(nn.Module):
    def __init__(self, ngf):
        super().__init__()
        self.img = nn.Sequential(conv3x3(ngf, 3), nn.Tanh())

    def forward(self, h_code):
        return self.img(h_code)


class GET_MASK(nn.Module):
    def __init__(self, ngf):
        super().__init__()
        self.img = nn.Sequential(conv3x3(ngf, 1), nn.Sigmoid())

    def forward(self, h_code):
        return self.img(h_code)

# IN for generator
def upBlock(in_planes, out_planes):
    block = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                          conv3x3(in_planes, out_planes * 2),
                          nn.InstanceNorm2d(out_planes * 2),
                          GLU())
    return block


####### for discriminator

# BN
def downBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


def Block3x3_leakRelu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


def encode_image_by_16times(ndf):
    encode_img = nn.Sequential(
        # --> state size. ndf x in_size/2 x in_size/2
        nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 2ndf x x in_size/4 x in_size/4
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 4ndf x in_size/8 x in_size/8
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 8ndf x in_size/16 x in_size/16
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return encode_img