from copy import deepcopy
from itertools import chain

from torch.autograd import Variable

from config import cfg
import torch.optim as optim
from model import StructureNet, ModificationGen, D_NET_256 as Dis256, D_NET_128 as Dis128
from pretrained import VGG16, CNN_ENCODER, RNN_ENCODER
from datasets import get_dataloader
from GlobalAttention import words_similarity, sent_similarity
import random
from utils import *
from torch.nn import functional as F
from utils import mkdir

gpu_id = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
gpu_num = len(gpu_id.split(','))
device = torch.device("cuda")
gpus = [ix for ix in range(gpu_num)]
bs = 22
TEXT_E_DIR = "../data/Models/bird_DAMSM/Model/text_encoder_299.pth"
CNN_E_DIR = "../data/Models/bird_DAMSM/Model/image_encoder_299.pth"
VGG_PATH = "../Models/vgg16.pth"
DEBUG = False


def load_network(n_words):
    text_encoder = RNN_ENCODER(n_words, nhidden=cfg.TEXT.EMBEDDING_DIM).cuda(gpus[0])
    text_encoder.load_state_dict(torch.load(TEXT_E_DIR))
    print("Load text_encoder from " + TEXT_E_DIR)

    cnn_encoder = CNN_ENCODER(train=False).cuda(gpus[0])
    cnn_encoder.load_state_dict(torch.load(CNN_E_DIR))
    print("Load img_encoder from " + CNN_E_DIR)

    structureNet = StructureNet()
    structureNet.apply(weights_init)
    structureNet = torch.nn.DataParallel(structureNet, device_ids=gpus)

    modificationGen = ModificationGen()
    modificationGen.apply(weights_init)
    modificationGen = torch.nn.DataParallel(modificationGen, device_ids=gpus)

    dis2 = Dis256()
    dis2.apply(weights_init)
    dis2 = torch.nn.DataParallel(dis2, device_ids=gpus)

    dis = Dis128()
    dis.apply(weights_init)
    dis = torch.nn.DataParallel(dis, device_ids=gpus)

    vgg = VGG16(model_path=VGG_PATH).to(device)
    vgg = torch.nn.DataParallel(vgg, device_ids=gpus)

    epoch = 0

    return structureNet, modificationGen, dis2, dis, cnn_encoder, text_encoder, vgg, epoch


def define_optimizers(structureNet, modificationGen, netD, netD2):
    optimizerD2 = optim.Adam(netD2.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))
    params = chain(structureNet.parameters(), modificationGen.parameters())
    optimizerGE = optim.Adam(params, lr=2e-4, betas=(0.5, 0.999))
    return optimizerD2, optimizerD, optimizerGE


def save_gen_model(structureNet, modificationGen, epoch, model_dir):
    # torch.save(structureNet.state_dict(), '%s/structureNet_%d.pth' % (model_dir, epoch),
    # _use_new_zipfile_serialization=False)
    # torch.save(modificationGen.state_dict(), '%s/modificationGen_%d.pth' % (model_dir, epoch),
    # _use_new_zipfile_serialization=False)
    torch.save(structureNet.state_dict(), '%s/structureNet_%d.pth' % (model_dir, epoch))
    torch.save(modificationGen.state_dict(), '%s/modificationGen_%d.pth' % (model_dir, epoch))

class CrossEntropy():
    def __init__(self):
        self.code_loss = nn.CrossEntropyLoss()

    def __call__(self, prediction, label):
        # check label if hard (onehot)
        if label.max(dim=1)[0].min() == 1:
            return self.code_loss(prediction, torch.nonzero(label.long())[:, 1])
        else:
            log_prediction = torch.log_softmax(prediction, dim=1)
            return (- log_prediction * label).sum(dim=1).mean(dim=0)


class Trainer(object):
    def __init__(self, output_dir):

        # make dir for all kinds of output
        self.model_dir = os.path.join(output_dir, 'Model')
        self.image_dir = os.path.join(output_dir, 'Image')
        mkdir(self.image_dir)
        mkdir(self.model_dir)

        # other variables
        self.batch_size = bs
        self.normal_list = [i for i in range(self.batch_size)]
        self.adjusted_list = [i for i in range(self.batch_size - 1)]
        self.adjusted_list.insert(0, self.batch_size - 1)

        # make dataloader
        self.dataloader, self.n_words, self.ixtoword = get_dataloader(bs=self.batch_size, split="train",
                                                                      add_seg=False, drop_last=True, debug=False)

        self.structureNet, self.modificationGen, self.dis2, self.dis, self.cnn_encoder, \
        self.text_encoder, self.vgg, self.start_epoch = load_network(n_words=self.n_words)

        self.optimizerD2, self.optimizerD, self.optimizerGE = \
            define_optimizers(self.structureNet, self.modificationGen, self.dis, self.dis2)

        # get fix sample
        self.fix_img, self.sent_emb, self.words_embs, self.mask, self.noise = self.save_fix()

    def save_fix(self):
        fix_img, _, segs, _, caps, cap_lens, _ = self.prepare_data(next(iter(self.dataloader)))

        # save text
        texts = list()
        for i in range(len(caps)):
            sent = ""
            cap = caps[i].data.cpu().numpy()
            cap_len = cap_lens[i]
            for j in range(cap_len - 1):
                word = self.ixtoword[cap[j]].encode('ascii', 'ignore').decode('ascii')
                sent += (word + " ")
            sent += self.ixtoword[cap[cap_len - 1]].encode('ascii', 'ignore').decode('ascii')
            texts.append(sent)

        save_text_results(texts, 1, self.image_dir)

        with torch.no_grad():
            hidden = self.text_encoder.init_hidden(self.batch_size)
            words_embs, sent_emb = self.text_encoder(caps, cap_lens, hidden)
            mask = (caps == 0)
            num_words = words_embs.size(2)
            if mask.size(1) > num_words:
                mask = mask[:, :num_words]

        bg = fix_img * (1 - segs)
        fg = fix_img * segs

        save_img_results(fix_img.cpu(), None, 0, self.image_dir)
        save_img_results(fg.cpu(), None, 1, self.image_dir)
        save_img_results(segs.cpu(), None, 2, self.image_dir)
        save_img_results(bg.cpu(), None, 3, self.image_dir)

        noise = torch.FloatTensor(self.batch_size, cfg.GAN.Z_DIM).normal_(0, 1).to(device)

        return fix_img, sent_emb, words_embs, mask, noise

    def prepare_data(self, data):
        # 128, 256
        imgs_, segs_, class_ids, captions, cap_lens, _ = data
        imgs = imgs_[0]
        segs = segs_[0]
        imgs256 = imgs_[1]
        segs256 = segs_[1]
        # sort data by the length in a decreasing order
        sorted_cap_lens, sorted_cap_indices = \
            torch.sort(cap_lens, 0, True)

        imgs = imgs[sorted_cap_indices]
        segs = segs[sorted_cap_indices]
        imgs256 = imgs256[sorted_cap_indices]
        segs256 = segs256[sorted_cap_indices]

        captions = captions[sorted_cap_indices].squeeze()
        class_ids = class_ids[sorted_cap_indices].numpy()

        imgs = imgs.to(device)
        segs = segs.to(device)
        imgs256 = imgs256.to(device)
        segs256 = segs256.to(device)
        captions = captions.to(device)
        sorted_cap_lens.to(device)

        return imgs256, imgs, segs256, segs, captions, sorted_cap_lens, class_ids

    def toZeroThreshold(self, x, t=0.1):
        zeros = torch.zeros_like(x).to(device)
        return torch.where(x > t, x, zeros)

    def train(self):
        # prepare net, optimizer and loss
        self.structureNet.train()
        self.modificationGen.train()
        self.dis.train()
        self.dis2.train()

        self.vgg.eval()
        self.cnn_encoder.eval()
        self.text_encoder.eval()

        self.RF_loss = nn.BCELoss()
        self.L1 = nn.L1Loss()
        self.CE = CrossEntropy()

        for epoch in range(self.start_epoch, cfg.TRAIN.FIRST_MAX_EPOCH + 1):

            start_t = time.time()
            print("oops")

            for data in self.dataloader:

                self.log = ""
                img256, img, mask256, mask128, captions, cap_lens, class_ids = self.prepare_data(data)

                with torch.no_grad():
                    hidden = self.text_encoder.init_hidden(self.batch_size)
                    words_embs, sent_emb = self.text_encoder(captions, cap_lens, hidden)

                    mask = (captions == 0)
                    num_words = words_embs.size(2)
                    if mask.size(1) > num_words:
                        mask = mask[:, :num_words]

                    words_embs = words_embs.detach()
                    sent_emb = sent_emb.detach()

                # shape distribution
                [cond_feat2, cond_feat, input_feat], img_mask256, _ = self.structureNet(img256)
                img_mask256 = self.toZeroThreshold(img_mask256)

                cond_feat = cond_feat.detach()
                cond_feat2 = cond_feat2.detach()

                noise = torch.FloatTensor(self.batch_size, cfg.GAN.Z_DIM).normal_(0, 1).to(device)

                gen256, gen, attn_mask256, attn_mask, mu, logvar = \
                    self.modificationGen(input_feat, noise, sent_emb, words_embs, mask, cond_feat, cond_feat2, device)

                # for real pair
                real_img256 = img256
                real_fore256 = img256 * mask256
                real_img128 = img * mask128

                # for fake pair
                bg256 = img256 * (1 - img_mask256)
                bg256 = bg256.detach()

                # fake ony foreground
                fake_img128 = gen * attn_mask
                fake_fore256 = gen256 * attn_mask256
                fake_img256 = fake_fore256 + bg256 * (1 - attn_mask256)

                self.trainD(0, sent_emb, real_img128.detach(), fake_img128.detach(), None, None)

                self.trainD(1, sent_emb, real_fore256.detach(), fake_fore256.detach(),
                            real_img256.detach(), fake_img256.detach())

                self.trainEG(fake_img256, fake_fore256, fake_img128, mu, logvar, sent_emb, words_embs,
                             cap_lens, class_ids, mask256, img_mask256, real_img256)

            self.structureNet.eval()
            self.modificationGen.eval()

            with torch.no_grad():

                [cond_feat2, cond_feat, input_feat], img_mask, _ = self.structureNet(self.fix_img)
                img_mask = self.toZeroThreshold(img_mask)
                bg = self.fix_img * (1 - img_mask)

                fake_fore256, fake_fore, fake_attn256, fake_attn, _, _ = \
                    self.modificationGen(input_feat, self.noise, self.sent_emb[self.adjusted_list],
                                         self.words_embs[self.adjusted_list], self.mask[self.adjusted_list],
                                         cond_feat=cond_feat, cond_feat2=cond_feat2)

                fake_masked256 = fake_fore256 * fake_attn256
                fake_masked = fake_fore * fake_attn
                fake_img256 = bg * (1 - fake_attn256) + fake_masked256

                # reconstruction
                rec_fore, _, rec_attn, _ = self.modificationGen(input_feat, self.noise, self.sent_emb,
                                           self.words_embs, self.mask, cond_feat=cond_feat, cond_feat2=cond_feat2)

                rec_masked = rec_fore * rec_attn

                save_img = [img_mask, fake_attn, fake_attn256, fake_masked, fake_masked256, fake_img256, rec_masked]
                save_img_results(None, save_img, epoch, self.image_dir)

            self.structureNet.train()
            self.modificationGen.train()

            end_t = time.time()
            print('''[%d/%d] Time: %.2fs\n'''
                  '''%s'''
                  % (epoch, cfg.TRAIN.FIRST_MAX_EPOCH, end_t - start_t, self.log))

            if (epoch + 1) % 50 == 0:
                save_gen_model(self.structureNet, self.modificationGen, epoch, self.model_dir)

    def trainD(self, idx, sent_emb, real_fg, fake_fg, real_img=None, fake_img=None):

        if idx == 0:
            dis = self.dis
            optimizer = self.optimizerD
        else:
            dis = self.dis2
            optimizer = self.optimizerD2

        optimizer.zero_grad()
        # get feature
        real_fg_feat = dis.module.get_feat(real_fg)
        fake_fg_feat = dis.module.get_feat(fake_fg)
        # prediction
        real_fg_pred, real_cond_pred = dis(real_fg_feat, sent_emb)
        fake_fg_pred, fake_cond_pred = dis(fake_fg_feat, sent_emb)
        _, wrong_cond_pred = dis(real_fg_feat, sent_emb)

        # get label
        real_label = torch.ones_like(real_fg_pred)
        fake_label = torch.zeros_like(fake_fg_pred)

        # get loss
        real_loss = self.RF_loss(real_fg_pred, real_label)
        fake_loss = self.RF_loss(fake_fg_pred, fake_label)
        real_cond_loss = self.RF_loss(real_cond_pred, real_label)
        fake_cond_loss = self.RF_loss(fake_cond_pred, fake_label) / 2.
        wrong_cond_loss = self.RF_loss(wrong_cond_pred, fake_label) / 2.

        d_loss = (real_loss + fake_loss) + (real_cond_loss + fake_cond_loss + wrong_cond_loss)

        if idx == 0:
            self.log += 'netD_0 loss : real_fg %.4f, fake_fg %.4f\n' \
                        'real_cond %.4f, fake_cond %.4f, wrong_cond %.4f\n\n' \
                        % (real_loss, fake_loss, real_cond_loss, fake_cond_loss, wrong_cond_loss)
        else:
            # get feature
            real_feat = dis.module.get_feat(real_img)
            fake_feat = dis.module.get_feat(fake_img)
            # prediction
            real_pred = dis(real_feat)
            fake_pred = dis(fake_feat)

            real_loss2 = self.RF_loss(real_pred, real_label) / 2.
            fake_loss2 = self.RF_loss(fake_pred, fake_label) / 2.
            d_loss += (real_loss2 + fake_loss2)
            self.log += 'netD_1 loss : real_fg %.4f, fake_fg %.4f, real %.4f, fake: %.4f\n' \
                        'real_cond %.4f, fake_cond %.4f, wrong_cond %.4f\n\n' \
                        % (real_loss, fake_loss, real_loss2, fake_loss2, real_cond_loss,
                           fake_cond_loss, wrong_cond_loss)

        d_loss.backward()
        optimizer.step()

    def trainEG(self, fake_img256, fake_fg256, fake_img128, mu, logvar, sent_emb, word_embs, cap_lens,
                class_ids, real_mask, gen_mask, real_img):

        bs = self.batch_size
        self.optimizerGE.zero_grad()
        gen_mask = gen_mask.repeat(1, 3, 1, 1)
        mask_loss = nn.BCEWithLogitsLoss()(gen_mask, real_mask) * cfg.TRAIN.WEIGHT.LAMBDA_MASK

        kl_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        g_kl_loss = torch.sum(kl_element).mul_(-0.5) * cfg.TRAIN.WEIGHT.LAMBDA_KL

        # pred feat for 256
        pred_feat256 = self.dis2.module.get_feat(fake_img256)
        pred256 = self.dis2(pred_feat256)

        pred_fg_feat256 = self.dis2.module.get_feat(fake_fg256)
        pred_fg256, pred_cond256 = self.dis2(pred_fg_feat256, sent_emb)

        # get loss for 256
        real_label = torch.ones_like(pred256)
        pred_fg_loss256 = self.RF_loss(pred_fg256, real_label)
        pred_loss256 = self.RF_loss(pred256, real_label)
        pred_cond_loss256 = self.RF_loss(pred_cond256, real_label)

        # pred feat for 128
        pred_feat128 = self.dis.module.get_feat(fake_img128)
        pred128, pred_cond128 = self.dis(pred_feat128, sent_emb)

        # get loss for 128
        real_label = torch.ones_like(pred128)
        pred_loss128 = self.RF_loss(pred128, real_label)
        pred_cond_loss128 = self.RF_loss(pred_cond128, real_label)

        g_loss = (pred_loss128 + pred_cond_loss128) + (pred_loss256 + pred_fg_loss256 + pred_cond_loss256) \
                 + g_kl_loss + mask_loss

        # text-image similarity loss
        region_feature, cnn_code = self.cnn_encoder(fake_img256)
        word_sim, _ = words_similarity(region_feature, word_embs, cap_lens, bs,
                                       class_ids=class_ids)

        sent_sim = sent_similarity(cnn_code, sent_emb, bs, class_ids=class_ids)

        match_labels = Variable(torch.LongTensor(range(self.batch_size))).cuda()

        word_loss = (nn.CrossEntropyLoss()(word_sim, match_labels) +
                     nn.CrossEntropyLoss()(word_sim.transpose(0, 1), match_labels)) * cfg.TRAIN.WEIGHT.LAMBDA_SIM

        sent_loss = (nn.CrossEntropyLoss()(sent_sim, match_labels) +
                     nn.CrossEntropyLoss()(sent_sim.transpose(0, 1), match_labels)) * cfg.TRAIN.WEIGHT.LAMBDA_SIM

        # vgg loss
        real_features = self.vgg(real_img)
        fake_features = self.vgg(fake_img256)
        feature_loss = 0
        for i in range(len(real_features)):
            cur_real_features = real_features[i]
            cur_fake_features = fake_features[i]
            # feature_loss += F.mse_loss(cur_real_features, cur_fake_features) * self.vgg.weights[i]
            feature_loss += F.mse_loss(cur_real_features, cur_fake_features) * cfg.TRAIN.WEIGHT.LAMBDA_VGG

        self.log += "netG: pred_fg_loss %.4f, pred_cond_loss %.4f, pred_fg_loss2 %.4f, pred_cond_loss2 %.4f, " \
                    "pred_loss2 %.4f\n" \
                    "word loss %.4f, sent_loss %.4f, kl_loss %.4f, mask loss %.4f, vgg loss %.4f\n\n" % \
                    (pred_loss128, pred_cond_loss128, pred_fg_loss256, pred_cond_loss256, pred_loss256,
                     word_loss, sent_loss, g_kl_loss, mask_loss, feature_loss)

        g_loss += (word_loss + sent_loss) + feature_loss
        g_loss.backward()
        self.optimizerGE.step()


if __name__ == "__main__":
    print("start")
    manualSeed = random.randint(1, 10000)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    # prepare output folder for this running and save all files

    output_dir = make_output_dir(del_old=True)
    shutil.copy2(sys.argv[0], output_dir)
    shutil.copy2('model.py', output_dir)
    shutil.copy2('pretrained.py', output_dir)
    shutil.copy2('base_function.py.py', output_dir)
    shutil.copy2('datasets.py', output_dir)
    shutil.copy2('config.py', output_dir)
    shutil.copy2('utils.py', output_dir)
    trainer = Trainer(output_dir)
    print('start training now')
    trainer.train()
