from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import transforms

from config import cfg
import numpy as np
import os
import time
import torch.backends.cudnn as cudnn
import torch

import sys
import shutil
import torch.nn as nn
import skimage.transform
import ntpath

import torchvision.utils as vutils


def mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)


def path_leaf(path):
    return ntpath.basename(path)


def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def make_output_dir(out_dir=None, del_old=True):
    root_path = '../../data/output/'
    if out_dir is not None:
        new_output = out_dir
    else:
        args = sys.argv[1:]
        if len(args) == 0:
            # raise RuntimeError('output folder must be specified')
            import datetime
            import dateutil.tz
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            new_output = '%s_%s' % \
                         (cfg.DATASET_NAME, now.strftime('%Y_%m_%d_%H_%M_%S'))
        else:
            new_output = args[0]

    path = os.path.join(root_path, new_output)

    if del_old:
        if os.path.exists(path):
            print('WARNING: experiment directory exists, it will be erased and recreated in 3s')
            time.sleep(3)
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            os.makedirs(path)

    return path


def child_to_parent(child_c_code, classes_child, classes_parent):
    ratio = classes_child / classes_parent
    arg_parent = torch.argmax(child_c_code, dim=1) / ratio
    parent_c_code = torch.zeros([child_c_code.size(0), classes_parent]).cuda()
    for i in range(child_c_code.size(0)):
        parent_c_code[i][int(arg_parent[i])] = 1
    return parent_c_code


def save_text_results(text, count, text_dir):
    file_name = '%s/real_sample%03d.txt' % (text_dir, count)
    f = open(file_name, "w+")
    for sent in text:
        f.write(sent + "\n")
    f.close()


def get_tensor_img(dir, order=False):
    normalize = transforms.Compose(
        [transforms.Resize(128, 128), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    fileList = os.listdir(dir)
    img_list = []

    for i in range(len(fileList)):
        file = os.path.join(dir, fileList[i])
        img = Image.open(file).convert('RGB')
        img_list.append(normalize(img))

    if not order:
        from random import random
        # also could use shuffle function
        img_nums = len(img_list)
        # ids = [i for i in range(img_nums)]
        for i in range(img_nums):
            j = int(random() * img_nums)
            # if j > img_nums - 1: j = img_nums - 1
            img_list[i], img_list[j] = img_list[j], img_list[i]

    return torch.stack(img_list, dim=0)


def get_tensor_img2(dir, filename):
    normalize = transforms.Compose(
        [transforms.Resize(128, 128), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    with open(filename, "r") as f:
        img_names = f.read().split('\n')

    if img_names[-1] == ' ':
        img_names.pop()

    img_list = []
    for i in range(len(img_names)):
        file = os.path.join(dir, img_names[i])
        img = Image.open(file).convert('RGB')
        img_list.append(normalize(img))

    return torch.stack(img_list, dim=0)


def save_img_results(imgs_tcpu, fake_imgs, count, image_dir, nrow=8):
    num = cfg.TRAIN.VIS_COUNT * 8888

    if imgs_tcpu is not None:
        real_img = imgs_tcpu[:][0:num]
        vutils.save_image(
            real_img, '%s/real_samples%03d.png' % (image_dir, count),
            scale_each=True, normalize=True, nrow=nrow)
        real_img_set = vutils.make_grid(real_img).numpy()
        real_img_set = np.transpose(real_img_set, (1, 2, 0))
        real_img_set = real_img_set * 255
        real_img_set = real_img_set.astype(np.uint8)

    if fake_imgs is not None:

        for i in range(len(fake_imgs)):
            fake_img = fake_imgs[i][0:num]

            vutils.save_image(
                fake_img.data, '%s/count_%09d_fake_samples%d.png' %
                               (image_dir, count, i), scale_each=True, normalize=True, nrow=nrow)

            fake_img_set = vutils.make_grid(fake_img.data).cpu().numpy()

            fake_img_set = np.transpose(fake_img_set, (1, 2, 0))
            fake_img_set = (fake_img_set + 1) * 255 / 2
            fake_img_set = fake_img_set.astype(np.uint8)


def zero_last_layer(encoder):
    encoder.model_z[4].weight.data.fill_(0.0)
    encoder.model_z[4].bias.data.fill_(0.0)
    return encoder


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0):
    # adapted from cyclegan
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2


    Returns the gradient penalty loss
    """

    if type == 'real':  # either use real images, fake images, or a linear interpolation of two.
        interpolatesv = real_data
    elif type == 'fake':
        interpolatesv = fake_data
    elif type == 'mixed':
        interpolatesv = []
        for i in range(len(real_data)):
            alpha = torch.rand(real_data[i].shape[0], 1)
            alpha = alpha.expand(real_data[i].shape[0],
                                 real_data[i].nelement() // real_data[i].shape[0]).contiguous().view(
                *real_data[i].shape)
            alpha = alpha.to(device)
            interpolatesv.append(alpha * real_data[i] + ((1 - alpha) * fake_data[i]))
    else:
        raise NotImplementedError('{} not implemented'.format(type))

    # require grad
    for i in range(len(interpolatesv)):
        interpolatesv[i].requires_grad_(True)

    # feed into D
    disc_interpolates = netD(*interpolatesv)

    # cal penalty

    gradient_penalty = 0
    for i in range(len(disc_interpolates)):
        for j in range(len(interpolatesv)):
            gradients = torch.autograd.grad(outputs=disc_interpolates[i], inputs=interpolatesv[j],
                                            grad_outputs=torch.ones(disc_interpolates[i].size()).to(device),
                                            create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)

            if gradients[0] is not None:  # it will return None if input is not used in this output (allow unused)
                gradients = gradients[0].view(real_data[j].size(0), -1)  # flat the data
                gradient_penalty += (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean()  # added eps

    return gradient_penalty


def MA_GP_penalty(netD, real_img, sent_embs):
    interpolated = (real_img.data).requires_grad_(True)
    sent_inter = (sent_embs.data).requires_grad_(True)
    features = netD(interpolated)
    out = netD.module.COND_DNET(features, sent_inter)
    grads = torch.autograd.grad(outputs=out,
                                inputs=(interpolated, sent_inter),
                                grad_outputs=torch.ones(out.size()).cuda(),
                                retain_graph=True,
                                create_graph=True,
                                only_inputs=True)
    grad0 = grads[0].view(grads[0].size(0), -1)
    grad1 = grads[1].view(grads[1].size(0), -1)
    grad = torch.cat((grad0, grad1), dim=1)
    grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
    penalty = torch.mean((grad_l2norm) ** 6) * 2

    return penalty


#
# For visualization ################################################
COLOR_DIC = {0: [128, 64, 128], 1: [244, 35, 232],
             2: [70, 70, 70], 3: [102, 102, 156],
             4: [190, 153, 153], 5: [153, 153, 153],
             6: [250, 170, 30], 7: [220, 220, 0],
             8: [107, 142, 35], 9: [152, 251, 152],
             10: [70, 130, 180], 11: [220, 20, 60],
             12: [255, 0, 0], 13: [0, 0, 142],
             14: [119, 11, 32], 15: [0, 60, 100],
             16: [0, 80, 100], 17: [0, 0, 230],
             18: [0, 0, 70], 19: [0, 0, 0]}
FONT_MAX = 40


def drawCaption(convas, captions, ixtoword, vis_size, off1=2, off2=2):
    num = captions.size(0)
    img_txt = Image.fromarray(convas)
    fnt = ImageFont.truetype('../eval/FreeMono.ttf', 30)
    # get a drawing context
    d = ImageDraw.Draw(img_txt)
    sentence_list = []
    for i in range(num):
        cap = captions[i].data.cpu().numpy()
        sentence = []
        for j in range(len(cap)):
            if cap[j] == 0:
                break
            word = ixtoword[cap[j]].encode('ascii', 'ignore').decode('ascii')
            d.text(((j + off1) * (vis_size + off2), i * FONT_MAX), '%d:%s' % (j, word[:6]),
                   font=fnt, fill=(255, 255, 255, 255))
            sentence.append(word)
        sentence_list.append(sentence)
    return img_txt, sentence_list


def build_super_images(real_imgs, captions, ixtoword,
                       attn_maps, att_sze, lr_imgs=None,
                       batch_size=cfg.TRAIN.BATCH_SIZE,
                       max_word_num=cfg.TEXT.WORDS_NUM, nvis=4):
    real_imgs = real_imgs[:nvis]
    if lr_imgs is not None:
        lr_imgs = lr_imgs[:nvis]

    if att_sze == 17:
        vis_size = att_sze * 16
    else:
        vis_size = real_imgs.size(2)

    text_convas = \
        np.ones([batch_size * FONT_MAX,
                 (max_word_num + 2) * (vis_size + 2), 3],
                dtype=np.uint8)

    for i in range(max_word_num):
        istart = (i + 2) * (vis_size + 2)
        iend = (i + 3) * (vis_size + 2)
        text_convas[:, istart:iend, :] = COLOR_DIC[i]

    real_imgs = \
        nn.Upsample(size=(vis_size, vis_size), mode='bilinear')(real_imgs)
    # [-1, 1] --> [0, 1]
    real_imgs.add_(1).div_(2).mul_(255)
    real_imgs = real_imgs.data.numpy()
    # b x c x h x w --> b x h x w x c
    real_imgs = np.transpose(real_imgs, (0, 2, 3, 1))

    # ??
    pad_sze = real_imgs.shape
    middle_pad = np.zeros([pad_sze[2], 2, 3])
    post_pad = np.zeros([pad_sze[1], pad_sze[2], 3])
    if lr_imgs is not None:
        lr_imgs = \
            nn.Upsample(size=(vis_size, vis_size), mode='bilinear')(lr_imgs)
        # [-1, 1] --> [0, 1]
        lr_imgs.add_(1).div_(2).mul_(255)
        lr_imgs = lr_imgs.data.numpy()
        # b x c x h x w --> b x h x w x c
        lr_imgs = np.transpose(lr_imgs, (0, 2, 3, 1))

    # batch x seq_len x 17 x 17 --> batch x 1 x 17 x 17
    seq_len = max_word_num
    img_set = []
    num = nvis  # len(attn_maps)

    text_map, sentences = \
        drawCaption(text_convas, captions, ixtoword, vis_size)
    text_map = np.asarray(text_map).astype(np.uint8)

    bUpdate = 1
    for i in range(num):
        attn = attn_maps[i].cpu().view(1, -1, att_sze, att_sze)
        # --> 1 x dim x 17 x 17
        attn_max = attn.max(dim=1, keepdim=True)
        attn = torch.cat([attn_max[0], attn], 1)
        #
        attn = attn.view(-1, 1, att_sze, att_sze)
        attn = attn.repeat(1, 3, 1, 1).data.numpy()
        # n x c x h x w --> n x h x w x c
        attn = np.transpose(attn, (0, 2, 3, 1))
        num_attn = attn.shape[0]
        #
        img = real_imgs[i]
        if lr_imgs is None:
            lrI = img
        else:
            lrI = lr_imgs[i]

        row = [lrI, middle_pad]
        row_merge = [img, middle_pad]

        row_beforeNorm = []
        minVglobal, maxVglobal = 1, 0
        for j in range(num_attn):
            one_map = attn[j]
            if (vis_size // att_sze) > 1:
                one_map = \
                    skimage.transform.pyramid_expand(one_map, sigma=20,
                                                     upscale=vis_size // att_sze, multichannel=True)
            row_beforeNorm.append(one_map)
            minV = one_map.min()
            maxV = one_map.max()
            if minVglobal > minV:
                minVglobal = minV
            if maxVglobal < maxV:
                maxVglobal = maxV
        for j in range(seq_len + 1):
            if j < num_attn:
                one_map = row_beforeNorm[j]
                one_map = (one_map - minVglobal) / (maxVglobal - minVglobal)
                one_map *= 255
                #
                PIL_im = Image.fromarray(np.uint8(img))
                PIL_att = Image.fromarray(np.uint8(one_map))
                merged = \
                    Image.new('RGBA', (vis_size, vis_size), (0, 0, 0, 0))
                mask = Image.new('L', (vis_size, vis_size), (210))
                # I see
                merged.paste(PIL_im, (0, 0))
                merged.paste(PIL_att, (0, 0), mask)
                merged = np.array(merged)[:, :, :3]
            else:
                one_map = post_pad
                merged = post_pad
            row.append(one_map)
            row.append(middle_pad)
            #
            row_merge.append(merged)
            row_merge.append(middle_pad)
        row = np.concatenate(row, 1)
        row_merge = np.concatenate(row_merge, 1)
        txt = text_map[i * FONT_MAX: (i + 1) * FONT_MAX]
        if txt.shape[1] != row.shape[1]:
            print('txt', txt.shape, 'row', row.shape)
            bUpdate = 0
            break
        row = np.concatenate([txt, row, row_merge], 0)
        img_set.append(row)
    if bUpdate:
        img_set = np.concatenate(img_set, 0)
        img_set = img_set.astype(np.uint8)
        return img_set, sentences
    else:
        return None


def build_super_images2(real_imgs, captions, cap_lens, ixtoword,
                        attn_maps, att_sze, vis_size=256, topK=5):
    batch_size = real_imgs.size(0)
    max_word_num = np.max(cap_lens)
    text_convas = np.ones([batch_size * FONT_MAX,
                           max_word_num * (vis_size + 2), 3],
                          dtype=np.uint8)

    real_imgs = \
        nn.Upsample(size=(vis_size, vis_size), mode='bilinear')(real_imgs)
    # [-1, 1] --> [0, 1]
    real_imgs.add_(1).div_(2).mul_(255)
    real_imgs = real_imgs.data.numpy()
    # b x c x h x w --> b x h x w x c
    real_imgs = np.transpose(real_imgs, (0, 2, 3, 1))
    pad_sze = real_imgs.shape
    middle_pad = np.zeros([pad_sze[2], 2, 3])

    # batch x seq_len x 17 x 17 --> batch x 1 x 17 x 17
    img_set = []
    num = len(attn_maps)

    text_map, sentences = \
        drawCaption(text_convas, captions, ixtoword, vis_size, off1=0)
    text_map = np.asarray(text_map).astype(np.uint8)

    bUpdate = 1
    for i in range(num):
        attn = attn_maps[i].cpu().view(1, -1, att_sze, att_sze)
        #
        attn = attn.view(-1, 1, att_sze, att_sze)
        attn = attn.repeat(1, 3, 1, 1).data.numpy()
        # n x c x h x w --> n x h x w x c
        attn = np.transpose(attn, (0, 2, 3, 1))
        num_attn = cap_lens[i]
        thresh = 2. / float(num_attn)
        #
        img = real_imgs[i]
        row = []
        row_merge = []
        row_txt = []
        row_beforeNorm = []
        conf_score = []

        for j in range(num_attn):
            one_map = attn[j]
            mask0 = one_map > (2. * thresh)
            conf_score.append(np.sum(one_map * mask0))
            mask = one_map > thresh
            one_map = one_map * mask

            # focus there
            if (vis_size // att_sze) > 1:
                one_map = \
                    skimage.transform.pyramid_expand(one_map, sigma=20,
                                                     upscale=vis_size // att_sze, multichannel=True)
            minV = one_map.min()
            maxV = one_map.max()
            one_map = (one_map - minV) / (maxV - minV)
            row_beforeNorm.append(one_map)
        sorted_indices = np.argsort(conf_score)[::-1]

        for j in range(num_attn):
            one_map = row_beforeNorm[j]
            one_map *= 255
            #
            PIL_im = Image.fromarray(np.uint8(img))
            PIL_att = Image.fromarray(np.uint8(one_map))
            merged = \
                Image.new('RGBA', (vis_size, vis_size), (0, 0, 0, 0))
            mask = Image.new('L', (vis_size, vis_size), (180))  # (210)
            merged.paste(PIL_im, (0, 0))
            merged.paste(PIL_att, (0, 0), mask)
            merged = np.array(merged)[:, :, :3]

            row.append(np.concatenate([one_map, middle_pad], 1))
            row_merge.append(np.concatenate([merged, middle_pad], 1))
            #
            txt = text_map[i * FONT_MAX:(i + 1) * FONT_MAX,
                  j * (vis_size + 2):(j + 1) * (vis_size + 2), :]
            row_txt.append(txt)
        # reorder
        row_new = []
        row_merge_new = []
        txt_new = []
        for j in range(num_attn):
            idx = sorted_indices[j]
            row_new.append(row[idx])
            row_merge_new.append(row_merge[idx])
            txt_new.append(row_txt[idx])
        row = np.concatenate(row_new[:topK], 1)
        row_merge = np.concatenate(row_merge_new[:topK], 1)
        txt = np.concatenate(txt_new[:topK], 1)
        if txt.shape[1] != row.shape[1]:
            print('Warnings: txt', txt.shape, 'row', row.shape,
                  'row_merge_new', row_merge_new.shape)
            bUpdate = 0
            break
        row = np.concatenate([txt, row_merge], 0)
        img_set.append(row)
    if bUpdate:
        img_set = np.concatenate(img_set, 0)
        img_set = img_set.astype(np.uint8)
        return img_set, sentences
    else:
        return None


from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import transforms

from config import cfg
import numpy as np
import os
import time
import torch.backends.cudnn as cudnn
import torch

import sys
import shutil
import torch.nn as nn
import skimage.transform
import ntpath

import torchvision.utils as vutils


def mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)


def path_leaf(path):
    return ntpath.basename(path)


def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def make_output_dir(out_dir=None, del_old=True):
    root_path = '../../data/output/'
    if out_dir is not None:
        new_output = out_dir
    else:
        args = sys.argv[1:]
        if len(args) == 0:
            # raise RuntimeError('output folder must be specified')
            import datetime
            import dateutil.tz
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            new_output = '%s_%s' % \
                         (cfg.DATASET_NAME, now.strftime('%Y_%m_%d_%H_%M_%S'))
        else:
            new_output = args[0]

    path = os.path.join(root_path, new_output)

    if del_old:
        if os.path.exists(path):
            print('WARNING: experiment directory exists, it will be erased and recreated in 3s')
            time.sleep(3)
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            os.makedirs(path)

    return path


def save_text_results(text, count, text_dir):
    file_name = '%s/real_sample%03d.txt' % (text_dir, count)
    f = open(file_name, "w+")
    for sent in text:
        f.write(sent + "\n")
    f.close()


def save_img_results(imgs_tcpu, fake_imgs, count, image_dir, nrow=8):
    num = cfg.TRAIN.VIS_COUNT * 8888

    if imgs_tcpu is not None:
        real_img = imgs_tcpu[:][0:num]
        vutils.save_image(
            real_img, '%s/real_samples%03d.png' % (image_dir, count),
            scale_each=True, normalize=True, nrow=nrow)
        real_img_set = vutils.make_grid(real_img).numpy()
        real_img_set = np.transpose(real_img_set, (1, 2, 0))
        real_img_set = real_img_set * 255
        real_img_set = real_img_set.astype(np.uint8)

    if fake_imgs is not None:

        for i in range(len(fake_imgs)):
            fake_img = fake_imgs[i][0:num]

            vutils.save_image(
                fake_img.data, '%s/count_%09d_fake_samples%d.png' %
                               (image_dir, count, i), scale_each=True, normalize=True, nrow=nrow)

            fake_img_set = vutils.make_grid(fake_img.data).cpu().numpy()

            fake_img_set = np.transpose(fake_img_set, (1, 2, 0))
            fake_img_set = (fake_img_set + 1) * 255 / 2
            fake_img_set = fake_img_set.astype(np.uint8)


def zero_last_layer(encoder):
    encoder.model_z[4].weight.data.fill_(0.0)
    encoder.model_z[4].bias.data.fill_(0.0)
    return encoder

#
# For visualization ################################################
COLOR_DIC = {0: [128, 64, 128], 1: [244, 35, 232],
             2: [70, 70, 70], 3: [102, 102, 156],
             4: [190, 153, 153], 5: [153, 153, 153],
             6: [250, 170, 30], 7: [220, 220, 0],
             8: [107, 142, 35], 9: [152, 251, 152],
             10: [70, 130, 180], 11: [220, 20, 60],
             12: [255, 0, 0], 13: [0, 0, 142],
             14: [119, 11, 32], 15: [0, 60, 100],
             16: [0, 80, 100], 17: [0, 0, 230],
             18: [0, 0, 70], 19: [0, 0, 0]}
FONT_MAX = 40


def drawCaption(convas, captions, ixtoword, vis_size, off1=2, off2=2):
    num = captions.size(0)
    img_txt = Image.fromarray(convas)
    fnt = ImageFont.truetype('../eval/FreeMono.ttf', 30)
    # get a drawing context
    d = ImageDraw.Draw(img_txt)
    sentence_list = []
    for i in range(num):
        cap = captions[i].data.cpu().numpy()
        sentence = []
        for j in range(len(cap)):
            if cap[j] == 0:
                break
            word = ixtoword[cap[j]].encode('ascii', 'ignore').decode('ascii')
            d.text(((j + off1) * (vis_size + off2), i * FONT_MAX), '%d:%s' % (j, word[:6]),
                   font=fnt, fill=(255, 255, 255, 255))
            sentence.append(word)
        sentence_list.append(sentence)
    return img_txt, sentence_list


def build_super_images(real_imgs, captions, ixtoword,
                       attn_maps, att_sze, lr_imgs=None,
                       batch_size=cfg.TRAIN.BATCH_SIZE,
                       max_word_num=cfg.TEXT.WORDS_NUM, nvis=4):
    real_imgs = real_imgs[:nvis]
    if lr_imgs is not None:
        lr_imgs = lr_imgs[:nvis]

    if att_sze == 17:
        vis_size = att_sze * 16
    else:
        vis_size = real_imgs.size(2)

    text_convas = \
        np.ones([batch_size * FONT_MAX,
                 (max_word_num + 2) * (vis_size + 2), 3],
                dtype=np.uint8)

    for i in range(max_word_num):
        istart = (i + 2) * (vis_size + 2)
        iend = (i + 3) * (vis_size + 2)
        text_convas[:, istart:iend, :] = COLOR_DIC[i]

    real_imgs = \
        nn.Upsample(size=(vis_size, vis_size), mode='bilinear')(real_imgs)
    # [-1, 1] --> [0, 1]
    real_imgs.add_(1).div_(2).mul_(255)
    real_imgs = real_imgs.data.numpy()
    # b x c x h x w --> b x h x w x c
    real_imgs = np.transpose(real_imgs, (0, 2, 3, 1))

    # ??
    pad_sze = real_imgs.shape
    middle_pad = np.zeros([pad_sze[2], 2, 3])
    post_pad = np.zeros([pad_sze[1], pad_sze[2], 3])
    if lr_imgs is not None:
        lr_imgs = \
            nn.Upsample(size=(vis_size, vis_size), mode='bilinear')(lr_imgs)
        # [-1, 1] --> [0, 1]
        lr_imgs.add_(1).div_(2).mul_(255)
        lr_imgs = lr_imgs.data.numpy()
        # b x c x h x w --> b x h x w x c
        lr_imgs = np.transpose(lr_imgs, (0, 2, 3, 1))

    # batch x seq_len x 17 x 17 --> batch x 1 x 17 x 17
    seq_len = max_word_num
    img_set = []
    num = nvis  # len(attn_maps)

    text_map, sentences = \
        drawCaption(text_convas, captions, ixtoword, vis_size)
    text_map = np.asarray(text_map).astype(np.uint8)

    bUpdate = 1
    for i in range(num):
        attn = attn_maps[i].cpu().view(1, -1, att_sze, att_sze)
        # --> 1 x dim x 17 x 17
        attn_max = attn.max(dim=1, keepdim=True)
        attn = torch.cat([attn_max[0], attn], 1)
        #
        attn = attn.view(-1, 1, att_sze, att_sze)
        attn = attn.repeat(1, 3, 1, 1).data.numpy()
        # n x c x h x w --> n x h x w x c
        attn = np.transpose(attn, (0, 2, 3, 1))
        num_attn = attn.shape[0]
        #
        img = real_imgs[i]
        if lr_imgs is None:
            lrI = img
        else:
            lrI = lr_imgs[i]

        row = [lrI, middle_pad]
        row_merge = [img, middle_pad]

        row_beforeNorm = []
        minVglobal, maxVglobal = 1, 0
        for j in range(num_attn):
            one_map = attn[j]
            if (vis_size // att_sze) > 1:
                one_map = \
                    skimage.transform.pyramid_expand(one_map, sigma=20,
                                                     upscale=vis_size // att_sze, multichannel=True)
            row_beforeNorm.append(one_map)
            minV = one_map.min()
            maxV = one_map.max()
            if minVglobal > minV:
                minVglobal = minV
            if maxVglobal < maxV:
                maxVglobal = maxV
        for j in range(seq_len + 1):
            if j < num_attn:
                one_map = row_beforeNorm[j]
                one_map = (one_map - minVglobal) / (maxVglobal - minVglobal)
                one_map *= 255
                #
                PIL_im = Image.fromarray(np.uint8(img))
                PIL_att = Image.fromarray(np.uint8(one_map))
                merged = \
                    Image.new('RGBA', (vis_size, vis_size), (0, 0, 0, 0))
                mask = Image.new('L', (vis_size, vis_size), (210))
                # I see
                merged.paste(PIL_im, (0, 0))
                merged.paste(PIL_att, (0, 0), mask)
                merged = np.array(merged)[:, :, :3]
            else:
                one_map = post_pad
                merged = post_pad
            row.append(one_map)
            row.append(middle_pad)
            #
            row_merge.append(merged)
            row_merge.append(middle_pad)
        row = np.concatenate(row, 1)
        row_merge = np.concatenate(row_merge, 1)
        txt = text_map[i * FONT_MAX: (i + 1) * FONT_MAX]
        if txt.shape[1] != row.shape[1]:
            print('txt', txt.shape, 'row', row.shape)
            bUpdate = 0
            break
        row = np.concatenate([txt, row, row_merge], 0)
        img_set.append(row)
    if bUpdate:
        img_set = np.concatenate(img_set, 0)
        img_set = img_set.astype(np.uint8)
        return img_set, sentences
    else:
        return None


def build_super_images2(real_imgs, captions, cap_lens, ixtoword,
                        attn_maps, att_sze, vis_size=256, topK=5):
    batch_size = real_imgs.size(0)
    max_word_num = np.max(cap_lens)
    text_convas = np.ones([batch_size * FONT_MAX,
                           max_word_num * (vis_size + 2), 3],
                          dtype=np.uint8)

    real_imgs = \
        nn.Upsample(size=(vis_size, vis_size), mode='bilinear')(real_imgs)
    # [-1, 1] --> [0, 1]
    real_imgs.add_(1).div_(2).mul_(255)
    real_imgs = real_imgs.data.numpy()
    # b x c x h x w --> b x h x w x c
    real_imgs = np.transpose(real_imgs, (0, 2, 3, 1))
    pad_sze = real_imgs.shape
    middle_pad = np.zeros([pad_sze[2], 2, 3])

    # batch x seq_len x 17 x 17 --> batch x 1 x 17 x 17
    img_set = []
    num = len(attn_maps)

    text_map, sentences = \
        drawCaption(text_convas, captions, ixtoword, vis_size, off1=0)
    text_map = np.asarray(text_map).astype(np.uint8)

    bUpdate = 1
    for i in range(num):
        attn = attn_maps[i].cpu().view(1, -1, att_sze, att_sze)
        #
        attn = attn.view(-1, 1, att_sze, att_sze)
        attn = attn.repeat(1, 3, 1, 1).data.numpy()
        # n x c x h x w --> n x h x w x c
        attn = np.transpose(attn, (0, 2, 3, 1))
        num_attn = cap_lens[i]
        thresh = 2. / float(num_attn)
        #
        img = real_imgs[i]
        row = []
        row_merge = []
        row_txt = []
        row_beforeNorm = []
        conf_score = []

        for j in range(num_attn):
            one_map = attn[j]
            mask0 = one_map > (2. * thresh)
            conf_score.append(np.sum(one_map * mask0))
            mask = one_map > thresh
            one_map = one_map * mask

            # focus there
            if (vis_size // att_sze) > 1:
                one_map = \
                    skimage.transform.pyramid_expand(one_map, sigma=20,
                                                     upscale=vis_size // att_sze, multichannel=True)
            minV = one_map.min()
            maxV = one_map.max()
            one_map = (one_map - minV) / (maxV - minV)
            row_beforeNorm.append(one_map)
        sorted_indices = np.argsort(conf_score)[::-1]

        for j in range(num_attn):
            one_map = row_beforeNorm[j]
            one_map *= 255
            #
            PIL_im = Image.fromarray(np.uint8(img))
            PIL_att = Image.fromarray(np.uint8(one_map))
            merged = \
                Image.new('RGBA', (vis_size, vis_size), (0, 0, 0, 0))
            mask = Image.new('L', (vis_size, vis_size), (180))  # (210)
            merged.paste(PIL_im, (0, 0))
            merged.paste(PIL_att, (0, 0), mask)
            merged = np.array(merged)[:, :, :3]

            row.append(np.concatenate([one_map, middle_pad], 1))
            row_merge.append(np.concatenate([merged, middle_pad], 1))
            #
            txt = text_map[i * FONT_MAX:(i + 1) * FONT_MAX,
                  j * (vis_size + 2):(j + 1) * (vis_size + 2), :]
            row_txt.append(txt)
        # reorder
        row_new = []
        row_merge_new = []
        txt_new = []
        for j in range(num_attn):
            idx = sorted_indices[j]
            row_new.append(row[idx])
            row_merge_new.append(row_merge[idx])
            txt_new.append(row_txt[idx])
        row = np.concatenate(row_new[:topK], 1)
        row_merge = np.concatenate(row_merge_new[:topK], 1)
        txt = np.concatenate(txt_new[:topK], 1)
        if txt.shape[1] != row.shape[1]:
            print('Warnings: txt', txt.shape, 'row', row.shape,
                  'row_merge_new', row_merge_new.shape)
            bUpdate = 0
            break
        row = np.concatenate([txt, row_merge], 0)
        img_set.append(row)
    if bUpdate:
        img_set = np.concatenate(img_set, 0)
        img_set = img_set.astype(np.uint8)
        return img_set, sentences
    else:
        return None