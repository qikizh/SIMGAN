from collections import defaultdict
from copy import deepcopy

import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os
from nltk.tokenize import RegexpTokenizer
import os.path
import random
import numpy as np
from config import cfg
import torch.utils.data as data
import os
import sys
import os.path
import pandas as pd
import torch
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data.dataset import T_co
from torchvision.transforms import RandomCrop, RandomHorizontalFlip, functional

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


class RandomCrop_img_mask(RandomCrop):
    def __init__(self, size):
        RandomCrop.__init__(self, size)

    def __call__(self, img, seg):
        # simply some parameters like: padding, pad_if_needed
        i, j, h, w = RandomCrop.get_params(img, self.size)
        return functional.crop(img, i, j, h, w), functional.crop(seg, i, j, h, w)


class RandomHorizontalFlip_img_mask(RandomHorizontalFlip):
    def __init__(self):
        RandomHorizontalFlip.__init__(self)

    def __call__(self, img, seg):
        if random.random() < self.p:
            return functional.hflip(img), functional.hflip(seg)
        else:
            return img, seg


def get_img_masks(img_path, seg_path, imsize, bbox=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    seg = Image.open(seg_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])
        seg = seg.crop([x1, y1, x2, y2])

    tmp_size = int(imsize[2] * 76 / 64)

    img = transforms.Resize([tmp_size, tmp_size])(img)
    seg = transforms.Resize([tmp_size, tmp_size])(seg)
    img, seg = RandomCrop_img_mask(imsize[2])(img, seg)
    img, seg = RandomHorizontalFlip_img_mask()(img, seg)

    img256 = normalize(img)
    img128 = normalize(transforms.Resize([imsize[1], imsize[1]])(img))

    seg256 = transforms.ToTensor()(seg)
    seg128 = transforms.ToTensor()(transforms.Resize([imsize[1], imsize[1]])(seg))

    # 128 x 128
    return [img128, img256], [seg128, seg256]


def get_imgs(img_path, imsize, bbox=None, transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    rec_img256 = normalize(img)
    rec_img128 = normalize(transforms.Resize(imsize[1])(img))

    return [rec_img128, rec_img256]


class InitDataMethod:
    def __init__(self):
        super(InitDataMethod, self).__init__()

    @staticmethod
    def init_bounding_box(data_dir, bbox_pickle_path):
        bbox_path = os.path.join(data_dir, 'bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        filepath = os.path.join(data_dir, 'images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            bbox = df_bounding_boxes.iloc[i][1:].tolist()
            key = filenames[i][:-4]
            filename_bbox[key] = bbox

        # write pickle
        write_file = open(bbox_pickle_path, 'wb')
        pickle.dump(filename_bbox, write_file)
        write_file.close()

        return filename_bbox

    @staticmethod
    def init_dictionary(data_dir, train_names, test_names, tokenizer, embeddings_num, caption_pickle_path):
        train_captions = InitDataMethod.load_captions(data_dir, train_names, tokenizer, embeddings_num)
        test_captions = InitDataMethod.load_captions(data_dir, test_names, tokenizer, embeddings_num)
        word_counts = defaultdict(float)

        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            # this train_captions_new hold index of each word in sentence
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)

        with open(caption_pickle_path, 'wb') as f:
            pickle.dump([train_captions, test_captions, ixtoword, wordtoix], f, protocol=2)
            print('Save to: ', caption_pickle_path)

        return train_captions_new, test_captions_new, ixtoword, wordtoix, len(ixtoword)

    @staticmethod
    # load captions from .txt, return captions
    def load_captions(data_dir, filenames, tokenizer, embeddings_num):
        all_captions = []
        for i in range(len(filenames)):
            cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "r") as f:
                captions = f.read().split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokens = tokenizer.tokenize(cap.lower())
                    # print('tokens', tokens)
                    if len(tokens) == 0:
                        print('cap', cap)
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
                    cnt += 1
                    if cnt == embeddings_num:
                        break
                if cnt < embeddings_num:
                    print('ERROR: the captions for %s less than %d'
                          % (filenames[i], cnt))
        return all_captions

    @staticmethod
    def init_class_ids():
        # TODO
        pass

    @staticmethod
    def init_filenames(data_dir, filenames_pickle_path):
        # TODO
        pass


class TextDataset(data.Dataset):
    tokenizer = RegexpTokenizer(r'\w+')

    def __init__(self, data_dir, base_size=64, split='train', add_mask=False, transform=None):
        super(self).__init__()

        self.data_dir = data_dir  # '../../data/birds'
        self.split = split
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE
        self.words_num = cfg.TEXT.WORDS_NUM
        self.add_mask = add_mask

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        if data_dir.find('birds') != -1:
            self.bbox = self.load_bbox()
        else:
            self.bbox = None

        # load text data
        self.filenames, self.captions, self.ixtoword, self.wordtoix, self.n_words = self.load_text_data(data_dir, split)
        self.class_ids = self.load_class_id(os.path.join(self.data_dir, self.split), len(self.filenames))
        self.iterator = self.prepare_pairs
        print("loading from %s/%s, the number is %d." % (self.data_dir, self.split, len(self.filenames)))

    def load_bbox(self):
        """
        Returns a dictionary with image filename as 'key' and its bounding box coordinates as 'value'
        """
        data_dir = self.data_dir
        bbox_pickle_path = os.path.join(data_dir, 'bounding_boxes.pickle')
        if not os.path.exists(bbox_pickle_path):
            filename_bbox = InitDataMethod.init_bounding_box(self.data_dir, bbox_pickle_path)
        else:
            read_file = open(bbox_pickle_path, 'rb')
            filename_bbox = pickle.load(read_file)
            read_file.close()

        return filename_bbox

    @staticmethod
    def load_filenames(data_dir, split):
        pickle_file_path = os.path.join(data_dir, split, 'filenames.pickle')
        if os.path.exists(pickle_file_path):
            read_file = open(pickle_file_path, 'rb')
            filenames = pickle.load(read_file)
            read_file.close()
        else:
            filenames = []
        print('Load filenames from: %s (%d)' % (pickle_file_path, len(filenames)))
        return filenames

    @staticmethod
    def load_class_id(data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f, encoding='latin1')
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'captions.pickle')
        train_names = self.load_filenames(data_dir, 'train')
        test_names = self.load_filenames(data_dir, 'test')
        if not os.path.isfile(filepath):
            train_captions, test_captions, ixtoword, wordtoix, n_words = \
                InitDataMethod.init_dictionary(data_dir, train_names, test_names, self.tokenizer, self.embeddings_num)
        else:
            with open(filepath, 'rb') as f:
                print("filepath", filepath)
                x = pickle.load(f)
                train_captions, test_captions = x[0], x[1]
                ixtoword, wordtoix = x[2], x[3]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)
        if split == 'train':
            # a list of list: each list contains the indices of words in a sentence
            captions = train_captions
            filenames = train_names
        elif split == 'test':
            captions = test_captions
            filenames = test_names

        return filenames, captions, ixtoword, wordtoix, n_words

    # make captions from text to index
    @staticmethod
    def prepare_caption(captions, wordtoix):
        captions_new = []
        for t in captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            captions_new.append(rev)

        return captions_new

    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((self.words_num, 1), dtype='int64')
        x_len = num_words
        if num_words <= self.words_num:
            x[:num_words, 0] = sent_caption
        # if the sentence is too long
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = cfg.TEXT.WORDS_NUM
        return x, x_len

    def prepare_pairs(self, index):
        key = self.filenames[index]
        cls_id = self.class_ids[index]

        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir

        # random select a sentence
        sent_ix = np.random.randint(0, self.embeddings_num)
        new_sent_ix = index * self.embeddings_num + sent_ix
        caps, cap_len = self.get_caption(new_sent_ix)

        img_name = '%s/images/%s.jpg' % (data_dir, key)

        if self.add_mask:
            seg_name = '%s/segmentations/%s.png' % (data_dir, key)
            imgs, segs = get_img_masks(img_name, seg_name, self.imsize, bbox, normalize=self.norm)
            return imgs, segs, cls_id, caps, cap_len, key
        else:
            # 128, 256
            imgs = get_imgs(img_name, self.imsize, bbox, self.transform, normalize=self.norm)
            return imgs, cls_id, caps, cap_len, key

    def __getitem__(self, index):
        return self.iterator(index)

    def __len__(self):
        return len(self.filenames)


def get_dataloader(bs, split="train", add_seg=False, drop_last=True, debug=False):
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))

    if split == "test":
        image_transform = transforms.Resize([imsize, imsize])
    else:
        image_transform = transforms.Compose([transforms.Resize(int(imsize * 76 / 64)),
                                              transforms.RandomCrop(imsize),
                                              transforms.RandomHorizontalFlip()])

    dataset = TextDataset(cfg.DATA_DIR, base_size=64, split=split, add_mask=add_seg, transform=image_transform)
    # debug mode the wks need to set 0
    if debug: wks = 0
    else: wks = 4
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, drop_last=drop_last, shuffle=True, num_workers=wks)
    return dataloader, dataset.n_words, dataset.ixtoword
