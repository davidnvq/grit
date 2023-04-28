import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
from pathlib import Path
from collections import Counter
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader, DistributedSampler, BatchSampler
from engine.utils import nested_tensor_from_tensor_list

import torchvision.transforms as T
from collections import defaultdict
from datasets.caption.transforms import *

spacy_eng = spacy.load('en_core_web_sm')

train_image_path = Path("/media/local_workspace/quang/datasets/vietcap/train-images")
test_image_path = Path("/media/local_workspace/quang/datasets/vietcap/public-test-images")
train_caption_path = "/media/local_workspace/quang/datasets/vietcap/thesis_test/train_data.json"
test_caption_path = "/media/local_workspace/quang/datasets/vietcap/thesis_test/test_data.json"
vi_caption_path = "/media/local_workspace/quang/datasets/vietcap/vi_captions.json"


def get_transform(resize_name="maxwh", size=[384, 640], randaug=False):
    resize = RESIZE[resize_name](size)
    if randaug:
        return {
            'train': Compose([resize, RandAugment(), ToTensor(), normalize()]),
            'valid': Compose([resize, ToTensor(), normalize()]),
        }
    else:
        return {
            'train': Compose([resize, ToTensor(), normalize()]),
            'valid': Compose([resize, ToTensor(), normalize()]),
        }


class Vocabulary:

    def __init__(self, freq_threshold):
        self.itos = {0: "<unk>", 1: "<pad>", 2: "<bos>", 3: "<eos>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize(text):
        return [token.text.lower() for token in spacy_eng.tokenizer(text)]

    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenize(text)
        return [self.stoi[token] if token in self.stoi else self.stoi["<unk>"] for token in tokenized_text]


class CustomDataset(Dataset):

    def __init__(self, root_dir, captions_file, vicap_file=vi_caption_path, transform=None, freq_threshold=1):
        self.root_dir = root_dir
        self.captions_file = captions_file
        self.data = json.load(open(captions_file, "r"))
        self.transform = transform
        self.imgid2imgname = {entry['id']: entry['filename'] for entry in self.data['images']}
        self.captions = [ann['segment_caption'] for ann in self.data['annotations']]

        all_captions = json.load(open(vicap_file, "r"))
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(all_captions)

    def __len__(self):
        return len(self.data['annotations'])

    def __getitem__(self, idx):

        annotation = self.data['annotations'][idx]
        caption = annotation['segment_caption']
        image_id = annotation["image_id"]
        image_name = self.imgid2imgname[image_id]
        image = Image.open(os.path.join(self.root_dir, image_name)).convert('RGB')

        if self.transform is not None:
            img = self.transform(image)

        caption_vec = []
        caption_vec += [self.vocab.stoi["<bos>"]]
        caption_vec += self.vocab.numericalize(caption)
        caption_vec += [self.vocab.stoi["<eos>"]]
        return img, torch.tensor(caption_vec), caption


class DictDataset(CustomDataset):

    def __init__(self, root_dir, captions_file, vicap_file=vi_caption_path, transform=None, freq_threshold=1):
        super().__init__(root_dir, captions_file, vicap_file, transform, freq_threshold)
        self.img_id_2_captions = self.img_id_2_captions()
        self.img_ids = list(self.img_id_2_captions.keys())

    def img_id_2_captions(self):
        img_id_2_captions = defaultdict(list)
        for ann in self.data['annotations']:
            img_id_2_captions[ann['image_id']].append(" ".join(self.vocab.tokenize(ann['segment_caption'])))
        return img_id_2_captions

    def __len__(self):
        return len(self.img_id_2_captions)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        image_name = self.imgid2imgname[img_id]
        img = Image.open(os.path.join(self.root_dir, image_name)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        captions = self.img_id_2_captions[img_id]
        return img, captions, img_id


class EvalCollate:

    def __init__(self, pad_idx, batch_first=True, device="cuda"):
        self.pad_idx = pad_idx
        self.batch_first = batch_first
        self.device = device

    def __call__(self, batch):
        imgs = [item[0] for item in batch]
        imgs = nested_tensor_from_tensor_list(imgs).to(self.device)

        captions = [item[1] for item in batch]
        image_ids = [item[2] for item in batch]
        return {'samples': imgs, 'captions': captions, 'image_ids': image_ids}


class CapsCollate:

    def __init__(self, pad_idx, batch_first=True, device="cuda"):
        self.pad_idx = pad_idx
        self.batch_first = batch_first
        self.device = device

    def __call__(self, batch):
        imgs = [item[0] for item in batch]
        imgs = nested_tensor_from_tensor_list(imgs).to(self.device)

        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=self.batch_first, padding_value=self.pad_idx)
        targets = targets.to(self.device)
        return {'samples': imgs, 'captions': targets}


def get_datasets():
    transforms = get_transform(resize_name="maxwh", size=[384, 640], randaug=False)

    train_dataset = CustomDataset(
        root_dir=train_image_path,
        captions_file=train_caption_path,
        vicap_file=vi_caption_path,
        transform=transforms['train'],
    )
    valid_dataset = CustomDataset(
        root_dir=test_image_path,
        captions_file=test_caption_path,
        vicap_file=vi_caption_path,
        transform=transforms['valid'],
    )

    train_dict_dataset = DictDataset(
        root_dir=train_image_path,
        captions_file=train_caption_path,
        vicap_file=vi_caption_path,
        transform=transforms['train'],
    )
    valid_dict_dataset = DictDataset(
        root_dir=test_image_path,
        captions_file=test_caption_path,
        vicap_file=vi_caption_path,
        transform=transforms['valid'],
    )
    return {
        'train': train_dataset,
        'valid': valid_dataset,
        'train_dict': train_dict_dataset,
        'valid_dict': valid_dict_dataset,
    }


def get_dataloaders(device="cuda", batch_size=8):
    datasets = get_datasets()

    collators = {
        'train': CapsCollate(datasets['train'].vocab.stoi['<pad>'], device=device),
        'valid': CapsCollate(datasets['valid'].vocab.stoi['<pad>'], device=device),
        'train_dict': EvalCollate(datasets['train_dict'].vocab.stoi['<pad>'], device=device),
        'valid_dict': EvalCollate(datasets['valid_dict'].vocab.stoi['<pad>'], device=device)
    }

    samplers = {
        'train': DistributedSampler(datasets['train'], shuffle=True),
        'valid': DistributedSampler(datasets['valid'], shuffle=False),
        'train_dict': DistributedSampler(datasets['train_dict'], shuffle=True),
        'valid_dict': DistributedSampler(datasets['valid_dict'], shuffle=False)
    }

    dataloaders = {k: DataLoader(datasets[k], batch_size=batch_size, num_workers=4, collate_fn=collators[k], sampler=samplers[k]) for k in datasets}
    return samplers, dataloaders


if __name__ == "__main__":

    # get all captions to build vocabulary
    train_data = json.load(open(train_caption_path))
    test_data = json.load(open(test_caption_path))

    train_captions = [ann['segment_caption'] for ann in train_data['annotations']]
    test_captions = [ann['segment_caption'] for ann in test_data['annotations']]
    all_captions = train_captions + test_captions

    with open(vi_caption_path, 'w', encoding='utf-8') as f:
        json.dump(all_captions, f, indent=4, ensure_ascii=False)

    # build datasets
    datasets = get_datasets()
    train_dataset, test_dataset = datasets['train'], datasets['valid']

    # train_dataset
    img, target, caption = train_dataset[0]
    print(img.shape)
    print(target.shape)
    print(" ".join([train_dataset.vocab.itos[token] for token in target.numpy()]))
    print(caption)

    # test_dataset
    img, captions, img_id = test_dataset[0]
    print(img.shape)
    print(captions)
    print(img_id)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=CapsCollate(train_dataset.vocab.stoi["<pad>"]))
    valid_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=EvalCollate(train_dataset.vocab.stoi["<pad>"]))

    dataloaders = {'train': train_loader, 'valid': valid_loader}
    vocab = train_dataset.vocab

    for batch in valid_loader:
        imgs, captions = batch['samples'], batch['captions']
        print(imgs.tensors.shape)
        print(captions)
        break

    for batch in train_loader:
        imgs, captions = batch['samples'], batch['captions']
        print(imgs.tensors.shape)
        print(captions.shape)
        break
