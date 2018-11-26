import random
import csv
import os
import os.path
from PIL import Image
import glob
import numpy as np
import torch
import torch.utils.data as udata

def preprocess_data(root_dir):
    print('pre-processing data ...\n')
    # training data
    benign    = glob.glob(os.path.join(root_dir, 'Train', 'benign', '*.jpg')); benign.sort()
    malignant = glob.glob(os.path.join(root_dir, 'Train', 'malignant', '*.jpg')); malignant.sort()
    benign_seg    = glob.glob(os.path.join(root_dir, 'Train_Seg', 'benign', '*.png')); benign_seg.sort()
    malignant_seg = glob.glob(os.path.join(root_dir, 'Train_Seg', 'malignant', '*.png')); malignant_seg.sort()
    with open('train.csv', 'wt', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for k in range(len(benign)):
            filename = benign[k]
            filename_seg = benign_seg[k]
            writer.writerow([filename] + [filename_seg] + ['0'])
        for k in range(len(malignant)):
            filename = malignant[k]
            filename_seg = malignant_seg[k]
            writer.writerow([filename] + [filename_seg] + ['1'])
    # training data oversample
    benign    = glob.glob(os.path.join(root_dir, 'Train', 'benign', '*.jpg')); benign.sort()
    malignant = glob.glob(os.path.join(root_dir, 'Train', 'malignant', '*.jpg')); malignant.sort()
    benign_seg    = glob.glob(os.path.join(root_dir, 'Train_Seg', 'benign', '*.png')); benign_seg.sort()
    malignant_seg = glob.glob(os.path.join(root_dir, 'Train_Seg', 'malignant', '*.png')); malignant_seg.sort()
    with open('train_oversample.csv', 'wt', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for k in range(len(benign)):
            filename = benign[k]
            filename_seg = benign_seg[k]
            writer.writerow([filename] + [filename_seg] + ['0'])
        for i in range(4):
            for k in range(len(malignant)):
                filename = malignant[k]
                filename_seg = malignant_seg[k]
                writer.writerow([filename] + [filename_seg] + ['1'])
    # test data
    benign    = glob.glob(os.path.join(root_dir, 'Test', 'benign', '*.jpg')); benign.sort()
    malignant = glob.glob(os.path.join(root_dir, 'Test', 'malignant', '*.jpg')); malignant.sort()
    benign_seg    = glob.glob(os.path.join(root_dir, 'Test_Seg', 'benign', '*.png')); benign_seg.sort()
    malignant_seg = glob.glob(os.path.join(root_dir, 'Test_Seg', 'malignant', '*.png')); malignant_seg.sort()
    with open('test.csv', 'wt', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for k in range(len(benign)):
            filename = benign[k]
            filename_seg = benign_seg[k]
            writer.writerow([filename] + [filename_seg] + ['0'])
        for k in range(len(malignant)):
            filename = malignant[k]
            filename_seg = malignant_seg[k]
            writer.writerow([filename] + [filename_seg] + ['1'])

class ISIC(udata.Dataset):
    def __init__(self, csv_file, shuffle=True, rotate=True, transform=None, transform_seg=None):
        assert not (transform is None) ^ (transform_seg is None), 'is transform is (not) None, transform_seg must also be (not) None!'
        file = open(csv_file, newline='')
        reader = csv.reader(file, delimiter=',')
        self.pairs = [row for row in reader]
        if shuffle:
            random.shuffle(self.pairs)
        self.rotate = rotate
        self.transform = transform
        self.transform_seg = transform_seg
    def __len__(self):
        return len(self.pairs)
    def  __getitem__(self, idx):
        pair = self.pairs[idx]
        image = Image.open(pair[0])
        image_seg = Image.open(pair[1])
        label = int(pair[2])
        # center crop
        width, height = image.size
        new_size = 0.8 * min(width, height)
        left = (width - new_size)/2
        top = (height - new_size)/2
        right = (width + new_size)/2
        bottom = (height + new_size)/2
        image = image.crop((left, top, right, bottom))
        image_seg = image_seg.crop((left, top, right, bottom))
        # rotate
        if self.rotate:
            idx = random.randint(0,3)
            image = image.rotate(idx*90)
            image_seg = image_seg.rotate(idx*90)
        if self.transform:
            image = self.transform(image)
            image_seg = self.transform_seg(image)
        return image, label
