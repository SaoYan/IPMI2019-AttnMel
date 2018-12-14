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
    MEL   = glob.glob(os.path.join(root_dir, 'Train', 'MEL', '*.jpg')); MEL.sort()
    NV    = glob.glob(os.path.join(root_dir, 'Train', 'NV', '*.jpg')); NV.sort()
    BCC   = glob.glob(os.path.join(root_dir, 'Train', 'BCC', '*.jpg')); BCC.sort()
    AKIEC = glob.glob(os.path.join(root_dir, 'Train', 'AKIEC', '*.jpg')); AKIEC.sort()
    BKL   = glob.glob(os.path.join(root_dir, 'Train', 'BKL', '*.jpg')); BKL.sort()
    DF    = glob.glob(os.path.join(root_dir, 'Train', 'DF', '*.jpg')); DF.sort()
    VASC  = glob.glob(os.path.join(root_dir, 'Train', 'VASC', '*.jpg')); VASC.sort()
    with open('train.csv', 'wt', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for filename in MEL:
            writer.writerow([filename] + ['0'])
        for filename in NV:
            writer.writerow([filename] + ['1'])
        for filename in BCC:
            writer.writerow([filename] + ['2'])
        for filename in AKIEC:
            writer.writerow([filename] + ['3'])
        for filename in BKL:
            writer.writerow([filename] + ['4'])
        for filename in DF:
            writer.writerow([filename] + ['5'])
        for filename in VASC:
            writer.writerow([filename] + ['6'])
    # training data oversample
    MEL   = glob.glob(os.path.join(root_dir, 'Train', 'MEL', '*.jpg')); MEL.sort()
    NV    = glob.glob(os.path.join(root_dir, 'Train', 'NV', '*.jpg')); NV.sort()
    BCC   = glob.glob(os.path.join(root_dir, 'Train', 'BCC', '*.jpg')); BCC.sort()
    AKIEC = glob.glob(os.path.join(root_dir, 'Train', 'AKIEC', '*.jpg')); AKIEC.sort()
    BKL   = glob.glob(os.path.join(root_dir, 'Train', 'BKL', '*.jpg')); BKL.sort()
    DF    = glob.glob(os.path.join(root_dir, 'Train', 'DF', '*.jpg')); DF.sort()
    VASC  = glob.glob(os.path.join(root_dir, 'Train', 'VASC', '*.jpg')); VASC.sort()
    with open('train_oversample.csv', 'wt', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for i in range(6):
            for filename in MEL:
                writer.writerow([filename] + ['0'])
        for filename in NV:
            writer.writerow([filename] + ['1'])
        for i in range(13):
            for filename in BCC:
                writer.writerow([filename] + ['2'])
        for i in range(20):
            for filename in AKIEC:
                writer.writerow([filename] + ['3'])
        for i in range(6):
            for filename in BKL:
                writer.writerow([filename] + ['4'])
        for i in range(58):
            for filename in DF:
                writer.writerow([filename] + ['5'])
        for i in range(47):
            for filename in VASC:
                writer.writerow([filename] + ['6'])
    # test data
    MEL   = glob.glob(os.path.join(root_dir, 'Test', 'MEL', '*.jpg')); MEL.sort()
    NV    = glob.glob(os.path.join(root_dir, 'Test', 'NV', '*.jpg')); NV.sort()
    BCC   = glob.glob(os.path.join(root_dir, 'Test', 'BCC', '*.jpg')); BCC.sort()
    AKIEC = glob.glob(os.path.join(root_dir, 'Test', 'AKIEC', '*.jpg')); AKIEC.sort()
    BKL   = glob.glob(os.path.join(root_dir, 'Test', 'BKL', '*.jpg')); BKL.sort()
    DF    = glob.glob(os.path.join(root_dir, 'Test', 'DF', '*.jpg')); DF.sort()
    VASC  = glob.glob(os.path.join(root_dir, 'Test', 'VASC', '*.jpg')); VASC.sort()
    with open('test.csv', 'wt', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for filename in MEL:
            writer.writerow([filename] + ['0'])
        for filename in NV:
            writer.writerow([filename] + ['1'])
        for filename in BCC:
            writer.writerow([filename] + ['2'])
        for filename in AKIEC:
            writer.writerow([filename] + ['3'])
        for filename in BKL:
            writer.writerow([filename] + ['4'])
        for filename in DF:
            writer.writerow([filename] + ['5'])
        for filename in VASC:
            writer.writerow([filename] + ['6'])

class ISIC2018(udata.Dataset):
    def __init__(self, csv_file, shuffle=True, transform=None):
        file = open(csv_file, newline='')
        reader = csv.reader(file, delimiter=',')
        self.pairs = [row for row in reader]
        if shuffle:
            random.shuffle(self.pairs)
        self.transform = transform
    def __len__(self):
        return len(self.pairs)
    def  __getitem__(self, idx):
        pair = self.pairs[idx]
        image = Image.open(pair[0])
        label = int(pair[1])
        # construct one sample
        sample = {'image': image, 'image_seg': image, 'label': label}
        # transform
        if self.transform:
            sample = self.transform(sample)
        return sample
