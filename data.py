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
    NV    = glob.glob(os.path.join(root_dir, 'Train', 'NV', '*.jpg'))
    MEL   = glob.glob(os.path.join(root_dir, 'Train', 'MEL', '*.jpg'))
    BKL   = glob.glob(os.path.join(root_dir, 'Train', 'BKL', '*.jpg'))
    BCC   = glob.glob(os.path.join(root_dir, 'Train', 'BCC', '*.jpg'))
    AKIEC = glob.glob(os.path.join(root_dir, 'Train', 'AKIEC', '*.jpg'))
    VASC  = glob.glob(os.path.join(root_dir, 'Train', 'VASC', '*.jpg'))
    DF    = glob.glob(os.path.join(root_dir, 'Train', 'DF', '*.jpg'))
    with open('train.csv', 'wt', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for filename in NV:
            writer.writerow([filename] + ['0'])
        for i in range(6):
            for filename in MEL:
                writer.writerow([filename] + ['1'])
        for i in range(6):
            for filename in BKL:
                writer.writerow([filename] + ['2'])
        for i in range(13):
            for filename in BCC:
                writer.writerow([filename] + ['3'])
        for i in range(20):
            for filename in AKIEC:
                writer.writerow([filename] + ['4'])
        for i in range(47):
            for filename in VASC:
                writer.writerow([filename] + ['5'])
        for i in range(58):
            for filename in DF:
                writer.writerow([filename] + ['6'])
    # test data
    NV    = glob.glob(os.path.join(root_dir, 'Test', 'NV', '*.jpg'))
    MEL   = glob.glob(os.path.join(root_dir, 'Test', 'MEL', '*.jpg'))
    BKL   = glob.glob(os.path.join(root_dir, 'Test', 'BKL', '*.jpg'))
    BCC   = glob.glob(os.path.join(root_dir, 'Test', 'BCC', '*.jpg'))
    AKIEC = glob.glob(os.path.join(root_dir, 'Test', 'AKIEC', '*.jpg'))
    VASC  = glob.glob(os.path.join(root_dir, 'Test', 'VASC', '*.jpg'))
    DF    = glob.glob(os.path.join(root_dir, 'Test', 'DF', '*.jpg'))
    with open('test.csv', 'wt', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for filename in NV:
            writer.writerow([filename] + ['0'])
        for filename in MEL:
            writer.writerow([filename] + ['1'])
        for filename in BKL:
            writer.writerow([filename] + ['2'])
        for filename in BCC:
            writer.writerow([filename] + ['3'])
        for filename in AKIEC:
            writer.writerow([filename] + ['4'])
        for filename in VASC:
            writer.writerow([filename] + ['5'])
        for filename in DF:
            writer.writerow([filename] + ['6'])

class ISIC2018(udata.Dataset):
    def __init__(self, csv_file, shuffle=False, transform=None):
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
        if self.transform:
            image = self.transform(image)
        return image, label
