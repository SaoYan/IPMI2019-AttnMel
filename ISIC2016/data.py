import random
import csv
import os
import os.path
from PIL import Image
import glob
import numpy as np
import torch
import torch.utils.data as udata

random.seed(10000)

def preprocess_data(root_dir):
    print('pre-processing data ...\n')
    # training data
    benign    = glob.glob(os.path.join(root_dir, 'Train', 'benign', '*.jpg'))
    malignant = glob.glob(os.path.join(root_dir, 'Train', 'malignant', '*.jpg'))
    with open('train.csv', 'wt', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for filename in benign:
            writer.writerow([filename] + ['0'])
        for filename in malignant:
            writer.writerow([filename] + ['1'])
    # training data oversample
    benign    = glob.glob(os.path.join(root_dir, 'Train', 'benign', '*.jpg'))
    malignant = glob.glob(os.path.join(root_dir, 'Train', 'malignant', '*.jpg'))
    with open('train_oversample.csv', 'wt', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for filename in benign:
            writer.writerow([filename] + ['0'])
        for i in range(4):
            for filename in malignant:
                writer.writerow([filename] + ['1'])
    # test data
    benign    = glob.glob(os.path.join(root_dir, 'Test', 'benign', '*.jpg'))
    malignant = glob.glob(os.path.join(root_dir, 'Test', 'malignant', '*.jpg'))
    with open('test.csv', 'wt', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for filename in benign:
            writer.writerow([filename] + ['0'])
        for filename in malignant:
            writer.writerow([filename] + ['1'])

class ISIC2016(udata.Dataset):
    def __init__(self, csv_file, shuffle=True, rotate=True, transform=None):
        file = open(csv_file, newline='')
        reader = csv.reader(file, delimiter=',')
        self.pairs = [row for row in reader]
        if shuffle:
            random.shuffle(self.pairs)
        self.rotate = rotate
        self.transform = transform
    def __len__(self):
        return len(self.pairs)
    def  __getitem__(self, idx):
        pair = self.pairs[idx]
        image = Image.open(pair[0])
        label = int(pair[1])
        # center crop
        width, height = image.size
        new_size = 0.8 * min(width, height)
        left = (width - new_size)/2
        top = (height - new_size)/2
        right = (width + new_size)/2
        bottom = (height + new_size)/2
        image = image.crop((left, top, right, bottom))
        # rotate
        if self.rotate:
            idx = random.randint(0,3)
            image = image.rotate(idx*90)
        if self.transform:
            image = self.transform(image)
        return image, label
