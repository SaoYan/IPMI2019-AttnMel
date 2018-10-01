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
    melanoma = glob.glob(os.path.join(root_dir, 'Train', 'melanoma', '*.jpg'))
    nevus    = glob.glob(os.path.join(root_dir, 'Train', 'nevus', '*.jpg'))
    sk    = glob.glob(os.path.join(root_dir, 'Train', 'seborrheic_keratosis', '*.jpg'))
    with open('train.csv', 'wt', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for filename in melanoma:
            writer.writerow([filename] + ['1'])
        for filename in nevus:
            writer.writerow([filename] + ['0'])
        # for filename in sk:
        #     writer.writerow([filename] + ['0'])
    # training data oversample
    melanoma = glob.glob(os.path.join(root_dir, 'Train', 'melanoma', '*.jpg'))
    nevus    = glob.glob(os.path.join(root_dir, 'Train', 'nevus', '*.jpg'))
    sk    = glob.glob(os.path.join(root_dir, 'Train', 'seborrheic_keratosis', '*.jpg'))
    with open('train_oversample.csv', 'wt', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for i in range(4):
            for filename in melanoma:
                writer.writerow([filename] + ['1'])
        for filename in nevus:
            writer.writerow([filename] + ['0'])
        # for filename in sk:
        #     writer.writerow([filename] + ['0'])
    # test data
    melanoma = glob.glob(os.path.join(root_dir, 'Test', 'melanoma', '*.jpg'))
    nevus    = glob.glob(os.path.join(root_dir, 'Test', 'nevus', '*.jpg'))
    sk    = glob.glob(os.path.join(root_dir, 'Test', 'seborrheic_keratosis', '*.jpg'))
    with open('test.csv', 'wt', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for filename in melanoma:
            writer.writerow([filename] + ['1'])
        for filename in nevus:
            writer.writerow([filename] + ['0'])
        # for filename in sk:
        #     writer.writerow([filename] + ['0'])

class ISIC2017(udata.Dataset):
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
        # center crop
        width, height = image.size
        new_size = 0.8 * min(width, height)
        left = (width - new_size)/2
        top = (height - new_size)/2
        right = (width + new_size)/2
        bottom = (height + new_size)/2
        image = image.crop((left, top, right, bottom))
        if self.transform:
            image = self.transform(image)
        return image, label
