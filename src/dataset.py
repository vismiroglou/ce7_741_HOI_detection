import os
import pandas as pd
import torch
import cv2 as cv
from glob import glob

class FrameDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.img_files = sorted(glob(os.path.join(root, 'clips' + '*/*/*/*.jpg')))
        self.columns = ['id', 'class','x1','y1','x2','y2']

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        annotations_path = img_path.replace('clips', 'annotations').replace('.jpg', '.txt')
        frame = cv.imread(img_path)
        annos = pd.read_csv(annotations_path, names = self.columns, delimiter=' ', usecols= range(6))



        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return frame, annos