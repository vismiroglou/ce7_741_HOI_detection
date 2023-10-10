import os
import pandas as pd
import torch
import cv2 as cv

class thermal_data(torch.utils.data.Dataset):
    def __init__(self, annotations_dir, transform=None, target_transform=None):
        self.img_labels = 
        self.clip_dir = 
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        frame = cv.imread(self.clip_dir)
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label