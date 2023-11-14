import os
import pandas as pd
import torch
import cv2
from glob import glob

class CropsPytorchDataset(torch.utils.data.Dataset):
    '''
    Oracle mode:
    For training, merge the bounding boxes of the interacting human-object pair.
    Crop the merged bounding box
    Output the crop with the interaction
    '''
    def __init__(self, img_dir:str, anno_file:str, label_encoder, threshold = 0, transform=None, target_transform=None):
        '''
        Expects a single annotation file with only interacting frames + frame directory
        Keeps the frames that are related to annotations.
        '''
        self.annotations = pd.read_csv(anno_file, sep=',')
        self.img_files = []
        for row in self.annotations.iterrows():
            frame_id = row[1]['frame_id']
            #frame = os.path.join(img_dir, str(row[1]['folder_name']), str(row[1]['clip_name']), 'frame_' + f'{frame_id:04}'+'.jpg') # Original
            frame = os.path.join(img_dir, str(row[1]['folder_name']), str(row[1]['clip_name']), 'frame_' + f'{frame_id:04}'+'.jpg')
            self.img_files.append(frame)
        
        self.label_encoder = label_encoder
        self.threshold = threshold
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        #Read img and annotation
        img_path = self.img_files[idx]
        annotations = self.annotations.loc[idx]

        #Create new label and encode it
        label = 'human-'+annotations['action']+'-'+annotations['object_class']
        label = self.label_encoder.transform([label])
        label = torch.tensor(label)

        #Get the merged bounding box and crop the frame around it
        x1 = min(annotations['hmn_x1'],annotations['obj_x1'])
        y1 = min(annotations['hmn_y1'],annotations['obj_y1'])
        x2 = max(annotations['hmn_x2'],annotations['obj_x2'])
        y2 = max(annotations['hmn_y2'],annotations['obj_y2'])

        #Turn frame to tensor. Ready to return. Might need to change if we need temporal info
        crop = cv2.imread(img_path)[y1:y2, x1:x2]#Read crop
        # crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)#Change to single-channel grayscale
        # org = cv2.imread(img_path) # Added original img
        # org = org.astype(float)/255
        # org = torch.tensor(org).type(torch.float)

        if self.transform == 'thresh':
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)#Change to single-channel grayscale
            _, crop = cv2.threshold(crop, 0, 255, cv2.THRESH_TOZERO_INV+cv2.THRESH_OTSU) #apply adaptive threshold
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)#Change to single-channel grayscale
        elif self.transform == 'ada_thresh':
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)#Change to single-channel grayscale
            crop = cv2.adaptiveThreshold(crop, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0) #apply adaptive threshold
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)#Change to single-channel grayscale
        
        crop = crop.astype(float)/255
        crop = crop.transpose((2, 0, 1))
        crop = torch.tensor(crop).type(torch.float)

        # target = {}
        # target['boxes'] = torch.tensor(annos[['x1', 'y1', 'x2', 'y2']].astype(int).to_numpy())
        # target['labels'] = torch.tensor(label)
    
        return crop, label
    
class CropsScikitDataset(torch.utils.data.Dataset):
    '''
    Oracle mode:
    For training, merge the bounding boxes of the interacting human-object pair.
    Crop the merged bounding box
    Output the crop with the interaction
    '''
    def __init__(self, img_dir:str, anno_file:str, label_encoder, threshold = 0, transform=None, target_transform=None):
        '''
        Expects a single annotation file with only interacting frames + frame directory
        Keeps the frames that are related to annotations.
        '''
        self.annotations = pd.read_csv(anno_file, sep=',')
        self.img_files = []
        for row in self.annotations.iterrows():
            frame_id = row[1]['frame_id']
            #frame = os.path.join(img_dir, str(row[1]['folder_name']), str(row[1]['clip_name']), 'frame_' + f'{frame_id:04}'+'.jpg') # Original
            frame = os.path.join(img_dir, str(row[1]['folder_name']), str(row[1]['clip_name']), 'image_' + f'{frame_id:04}'+'.jpg')
            self.img_files.append(frame)
        
        self.label_encoder = label_encoder
        self.threshold = threshold
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        #Read img and annotation
        img_path = self.img_files[idx]
        annotations = self.annotations.loc[idx]

        #Create new label and encode it
        label = 'human-'+annotations['action']+'-'+annotations['object_class']
        label = self.label_encoder.transform([label])
        label = torch.tensor(label)

        #Get the merged bounding box and crop the frame around it
        x1 = torch.tensor(min(annotations['hmn_x1'],annotations['obj_x1']))
        y1 = torch.tensor(min(annotations['hmn_y1'],annotations['obj_y1']))
        x2 = torch.tensor(max(annotations['hmn_x2'],annotations['obj_x2']))
        y2 = torch.tensor(max(annotations['hmn_y2'],annotations['obj_y2']))

        #Turn frame to tensor. Ready to return. Might need to change if we need temporal info
        crop = cv2.imread(img_path)[y1:y2, x1:x2]#Read crop
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)#Change to single-channel grayscale
        org = cv2.imread(img_path) # Added original img
        org = org.astype(float)/255
        org = torch.tensor(org).type(torch.float)

        if self.transform == 'thresh':
            _, crop = cv2.threshold(crop, self.threshold, 255, cv2.THRESH_BINARY) #apply adaptive threshold

        elif self.transform == 'ada_thresh':
            crop = cv2.adaptiveThreshold(crop, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0) #apply adaptive threshold

        
        crop = crop.astype(float)/255
        # crop = crop.transpose((2, 0, 1))
        crop = torch.tensor(crop).type(torch.float)

        #target = {"label": label, "coords": (x1, y1, x2, y2)}
        # target['boxes'] = torch.tensor(annos[['x1', 'y1', 'x2', 'y2']].astype(int).to_numpy())
        # target['labels'] = torch.tensor(label)
    
        return crop, label, (x1, y1, x2, y2)
    

    
class ObjectDetectorDataset(torch.utils.data.Dataset):
    '''
    Merge interacting human-object pairs into a single object with a common bbox.
    Change their label to the triplet <human, verb, object>
    For every non interacting human or object change the label to <class, nointer>
    Classes: human-nointer, bicycle-nointer, motorcycle-nointer, vehicle-nointer
            human-ride-bicycle, human-walk-bicycle, human-ride-motorcycle, human-walk-motorcycle,
    To-add  human-enter-car, human-exit-car, human-park-bicycle
    '''
    
    def __init__(self, root, label_encoder, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.label_encoder = label_encoder
        self.target_transform = target_transform
        self.img_files = sorted(glob(os.path.join(root, 'clips' + '*/*/*.jpg')))
        self.columns = ['id', 'class','x1','y1','x2','y2']

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        #Read img and annotation
        img_path = self.img_files[idx]
        annotations_path = img_path.replace('clips', 'annotations').replace('.jpg', '.txt').replace('frame', 'annotations')
        
        #Turn frame to tensor. Ready to return. Might need to change if we need temporal info
        frame = torch.tensor(cv.imread(img_path))

        #Reading annotations as a DataFrame due to strings. Maybe there is a more optimal way
        annos = pd.read_csv(annotations_path, names = self.columns, delimiter=' ', usecols= range(6))

        target = {}
        target['boxes'] = torch.tensor(annos[['x1', 'y1', 'x2', 'y2']].astype(float).to_numpy())
        target['labels'] = torch.tensor(self.label_encoder.transform(annos['class']))
    
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return frame, target