import os
import pandas as pd
import torch
import cv2
from glob import glob
import numpy as np

class CropsPytorchDataset(torch.utils.data.Dataset):
    '''
    Oracle mode:
    For training, merge the bounding boxes of the interacting human-object pair.
    Crop the merged bounding box
    Output the crop with the interaction
    '''
    def __init__(self, img_dir:str, anno_file:str, label_encoder, target_shape=(77,62), threshold = 0, padding=True, transform=None, target_transform=None, find_pairs=False, weights=False):
        '''
        Expects a single annotation file with only interacting frames + frame directory
        Keeps the frames that are related to annotations.
        '''
        #Read the annotations and create the labels
        self.annotations = pd.read_csv(anno_file, sep=',')
        self.annotations['label'] = 'human-'+ self.annotations['action'] + '-' + self.annotations['object_class']
        #Add all img paths to a list. The same frame can exist multiple times if there are multiple interactions in it.
        #Every image corresponds to one row of the dataframe
        self.img_files = []
        for row in self.annotations.iterrows():
            frame_id = row[1]['frame_id']
            frame = os.path.join(img_dir, str(row[1]['folder_name']), str(row[1]['clip_name']), 'frame_' + f'{frame_id:04}'+'.jpg')
            self.img_files.append(frame)
        
        self.find_pairs = find_pairs
        self.label_encoder = label_encoder

        #Transformations
        self.padding = padding
        self.target_shape = target_shape
        self.threshold = threshold
        self.transform = transform
        self.target_transform = target_transform

        if weights:
            self.weights = self.calc_weights()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        #Read img and annotation
        img_path = self.img_files[idx]
        annotations = self.annotations.loc[idx]

        #Create new label and encode it
        label = 'human-'+annotations['action']+'-'+annotations['object_class']
        label = self.label_encoder.transform([label])

        if self.find_pairs:
            #Do not use the annotated human. Instead find the closest human to the object.
            org_anno_path = img_path.replace('clips', 'annotations').replace('frame', 'annotations').replace('.jpg', '.txt')
            obj_id = annotations['object_id']
            hmn_x1, hmn_y1, hmn_x2, hmn_y2 = self.find_pairs_func(org_anno_path, obj_id)

            x1 = min(hmn_x1,annotations['obj_x1'].item())
            y1 = min(hmn_y1,annotations['obj_y1'].item())
            x2 = max(hmn_x2,annotations['obj_x2'].item())
            y2 = max(hmn_y2,annotations['obj_y2'].item())
        
        else:
            #Use the annotated human
            x1 = min(annotations['hmn_x1'],annotations['obj_x1'])
            y1 = min(annotations['hmn_y1'],annotations['obj_y1'])
            x2 = max(annotations['hmn_x2'],annotations['obj_x2'])
            y2 = max(annotations['hmn_y2'],annotations['obj_y2'])
            

        #Turn frame to tensor. Ready to return. Might need to change if we need temporal info
        crop = cv2.imread(img_path)[y1:y2, x1:x2]#Read crop
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)#Change to single-channel grayscale
        
        if self.transform == 'thresh':
            _, crop = cv2.threshold(crop, 0, 255, cv2.THRESH_TOZERO_INV+cv2.THRESH_OTSU) #apply adaptive threshold
        elif self.transform == 'ada_thresh':
            crop = cv2.adaptiveThreshold(crop, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0) #apply adaptive threshold
        
        if self.padding:
                       
            current_shape = crop.shape[:2]
            if current_shape[0] <= self.target_shape[0] and current_shape[1] <= self.target_shape[1]:
                # Pad the crops which shapes dont match the target shape
                crop = np.pad(crop,pad_width=(((self.target_shape[0] - crop.shape[0])//2, (self.target_shape[0] - crop.shape[0] + 1)//2),
                                                    ((self.target_shape[1] - crop.shape[1])//2, (self.target_shape[1] - crop.shape[1] + 1)//2)),
                                                    mode="constant", constant_values=0.0)
            else:
                # If current shape dont match target we re-crop the crop to match
                # Calculate cropping values
                crop_height = min(self.target_shape[0], crop.shape[0]) # Min cropping height value
                crop_width = min(self.target_shape[1], crop.shape[1]) # Min cropping width value

                # Calculate the center starting indices of the crop
                start_height = (crop.shape[0] - crop_height) // 2
                start_width = (crop.shape[1] - crop_width) // 2

                # Center re-crop the original crop
                crop_cropped = crop[start_height:start_height + crop_height, start_width:start_width + crop_width] # Center cropping
        
                # Pad the remaining re-cropped to match target shape
                crop = np.pad(crop_cropped,pad_width=(((self.target_shape[0] - crop_cropped.shape[0])//2, (self.target_shape[0] - crop_cropped.shape[0] + 1)//2),
                                                            ((self.target_shape[1] - crop_cropped.shape[1])//2, (self.target_shape[1] - crop_cropped.shape[1] + 1)//2)),
                                                            mode="constant", constant_values=0.0)
        
        crop = crop.astype(float)/255
        crop = torch.tensor(crop).type(torch.float)
        crop = crop.unsqueeze(0)
    
        return crop, label
    
    def find_pairs_func(self, anno_path, obj_id):
        import math
        annotations = pd.read_csv(anno_path, sep=' ', header=None)
        min_distance = float('inf')
        min_distance_id = 0
        obj = annotations[annotations[0] == obj_id].index
        xc_o, yc_o = self.find_center(annotations.loc[obj, 2].item(), 
                                        annotations.loc[obj, 3].item(), 
                                        annotations.loc[obj, 4].item(), 
                                        annotations.loc[obj, 5].item())
        
        for idx in annotations.index.drop(obj):
            if annotations.loc[idx, 1] == 'human':
                xc_h, yc_h = self.find_center(annotations.loc[idx, 2].item(), 
                                            annotations.loc[idx, 3].item(), 
                                            annotations.loc[idx, 4].item(), 
                                            annotations.loc[idx, 5].item())
                
                distance = math.dist([xc_o, yc_o],[xc_h, yc_h])
                if distance < min_distance:
                    min_distance = distance
                    min_distance_id = annotations.loc[idx, 0]
        hmn = annotations[annotations[0] == min_distance_id].index
        x1_h, y1_h, x2_h, y2_h = annotations.iloc[hmn, 2].item(), annotations.iloc[hmn, 3].item(), annotations.iloc[hmn, 4].item(), annotations.iloc[hmn, 5].item()
        return(x1_h, y1_h, x2_h, y2_h)
        
    def find_center(self, x1, y1, x2, y2):
        xc = int((x1 + x2)/2)
        yc = int((y1 + y2)/2)
        return xc, yc
    
    def calc_weights(self):
        

    
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
    
        return crop, label, (x1, y1, x2, y2), org
    

class PairCropsScikitDataset(torch.utils.data.Dataset):
    '''
    Oracle mode:
    For training, merge the bounding boxes of the interacting human-object pair.
    Crop the merged bounding box
    Output the crop with the interaction
    '''
    def __init__(self, img_dir:str, anno_file:str, label_encoder, target_shape=(77,62), threshold = 0, padding=True, transform=None, target_transform=None, find_pairs=False):
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
        self.padding = padding
        self.target_shape = target_shape
        self.find_pairs = find_pairs
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
        
        if self.find_pairs:
            #Do not use the annotated human. Instead find the closest human to the object.
            org_anno_path = img_path.replace('clips', 'annotations').replace('images', 'annotations').replace('image', 'annotations').replace('.jpg', '.txt')
            obj_id = annotations['object_id']
            hmn_x1, hmn_y1, hmn_x2, hmn_y2 = self.find_pairs_func(org_anno_path, obj_id)

            x1 = min(hmn_x1,annotations['obj_x1'].item())
            y1 = min(hmn_y1,annotations['obj_y1'].item())
            x2 = max(hmn_x2,annotations['obj_x2'].item())
            y2 = max(hmn_y2,annotations['obj_y2'].item())
        
        else:
            #Use the annotated human
            x1 = min(annotations['hmn_x1'],annotations['obj_x1'])
            y1 = min(annotations['hmn_y1'],annotations['obj_y1'])
            x2 = max(annotations['hmn_x2'],annotations['obj_x2'])
            y2 = max(annotations['hmn_y2'],annotations['obj_y2'])
            
            

        #Turn frame to tensor. Ready to return. Might need to change if we need temporal info
        coords = (x1,y1,x2,y2)
        crop = cv2.imread(img_path)[y1:y2, x1:x2]#Read crop
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)#Change to single-channel grayscale
        
        if self.transform == 'thresh':
            _, crop = cv2.threshold(crop, 0, 255, cv2.THRESH_TOZERO_INV+cv2.THRESH_OTSU) #apply adaptive threshold
        elif self.transform == 'ada_thresh':
            crop = cv2.adaptiveThreshold(crop, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0) #apply adaptive threshold
            
        if self.padding:
            current_shape = crop.shape[:2]
            if current_shape[0] <= self.target_shape[0] and current_shape[1] <= self.target_shape[1]:
                # Pad the crops which shapes dont match the target shape
                crop = np.pad(crop,pad_width=(((self.target_shape[0] - crop.shape[0])//2, (self.target_shape[0] - crop.shape[0] + 1)//2),
                                                    ((self.target_shape[1] - crop.shape[1])//2, (self.target_shape[1] - crop.shape[1] + 1)//2)),
                                                    mode="constant", constant_values=0.0)
            else:
                # If current shape dont match target we re-crop the crop to match
                # Calculate cropping values
                crop_height = min(self.target_shape[0], crop.shape[0]) # Min cropping height value
                crop_width = min(self.target_shape[1], crop.shape[1]) # Min cropping width value

                # Calculate the center starting indices of the crop
                start_height = (crop.shape[0] - crop_height) // 2
                start_width = (crop.shape[1] - crop_width) // 2

                # Center re-crop the original crop
                crop_cropped = crop[start_height:start_height + crop_height, start_width:start_width + crop_width] # Center cropping
        
                # Pad the remaining re-cropped to match target shape
                crop = np.pad(crop_cropped,pad_width=(((self.target_shape[0] - crop_cropped.shape[0])//2, 
                                                       (self.target_shape[0] - crop_cropped.shape[0] + 1)//2),
                                                      ((self.target_shape[1] - crop_cropped.shape[1])//2, 
                                                       (self.target_shape[1] - crop_cropped.shape[1] + 1)//2)),
                                                        mode="constant", constant_values=0.0)
        
        crop = crop.astype(float)/255
        crop = torch.tensor(crop).type(torch.float)
        crop = crop.unsqueeze(0)
    
        return crop, coords, label
    
    def find_pairs_func(self, anno_path, obj_id):
        import math
        annotations = pd.read_csv(anno_path, sep=' ', header=None)
        min_distance = float('inf')
        min_distance_id = 0
        obj = annotations[annotations[0] == obj_id].index
        xc_o, yc_o = self.find_center(annotations.loc[obj, 2].item(), 
                                        annotations.loc[obj, 3].item(), 
                                        annotations.loc[obj, 4].item(), 
                                        annotations.loc[obj, 5].item())
        
        for idx in annotations.index.drop(obj):
            if annotations.loc[idx, 1] == 'human':
                xc_h, yc_h = self.find_center(annotations.loc[idx, 2].item(), 
                                            annotations.loc[idx, 3].item(), 
                                            annotations.loc[idx, 4].item(), 
                                            annotations.loc[idx, 5].item())
                
                distance = math.dist([xc_o, yc_o],[xc_h, yc_h])
                if distance < min_distance:
                    min_distance = distance
                    min_distance_id = annotations.loc[idx, 0]
        hmn = annotations[annotations[0] == min_distance_id].index
        x1_h, y1_h, x2_h, y2_h = annotations.iloc[hmn, 2].item(), annotations.iloc[hmn, 3].item(), annotations.iloc[hmn, 4].item(), annotations.iloc[hmn, 5].item()
        return(x1_h, y1_h, x2_h, y2_h)
      
        
    def find_center(self, x1, y1, x2, y2):
        xc = int((x1 + x2)/2)
        yc = int((y1 + y2)/2)
        return xc, yc   
    
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