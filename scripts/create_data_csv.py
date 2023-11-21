from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append('../src/')
from utils import collate_fn
from dataset import CropsScikitDataset, PairCropsScikitDataset

from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import os

def preprocess(data: str, processing: str, pairs: bool, target_shape: tuple):
    # Separate features and labels in separate lists
    features, labels = zip(*data)
    labels = np.array(labels).reshape(-1)
    
    # Check whether it should process data as crops or coordinates
    if processing == "coords":
    # Uncomment for coordinates, else for crops
    #print("original",features)
        features = [coord[0] for coord in features]
        features = np.array(features)
        return features, labels
    elif processing == "crops":
        # Converts each img tensor to numpy arrays
        features = [image[0].numpy() for image in tqdm(features,total=len(features),desc="Converting to np array")]

        # We reshape the img from a 3D to 2D array
        features = [image.reshape(image.shape[0],-1) for image in tqdm(features,total=len(features),desc="Reshaping 3D to 2D")]

        """
        # Find the maximum height and width among all crops
        max_height = max(img.shape[0] for img in tqdm(features,total=len(features),desc="Finding max height"))
        max_width = max(img.shape[1] for img in tqdm(features,total=len(features),desc="Finding max width"))
        print(f"({max_height},{max_width})")
        
        # Former padding function
        Add centered padding around the crop to match max height and width
        To counter floor divison we add +1
        features = [np.pad(img,pad_width=(((max_height - img.shape[0])//2, (max_height - img.shape[0] + 1)//2),
                                        ((max_width - img.shape[1])//2, (max_width - img.shape[1] + 1)//2)),
                        mode="constant", constant_values=0.0) for img in tqdm(features,total=len(features),desc="Applying padding")]

        features = np.array(features)
        samples,nx,ny = features.shape
        features = features.reshape(samples,nx*ny)
        """
        
        # Checks if it should process as pairs
        if pairs == False:
            # Re-crop/pad crops to match target shape
            padded_crops = []
            for crop in features:
                current_shape = crop.shape[:2]
                
                if current_shape[0] <= target_shape[0] and current_shape[1] <= target_shape[1]:
                    # Pad the crops which shapes dont match the target shape
                    crop_padded = np.pad(crop,pad_width=(((target_shape[0] - crop.shape[0])//2, (target_shape[0] - crop.shape[0] + 1)//2),
                                                        ((target_shape[1] - crop.shape[1])//2, (target_shape[1] - crop.shape[1] + 1)//2)),
                                                        mode="constant", constant_values=0.0)
                    padded_crops.append(crop_padded)
                else:
                    # If current shape dont match target we re-crop the crop to match
                    # Calculate cropping values
                    crop_height = min(target_shape[0], crop.shape[0]) # Min cropping height value
                    crop_width = min(target_shape[1], crop.shape[1]) # Min cropping width value

                    # Calculate the center starting indices of the crop
                    start_height = (crop.shape[0] - crop_height) // 2
                    start_width = (crop.shape[1] - crop_width) // 2

                    # Center re-crop the original crop
                    crop_cropped = crop[start_height:start_height + crop_height, start_width:start_width + crop_width] # Center cropping
                    #crop_cropped2 = crop[:crop_height, :crop_width] # Top left cropping
            
                    # Pad the remaining re-cropped to match target shape
                    crop_padded = np.pad(crop_cropped,pad_width=(((target_shape[0] - crop_cropped.shape[0])//2, (target_shape[0] - crop_cropped.shape[0] + 1)//2),
                                                                ((target_shape[1] - crop_cropped.shape[1])//2, (target_shape[1] - crop_cropped.shape[1] + 1)//2)),
                                                                mode="constant", constant_values=0.0)
                    # Visualize
                    #plt.subplot(1,4,1)
                    #plt.title("Before re-cropping")
                    #plt.imshow(crop, cmap='gray', aspect='auto')
                    #plt.subplot(1,4,2)
                    #plt.title("After re-cropping (center)")
                    #plt.imshow(crop_cropped, cmap='gray', aspect='auto')
                    #plt.subplot(1,4,3)
                    #plt.title("After re-cropping (top)")
                    #plt.imshow(crop_cropped2, cmap='gray', aspect='auto')
                    #plt.subplot(1,4,4)
                    #plt.title("After padding re-cropped crop")
                    #plt.imshow(crop_padded, cmap='gray', aspect='auto')
                    #plt.tight_layout()
                    #plt.show()
                    
                    padded_crops.append(crop_padded)
                    
            # Reshape 3D to 2D array and return it
            features = np.array(padded_crops)
            samples,nx,ny = features.shape
            features = features.reshape(samples,nx*ny)
            
            return features, labels
        
        elif pairs == True:
            # Reshape 3D to 2D array and return it
            features = np.array(features)
            samples,nx,ny = features.shape
            features = features.reshape(samples,nx*ny)
            
            return features, labels

def to_csv(data: str, save_path: str, processing: str, pairs: bool, target_shape: tuple):
    # Preprocess data
    features, labels = preprocess(data,processing, pairs, target_shape)
    
    # Create dataframe of features and labels
    if processing == "crops":
        features_df = pd.DataFrame(features, columns=[f'pixel_{i}' for i in range(features.shape[1])])
    elif processing == "coords":
        features_df = pd.DataFrame(features, columns=['x1','y1','x2','y2']) # Only for coordinates
    labels_df = pd.DataFrame(labels, columns=['label'])

    # Concatenate features and labels along columns
    df = pd.concat([features_df, labels_df], axis=1)

    # Save to CSV
    print("Saving to csv...")
    df.to_csv(save_path, index=False)
    #print(df)

def create_dataset_csv(anno_path: str, img_path: str, save_to_path: str, target_shape: tuple, processing: str, pairs: bool, le):
    # Set seed for reproducibility
    torch.manual_seed(0)
    
    # Intialize custom dataset and datasetloader
    if pairs == False:
        dataset = CropsScikitDataset(anno_file=anno_path, img_dir = img_path, label_encoder=le)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
        
        # Load all the data into memory (HIGHLY INEFFICIENT!)
        data = []
        start_time = time.time()
        if processing == "crops":
            for crop, _, label in tqdm(dataloader,total=len(dataloader),leave=True,desc="Loading into memory"):
                    data.append((crop,label))
            end_time = time.time()
            print(f"Pairs: False\nProcessing: Crops\nTime: {end_time - start_time}")
        elif processing == "coords":
            for _, coord, label in tqdm(dataloader,total=len(dataloader),leave=True,desc="Loading into memory"):
                    data.append((coord,label))
            end_time = time.time()
            print(f"Pairs: False\nProcessing: Coords\nTime: {end_time - start_time}")
                    
    elif pairs == True:
        dataset = PairCropsScikitDataset(anno_file=anno_path, img_dir = img_path, label_encoder=le, find_pairs=True)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
        
        # Load all the data into memory (HIGHLY INEFFICIENT!)
        data = []
        start_time = time.time()
        if processing == "crops":
            for crop, _, label in tqdm(dataloader,total=len(dataloader),leave=True,desc="Loading into memory"):
                    data.append((crop,label))
            end_time = time.time()
            print(f"Pairs: True\nProcessing: Crops\nTime: {end_time - start_time}")
        elif processing == "coords":
            for _, coord, label in tqdm(dataloader,total=len(dataloader),leave=True,desc="Loading into memory"):
                    data.append((coord,label))
            end_time = time.time()
            print(f"Pairs: True\nProcessing: Coords\nTime: {end_time - start_time}")

    

    # Save data to csv
    to_csv(data, save_to_path, processing, pairs, target_shape)
    print(f"Finished saving '{processing}' as {os.path.basename(save_to_path)}!\n")
        
if __name__ == '__main__':
    # Setting up label encoder
    le = LabelEncoder()
    labels = ['human-hold-bicycle', 'human-ride-bicycle', 'human-ride-motorcycle', 'human-walk-bicycle', 'human-walk-motorcycle', 'human-hold-motorcycle']
    #labels = ['human-hold-bicycle', 'human-ride-bicycle', 'human-ride-motorcycle', 'human-walk-bicycle', 'human-walk-motorcycle']
    #labels = ['human-hold-bicycle', 'human-park-bicycle', 'human-park-motorcycle', 'human-pickup-bicycle', 'human-pickup-motorcycle', 'human-ride-bicycle', 'human-ride-motorcycle', 'human-ride-vehicle', 'human-walk-bicycle', 'human-walk-motorcycle', 'human-hold-motorcycle']
    le.fit(labels)

    # Initialize images path
    img_path = r'../../dataset/images'
    
    # Initialize dataset and dataloader for training/testing dataset
    # For crops
    anno_path_train = r'../data_anno/training_data.csv'
    save_to_path_train = r"../data_anno/training_data_pair_crops.csv"
    anno_path_test = r'../data_anno/testing_data.csv'
    save_to_path_test = r"../data_anno/testing_data_pair_crops.csv"
    
    # Create dataset csv for crops
    print("Creating dataset for crops...")
    #create_dataset_csv(anno_path=anno_path_train,
    #                   img_path=img_path,
    #                   save_to_path=save_to_path_train,
    #                   target_shape=(0,0),
    #                   processing="crops",
    #                   pairs=True,
    #                   le=le)
    create_dataset_csv(anno_path=anno_path_test,
                       img_path=img_path,
                       save_to_path=save_to_path_test,
                       target_shape=(0,0),
                       processing="crops",
                       pairs=True,
                       le=le)
    
    # For coords
    print("Creating dataset for coords...")
    anno_path_train= r'../data_anno/training_data.csv'
    save_to_path_train = r"../data_anno/training_data_pair_coords.csv"
    anno_path_test = r'../data_anno/testing_data.csv'
    save_to_path_test = r"../data_anno/testing_data_pair_coords.csv"
    
    # Create dataset csv for ccords
    #create_dataset_csv(anno_path=anno_path_train,
    #                   img_path=img_path,
    #                   save_to_path=save_to_path_train,
    #                   target_shape=(0,0),
    #                   processing="coords",
    #                   pairs=True,
    #                   le=le)
    create_dataset_csv(anno_path=anno_path_test,
                       img_path=img_path,
                       save_to_path=save_to_path_test,
                       target_shape=(0,0),
                       processing="coords",
                       pairs=True,
                       le=le)