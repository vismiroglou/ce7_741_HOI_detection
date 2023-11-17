from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append('../src/')
from utils import collate_fn
from dataset import CropsScikitDataset

from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

def preprocess(data, target_shape):
    # Separate features and labels in separate lists
    features, labels = zip(*data)
    labels = np.array(labels).reshape(-1)
    
    # Uncomment for coordinates, else for crops
    #features = [coord[0] for coord in features]
    #features = np.array(features)
    ##print(features)
    #return features, labels

    # Converts each img tensor to numpy arrays
    features = [image[0].numpy() for image in tqdm(features,total=len(features),desc="Converting to np array")]

    # We reshape the img from a 3D to 2D array
    features = [image.reshape(image.shape[0],-1) for image in tqdm(features,total=len(features),desc="Reshaping 3D to 2D")]

    """# Find the maximum height and width among all crops
    max_height = max(img.shape[0] for img in tqdm(features,total=len(features),desc="Finding max height"))
    max_width = max(img.shape[1] for img in tqdm(features,total=len(features),desc="Finding max width"))
    print(f"({max_height},{max_width})")
    """
    
    
    padded_crops = []
    #for crop in tqdm(features, total=len(features), desc="Resizing/Padding crops to (77,62)"):
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
        
    """ Former padding function
     Add centered padding around the crop to match max height and width
     To counter floor divison we add +1
    features = [np.pad(img,pad_width=(((max_height - img.shape[0])//2, (max_height - img.shape[0] + 1)//2),
                                    ((max_width - img.shape[1])//2, (max_width - img.shape[1] + 1)//2)),
                    mode="constant", constant_values=0.0) for img in tqdm(features,total=len(features),desc="Applying padding")]

    features = np.array(features)
    samples,nx,ny = features.shape
    features = features.reshape(samples,nx*ny)
    """
    
    features = np.array(padded_crops)
    samples,nx,ny = features.shape
    features = features.reshape(samples,nx*ny)
    
    return features, labels

def to_csv(data,save_path, target_shape):
    # Preprocess data
    features, labels = preprocess(data, target_shape)
    
    # Create dataframe of features and labels
    features_df = pd.DataFrame(features, columns=[f'pixel_{i}' for i in range(features.shape[1])])
    #features_df = pd.DataFrame(features, columns=['x1','y1','x2','y2']) # Only for coordinates
    labels_df = pd.DataFrame(labels, columns=['label'])

    # Concatenate features and labels along columns
    df = pd.concat([features_df, labels_df], axis=1)

    # Save to CSV
    print("Saving to csv...")
    df.to_csv(save_path, index=False)
    #print(df)



if __name__ == '__main__':
    # Setting up custom dataset and dataloader
    le = LabelEncoder()
    labels_741 = ['human-hold-bicycle', 'human-ride-bicycle', 'human-ride-motorcycle', 'human-walk-bicycle', 'human-walk-motorcycle']
    labels_742 = ['human-hold-bicycle', 'human-park-bicycle', 'human-park-motorcycle', 'human-pickup-bicycle', 'human-pickup-motorcycle', 'human-ride-bicycle', 'human-ride-motorcycle', 'human-ride-vehicle', 'human-walk-bicycle', 'human-walk-motorcycle', 'human-hold-motorcycle']
    le.fit(labels_742)

    # Initialize dataset and dataloader
    anno_path = r'../data_anno/training_data.csv'
    img_path = r'../../dataset/images'
    save_to_path = r"../data_anno/training_data_crops.csv"
    
    torch.manual_seed(0)
    dataset = CropsScikitDataset(anno_file=anno_path, img_dir = img_path, label_encoder=le)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    
    # Load all the data into memory
    data = []
    for crop, label, coord,_ in tqdm(dataloader,total=len(dataloader),leave=True,desc="Loading into memory"):
        data.append((crop,label))
    
    # Convert to csv file
    target_shape = (77,62)
    to_csv(data, save_to_path, target_shape)
    print("Finished!")