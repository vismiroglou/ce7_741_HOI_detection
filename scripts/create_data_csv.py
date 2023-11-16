from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append('../src/')
from utils import collate_fn
from dataset import CropsScikitDataset

from tqdm import tqdm
import pandas as pd
import numpy as np

def preprocess(data):
    # Separate features and labels in separate lists
    features, labels = zip(*data)
    labels = np.array(labels).reshape(-1)
    
    features = [coord[0] for coord in features]
    features = np.array(features)
    #print(features)
    return features, labels

    # Converts each img tensor to numpy arrays
    features = [image[0].numpy() for image in tqdm(features,total=len(features),desc="Converting to np array")]

    # We reshape the img from a 3D to 2D array
    features = [image.reshape(image.shape[0],-1) for image in tqdm(features,total=len(features),desc="Reshaping 3D to 2D")]

    # Find the maximum height and width among all crops
    max_height = max(img.shape[0] for img in tqdm(features,total=len(features),desc="Finding max height"))
    max_width = max(img.shape[1] for img in tqdm(features,total=len(features),desc="Finding max width"))
    #print(f"({max_height},{max_width})")

    # Add centered padding around the crop to match max height and width
    # To counter floor divison we add +1
    features = [np.pad(img,pad_width=(((max_height - img.shape[0])//2, (max_height - img.shape[0] + 1)//2),
                                    ((max_width - img.shape[1])//2, (max_width - img.shape[1] + 1)//2)),
                    mode="constant", constant_values=0.0) for img in tqdm(features,total=len(features),desc="Applying padding")]

    features = np.array(features)
    samples,nx,ny = features.shape
    features = features.reshape(samples,nx*ny)
    
    return features, labels

def to_csv(data,save_path):
    # Preprocess data
    features, labels = preprocess(data)

    # Create dataframe of features and labels
    #features_df = pd.DataFrame(features, columns=[f'pixel_{i}' for i in range(features.shape[1])])
    features_df = pd.DataFrame(features, columns=['x1','y1','x2','y2'])
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
    labels_742 = ['human-hold-bicycle', 'human-park-bicycle', 'human-park-motorcycle', 'human-pickup-bicycle', 'human-pickup-motorcycle', 'human-ride-bicycle', 'human-ride-motorcycle', 'human-ride-vehicle', 'human-walk-bicycle', 'human-walk-motorcycle']
    le.fit(labels_742)

    # Initialize dataset and dataloader
    anno_path = r'../data_anno/annotations_hoi_frame_742.csv'
    img_path = r'../../dataset/images'
    save_to_path = r"../data_anno/coords_742.csv"
    dataset = CropsScikitDataset(anno_file=anno_path, img_dir = img_path, label_encoder=le)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    
    # Load all the data into memory
    data = []
    for crop, label, coord,_ in tqdm(dataloader,total=len(dataloader),leave=True,desc="Loading into memory"):
        data.append((coord,label))
    
    # Convert to csv file
    to_csv(data, save_to_path)
    print("Finished!")