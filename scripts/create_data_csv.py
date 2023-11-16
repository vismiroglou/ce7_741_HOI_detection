from torch.utils.data import DataLoader
import sys
sys.path.append('../src/')
from utils import collate_fn
from dataset import CropsScikitDataset

from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

def preprocess(data):
    # Separate features and labels in separate lists
    features, labels = zip(*data)
    labels = np.array(labels).reshape(-1)

    # Converts each img tensor to numpy arrays
    features = [image[0].numpy() for image in features]

    # We reshape the img from a 3D to 2D array
    features = [image.reshape(image.shape[0],-1) for image in features]

    # Find the maximum height and width among all crops
    max_height = max(img.shape[0] for img in features)
    max_width = max(img.shape[1] for img in features)
    #print(f"({max_height},{max_width})")

    # Add centered padding around the crop to match max height and width
    # To counter floor divison we add +1
    features = [np.pad(img,pad_width=(((max_height - img.shape[0])//2, (max_height - img.shape[0] + 1)//2),
                                    ((max_width - img.shape[1])//2, (max_width - img.shape[1] + 1)//2)),
                    mode="constant", constant_values=0.0) for img in features]

    features = np.array(features)
    samples,nx,ny = features.shape
    features = features.reshape(samples,nx*ny)
    
    return features, labels

def to_csv(data):
    # Preprocess data
    features, labels = preprocess(data)

    # Create dataframe of features and labels
    features_df = pd.DataFrame(features, columns=[f'pixel_{i}' for i in range(features.shape[1])])
    labels_df = pd.DataFrame(labels, columns=['label'])

    # Concatenate features and labels along columns
    df = pd.concat([features_df, labels_df], axis=1)

    # Save to CSV
    df.to_csv('crops.csv', index=False)
    print(df)



if __name__ == '__main__':
    # Setting up custom dataset and dataloader
    le = LabelEncoder()
    le.fit(['human-ride-bicycle', 'human-walk-bicycle', 'human-hold-bicycle', 'human-ride-motorcycle', 'human-walk-motorcycle'])

    # Initialize dataset and dataloader
    anno_path = r'../data_anno/annotations_hoi_742.csv'
    img_path = r'../../dataset/images'
    dataset = CropsScikitDataset(anno_file=anno_path, img_dir = img_path, label_encoder=le)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    
    # Load all the data into memory
    data = []
    for batch in dataloader:
        crop, label, *_ = batch
        data.append((crop,label))
    
    # Convert to csv file
    to_csv(data)