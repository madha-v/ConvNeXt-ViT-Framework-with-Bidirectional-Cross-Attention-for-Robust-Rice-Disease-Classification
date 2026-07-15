# data/dataset.py
import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from .augmentations import get_transforms

class RiceDiseaseDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Read the folder name (text label) and file name (image_id) from Kaggle's CSV
        folder_name = str(row['label_string'])
        file_name = str(row['image_id'])
        
        # Kaggle stores images inside subfolders, so we build the full path here
        img_path = os.path.join(self.img_dir, folder_name, file_name)
        
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image missing at: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        # The model needs the math integer label we generated, not the text
        label = int(row['label_idx'])
        return image, label

def build_loaders(config):
    df = pd.read_csv(config['data']['csv_path'])
    
    # Translate Kaggle's text labels into numerical indexes (0-9) for the model
    labels_text = df['label'].unique()
    label_to_idx = {name: idx for idx, name in enumerate(sorted(labels_text))}
    
    # Save both versions in the dataframe
    df['label_string'] = df['label']
    df['label_idx'] = df['label'].map(label_to_idx)
    
    # Stratified split ensures all diseases are represented equally
    train_df, val_df = train_test_split(
        df, 
        test_size=config['data']['split_ratio'], 
        stratify=df['label_idx'], 
        random_state=42
    )

    train_trans, val_trans = get_transforms(config['data']['img_size'])

    train_ds = RiceDiseaseDataset(train_df, config['data']['img_dir'], transform=train_trans)
    val_ds = RiceDiseaseDataset(val_df, config['data']['img_dir'], transform=val_trans)

    train_loader = DataLoader(
        train_ds, batch_size=config['data']['batch_size'], shuffle=True,
        num_workers=config['data']['num_workers'], pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=config['data']['batch_size'], shuffle=False,
        num_workers=config['data']['num_workers'], pin_memory=True
    )

    return train_loader, val_loader, len(label_to_idx)