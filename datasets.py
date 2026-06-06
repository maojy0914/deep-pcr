
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py


class PathologyDataset(Dataset):

    def __init__(self, clinical_csv, features_dir, label_col='label'):
        self.clinical_csv = clinical_csv
        self.features_dir = features_dir
        self.label_col = label_col

        self.df = pd.read_csv(clinical_csv)

        self.samples = []
        for idx, row in self.df.iterrows():
            slide_id = row.get('slide_id', '')
            if pd.isna(slide_id) or slide_id == '':
                continue

            feature_file = os.path.join(features_dir, f"{slide_id}.h5")
            if os.path.exists(feature_file):
                self.samples.append({
                    'slide_id': slide_id,
                    'label': row.get(label_col, 0),
                    'idx': idx
                })


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        slide_id = sample['slide_id']
        label = sample['label']

        feature_file = os.path.join(self.features_dir, f"{slide_id}.h5")
        with h5py.File(feature_file, 'r') as f:
            features = f['features'][:]

        features = torch.from_numpy(features).float()

        return {
            'slide_id': slide_id,
            'features': features,
            'label': torch.tensor(label, dtype=torch.long)
        }


class MultiModalDataset(Dataset):

    def __init__(self, split_csv, features_dir, embedding_dir, clinical_csv, mode='train'):
        self.features_dir = features_dir
        self.embedding_dir = embedding_dir

        df = pd.read_csv(split_csv)

        if mode == 'train':
            slide_ids = df['train'].dropna().tolist()
        else:  # mode == 'test'
            slide_ids = df['test'].dropna().tolist()

        clinical_df = pd.read_csv(clinical_csv)
        slide_to_label = dict(zip(clinical_df['slide_id'], clinical_df['label']))

        self.samples = []
        for slide_id in slide_ids:
            self.samples.append({
                'slide_id': slide_id,
                'label': slide_to_label.get(slide_id, 0)
            })


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        slide_id = sample['slide_id']
        label = sample['label']

        feature_file = os.path.join(self.features_dir, f"{slide_id}.h5")
        with h5py.File(feature_file, 'r') as f:
            features = f['features'][:]

        embedding_file = os.path.join(self.embedding_dir, f"{slide_id}.pt")
        embedding = torch.load(embedding_file)

        features = torch.from_numpy(features).float()

        return {
            'slide_id': slide_id,
            'features': features,
            'embedding': embedding,
            'label': torch.tensor(label, dtype=torch.long)
        }