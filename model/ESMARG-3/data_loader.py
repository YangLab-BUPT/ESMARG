import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import defaultdict

class ARGDataset(Dataset):
    def __init__(self, data_dir, label_map_path='label_mapping.txt'):
        self.mean_representations = []
        self.labels = []
        self.label_map = {}
        self.max_length = 0
        self.label_counts = defaultdict(int)  
        self.load_label_map(label_map_path)
        
        files = os.listdir(data_dir)

        for file in tqdm(files, desc="Loading samples"):
            if file.endswith('.pt'):
                data = torch.load(os.path.join(data_dir, file))
                label = file.split('|')[1]
                if ';' in label:
                    label = label.split(';')[0].strip()
                if '.' in label:
                    label = label.split('.')[0].strip()
                for a, mean_representation in data['mean_representations'].items():
                    if label in self.label_map:
                        self.mean_representations.append(mean_representation)
                        self.labels.append(self.label_map[label])
                        self.label_counts[self.label_map[label]] += 1  
                    
                        if len(mean_representation) > self.max_length:
                            self.max_length = len(mean_representation)
                    else:
                        print(f"Label {label} not found in label mapping.")

    def load_label_map(self, label_map_path):
        if os.path.exists(label_map_path):
            with open(label_map_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(':')
                    if len(parts) >= 2:
                        label = ':'.join(parts[:-1])
                        index = int(parts[-1])
                        self.label_map[label] = index
        else:
            raise FileNotFoundError(f"Label mapping file not found at {label_map_path}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mean_representation = self.mean_representations[idx]
        label = self.labels[idx]
        
        return (mean_representation, label)

    def get_label_counts(self):
        
        return self.label_counts

    def print_label_counts(self):
      
        for label, count in self.label_counts.items():
            print(f"Label {label}: {count} samples")