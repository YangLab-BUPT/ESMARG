import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class ARGDataset(Dataset):
    def __init__(self, data_dir, esm_data_dir, label_map_path='label_mapping.txt'):
        self.mean_representations = []
       
        self.label_map = {}
        self.max_length = 0
        self.load_data(data_dir, esm_data_dir)

    
    def load_data(self, data_dir, esm_data_dir):
        files = os.listdir(data_dir)

        for file in tqdm(files, desc="Loading samples"):
            if file.endswith('.csv'):
                csv_path = os.path.join(data_dir, file)
                file_name = os.path.splitext(file)[0]
                print(file_name)
                with open(csv_path, 'r') as f:
                    lines = f.readlines()
                    lines = lines[1:]
                    for line in lines:
                        parts = line.strip().split(',')
                        if len(parts) >= 2:
                            value = float(parts[1])
                            if value > 0.5:
                                content = parts[0].split(' ')[0]

                                pt_file = os.path.join(esm_data_dir, f"{file_name}/{content}.pt")
                                if os.path.exists(pt_file):
                                    pt_data = torch.load(pt_file)
                                    
                                    for a, mean_representation in pt_data['mean_representations'].items():
                    
                                        self.mean_representations.append(mean_representation)
                        
                 

                                        if len(mean_representation) > self.max_length:
                                            self.max_length = len(mean_representation)
                                            print(self.max_length)
                                        
                                    
                                    break     
                                else:
                                    print(f"PT file not found for content {content}")
                

    

    def __getitem__(self, idx):
        mean_representation = self.mean_representations[idx]
        return mean_representation

    def __len__(self):
        return len(self.mean_representations)
