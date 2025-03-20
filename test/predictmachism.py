import torch
from torch.utils.data import DataLoader
from data_loader import ARGDataset
from model import MultiClassifier
import csv
import os
import time
'''This is the document used to make predictions about ARG resistance mechanisms, which can be predicted on test sequences using parameters that have been trained'''
start_time = time.time()

def load_label_map(label_map_path):
    label_map = {}
    with open(label_map_path, 'r') as f:
        for line in f:
            parts = line.strip().split(':')
            if len(parts) >= 2:
                label = ':'.join(parts[:-1])
                index = int(parts[-1])
                label_map[index] = label
    return label_map

class ARGDatasetWithFileNames(ARGDataset):
    def __init__(self, data_dir, label_map_path='../model/ESMARG-3/label_mapping.txt'):
        super().__init__(data_dir, label_map_path)
        self.file_names = [file for file in os.listdir(data_dir) if file.endswith('.pt')]

    def __getitem__(self, idx):
        mean_representation, label = super().__getitem__(idx)
        file_name = self.file_names[idx]
        return mean_representation, label, file_name

model_path = './model/ESMARG-3ESMARG3.pth'
label_map_path = './model/ESMARG-3label_mapping.txt'

label_map = load_label_map(label_map_path)

index_to_label = {index: label for index, label in label_map.items()}
label_to_index = {label: index for label, index in label_map.items()}

new_data_dir = 'ESMARG3test'
new_dataset = ARGDatasetWithFileNames(new_data_dir, label_map_path)

model = MultiClassifier(mean_representation_size=new_dataset.max_length, num_classes=7)
model.load_state_dict(torch.load(model_path))
model.eval()

new_loader = DataLoader(new_dataset, batch_size=1, shuffle=False)


predictions = []
count=0
with torch.no_grad():
    for mean_representations, _, file_name in new_loader:
        mean_representations = mean_representations[0]
        outputs = model(mean_representations)
        
        _, predicted = torch.max(outputs, dim=0)
        
        prdict_label =index_to_label[predicted.item()]
        true_label = file_name[0].split('|')[1]
        
        predictions.append((file_name, prdict_label, true_label))

end_time = time.time()
total_time = end_time - start_time

output_file = 'predictions_ESMARG3.csv'

with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['File Name', 'Predicted Label', 'True Label'])
    for file_name, predicted_label, true_label in predictions:
        if predicted_label == true_label.split('.')[0].strip():
            count = count + 1
        writer.writerow([file_name, predicted_label, true_label])



print(f'Predictions saved to {output_file}')
print(f'Total execution time: {total_time:.2f} seconds')