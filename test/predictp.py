import os
import torch
import torch.nn.functional as F
import time
import csv
import torch.nn as nn

class BinaryClassifier(nn.Module):
    def __init__(self, mean_representation_size, dropout_rate=0.3):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(mean_representation_size, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, mean_representation):
        x = self.fc1(mean_representation)
        x = self.dropout1(x)  
        x = F.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


testp_dir = 'ESMARG1positivetest'#you can change this folder for testing
max_length = 1280  
batch_size = 1000  

model = BinaryClassifier(mean_representation_size=max_length).to(device)
model.load_state_dict(torch.load(f'..model/ESMARG-1/ESMARG1.pth'))
model.eval()

total_time = 0
all_predictions = []
all_filenames = []

files = [f for f in os.listdir(testp_dir) if f.endswith('.pt')]
files = sorted(files)

for i in range(0, len(files), batch_size):
    batch_files = files[i:i+batch_size]
    batch_feature_vectors = []

    for filename in batch_files:
        feature_vector = torch.load(os.path.join(testp_dir, filename))

        if isinstance(feature_vector, dict):
            try:
                feature_vector = [torch.tensor(v, requires_grad=True).to(device) for v in feature_vector['mean_representations'].values()]
            except (KeyError, ValueError):
                print(f"Error processing {filename}: Unable to extract feature vector")
                continue
        elif torch.is_tensor(feature_vector):
            feature_vector = [feature_vector.detach().requires_grad_(True).to(device)]
        else:
            try:
                feature_vector = [torch.tensor(feature_vector, requires_grad=True).to(device)]
            except:
                print(f"Error processing {filename}: {feature_vector}")
                continue
        
        batch_feature_vectors.extend(feature_vector)
        all_filenames.extend([filename] * len(feature_vector))  

    padded_feature_vectors = []
    for vec in batch_feature_vectors:
        padded_vec = F.pad(vec, (0, max_length - len(vec)), mode='constant', value=0)
        padded_feature_vectors.append(padded_vec)

    start_time = time.time()
    with torch.no_grad():
        predictions = model(torch.stack(padded_feature_vectors).to(device))  
    prediction_time = time.time() - start_time
    
    total_time += round(prediction_time, 6)
    all_predictions.extend(predictions.tolist())

print(f"Total prediction time: {total_time:.4f} seconds")


with open(f'prediction.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)  
    writer.writerow(['Filename', 'Prediction'])  
    for filename, prediction in zip(all_filenames, all_predictions):
        writer.writerow([filename, prediction[0]])

print(f'All predictions saved to {csvfile}')