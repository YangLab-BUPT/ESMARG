import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
import random
import csv
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description='Different combinations of negative samples were used for training, and each combination randomly selected six different negative sample data')

parser.add_argument('--num', type=str)
parser.add_argument('--f1', type=str)
parser.add_argument('--f2', type=str)
args = parser.parse_args()
num=args.num
f1=args.f1
f2=args.f2


class BinaryClassifier(nn.Module):
    def __init__(self, mean_representation_size):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(mean_representation_size, 128)
        self.fc3 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, mean_representation):
        x = self.fc1(mean_representation)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

class ARGDataset(Dataset):
    def __init__(self, positive_dir, negative10_dir, negative20_dir):
        self.positive_mean_representations = []
        self.negative_mean_representations = []
        self.labels = []
        
        positive_files = os.listdir(positive_dir)
        for file in tqdm(positive_files, desc="Loading positive samples"):
            if file.endswith('.pt'):
                data = torch.load(os.path.join(positive_dir, file))
                for label, mean_representation in data['mean_representations'].items():
                    self.positive_mean_representations.append(mean_representation)
                    self.labels.append(1)
        
        negative10_files = os.listdir(negative10_dir)
        n1=int(f1)*10000
        n2=int(f2)*10000
        negative10_files = random.sample(negative10_files, min(n1, len(negative10_files)))
        for file in tqdm(negative10_files, desc="Loading negative10 samples"):
            if file.endswith('.pt'):
                data = torch.load(os.path.join(negative10_dir, file))
                for label, mean_representation in data['mean_representations'].items():
                    self.negative_mean_representations.append(mean_representation)
                    self.labels.append(0)
        
        negative20_files = os.listdir(negative20_dir)
        selected_files = random.sample(negative20_files, min(n2, len(negative20_files)))
        for file in tqdm(selected_files, desc="Loading negative20 samples"):
            if file.endswith('.pt'):
                data = torch.load(os.path.join(negative20_dir, file))
                for label, mean_representation in data['mean_representations'].items():
                    self.negative_mean_representations.append(mean_representation)
                    self.labels.append(0)

        self.all_mean_representations = self.positive_mean_representations + self.negative_mean_representations
        self.max_length = max(len(mean_representation) for mean_representation in self.all_mean_representations)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if idx < len(self.positive_mean_representations):
            mean_representation = self.positive_mean_representations[idx]
        else:
            mean_representation = self.negative_mean_representations[idx - len(self.positive_mean_representations)]
        
        mean_representation = F.pad(mean_representation, (0, self.max_length - len(mean_representation)), mode='constant', value=0)
        label = self.labels[idx]
        return (mean_representation, label)

'''You can replace the path of the train folder with the folder after extracting features using esm2'''
trainp_dir = 'trainptrue'
trainn_dir = 'trainn/trainn_2'
trainn2 = 'trainn/trainn'   
train_dataset = ARGDataset(trainp_dir, trainn_dir, trainn2)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


model = BinaryClassifier(mean_representation_size=1280).to(device) 
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 20
train_losses = []
train_accuracies = []
train_f1_scores = []
train_precisions = []
train_recalls = []
val_losses = []
val_accuracies = []
val_f1_scores = []
val_precisions = []
val_recalls = []


best_model_state_dict = None
best_val_f1 = 0

for epoch in range(num_epochs):
  train_loss = 0.0
  train_correct = 0
  train_total = 0
  train_y_true = []
  train_y_pred = []
    
  with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
        for i, (mean_representations, labels) in enumerate(train_loader):
            mean_representations = mean_representations.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(mean_representations)
            
            if torch.isnan(outputs).any():
                continue
            
            loss = criterion(outputs, labels.float().unsqueeze(1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == labels.unsqueeze(1)).sum().item()
            train_total += labels.size(0)
            
            train_y_true.extend(labels.tolist())
            train_y_pred.extend(predicted.squeeze().tolist())
            pbar.update(1)

  train_loss /= len(train_loader)
  train_accuracy = 100 * train_correct / train_total
  train_f1 = f1_score(train_y_true, train_y_pred)
  train_precision = precision_score(train_y_true, train_y_pred)
  train_recall = recall_score(train_y_true, train_y_pred)

  with torch.no_grad():
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    val_y_true = []
    val_y_pred = []

    with tqdm(total=len(val_loader), desc="Validating", unit="batch") as pbar:
        for mean_representations, labels in val_loader:
           
            mean_representations = mean_representations.to(device)
            labels = labels.to(device)
            
            outputs = model(mean_representations)

            if torch.isnan(outputs).any():
                print(f"Encountered nan values in outputs during validation. Skipping this batch.")
                continue
            
            loss = criterion(outputs, labels.float().unsqueeze(1))
            val_loss += loss.item()

            predicted = (outputs > 0.5).float()
            val_correct += (predicted == labels.unsqueeze(1)).sum().item()
            val_total += labels.size(0)

            val_y_true.extend(labels.tolist())
            val_y_pred.extend(predicted.cpu().detach().numpy().tolist())  
            pbar.update(1)


    val_loss /= len(val_loader)
    val_accuracy = 100 * val_correct / val_total
    val_f1 = f1_score(val_y_true, val_y_pred)
    val_precision = precision_score(val_y_true, val_y_pred)
    val_recall = recall_score(val_y_true, val_y_pred)

  train_losses.append(train_loss)
  train_accuracies.append(train_accuracy)
  train_f1_scores.append(train_f1)
  train_precisions.append(train_precision)
  train_recalls.append(train_recall)
  val_losses.append(val_loss)
  val_accuracies.append(val_accuracy)
  val_f1_scores.append(val_f1)
  val_precisions.append(val_precision)
  val_recalls.append(val_recall)
  print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Train F1: {train_f1:.4f}, Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%, Validation F1: {val_f1:.4f}, Validation Precision: {val_precision:.4f}, Validation Recall: {val_recall:.4f}')
  best_model_state_dict = model.state_dict()
  torch.save(best_model_state_dict, f'epoch/{epoch}_gmgc_{f1}_{f2}_{num}.pth')
  if val_f1 > best_val_f1:
    best_val_f1 = val_f1
    best_model_state_dict = model.state_dict()



with open(f'dmcompare/results_gmgc_{f1}_{f2}_{num}.csv', 'w', newline='') as f:
  writer = csv.writer(f)
  writer.writerow(['Train Losses', 'Train Accuracies', 'Train F1 Scores', 'Train Precisions', 'Train Recalls', 'Validation Losses', 'Validation Accuracies', 'Validation F1 Scores', 'Validation Precisions', 'Validation Recalls'])
  for i in range(len(train_losses)):
    writer.writerow([train_losses[i], train_accuracies[i], train_f1_scores[i], train_precisions[i], train_recalls[i], val_losses[i], val_accuracies[i], val_f1_scores[i], val_precisions[i], val_recalls[i]])
    

torch.save(best_model_state_dict, 'best_model_en.pth')