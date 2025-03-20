import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
import csv
from data_loader import ARGDataset
from model import MultiClassifier
from sklearn.model_selection import train_test_split


from sklearn.model_selection import train_test_split


data_dir = '../deeparg_train1'
dataset = ARGDataset(data_dir)
num_classes = len(dataset.label_map)

labels = [label for _, label in dataset]
unique_labels = set(labels)

train_dataset = []
val_dataset = []


for label in unique_labels:
    
    label_indices = [idx for idx, l in enumerate(labels) if l == label]
    
    if len(label_indices) == 1:
        
        train_dataset.append(dataset[label_indices[0]])
    else:
        
        train_indices, val_indices = train_test_split(label_indices, test_size=0.2, random_state=42)
        
        train_dataset.extend([dataset[idx] for idx in train_indices])
        val_dataset.extend([dataset[idx] for idx in val_indices])


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

model = MultiClassifier(mean_representation_size=dataset.max_length, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 15
best_model_state_dict = None
best_val_f1 = 0

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

for epoch in range(num_epochs):
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    train_y_true = []
    train_y_pred = []

    model.train()
    for i, (mean_representations, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(mean_representations)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)
        train_y_true.extend(labels.tolist())
        train_y_pred.extend(predicted.tolist())

    train_loss /= len(train_loader)
    train_accuracy = 100 * train_correct / train_total
    train_f1 = f1_score(train_y_true, train_y_pred, average='weighted',zero_division=1)
    train_precision = precision_score(train_y_true, train_y_pred, average='weighted',zero_division=1)
    train_recall = recall_score(train_y_true, train_y_pred, average='weighted',zero_division=1)

    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_y_true = []
        val_y_pred = []

        for mean_representations, labels in val_loader:
            outputs = model(mean_representations)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
            val_y_true.extend(labels.tolist())
            val_y_pred.extend(predicted.tolist())

        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        val_f1 = f1_score(val_y_true, val_y_pred, average='weighted',zero_division=1)
        val_precision = precision_score(val_y_true, val_y_pred, average='weighted',zero_division=1)
        val_recall = recall_score(val_y_true, val_y_pred, average='weighted',zero_division=1)

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

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_model_state_dict = model.state_dict()

with open('results66.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Train Losses', 'Train Accuracies', 'Train F1 Scores', 'Train Precisions', 'Train Recalls', 'Validation Losses', 'Validation Accuracies', 'Validation F1 Scores', 'Validation Precisions', 'Validation Recalls'])
    for i in range(len(train_losses)):
        writer.writerow([train_losses[i], train_accuracies[i], train_f1_scores[i], train_precisions[i], train_recalls[i], val_losses[i], val_accuracies[i], val_f1_scores[i], val_precisions[i], val_recalls[i]])


torch.save(best_model_state_dict, 'ESMARG_2.pth')

print(f'Best validation F1 score: {best_val_f1:.4f}')
print(f'Best validation Precision: {max(val_precisions):.4f}')
print(f'Best validation Recall: {max(val_recalls):.4f}')