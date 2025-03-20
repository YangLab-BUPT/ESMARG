import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiClassifier(nn.Module):
    def __init__(self, mean_representation_size, num_classes):
        super(MultiClassifier, self).__init__()
        self.fc1 = nn.Linear(mean_representation_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes) 

    def forward(self, mean_representation):
        x = self.fc1(mean_representation)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x