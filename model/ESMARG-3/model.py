import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiClassifier(nn.Module):
    def __init__(self, mean_representation_size, num_classes):
        super(MultiClassifier, self).__init__()
        self.fc1 = nn.Linear(mean_representation_size, 512)
        self.fc2 = nn.Linear(512, 64)
        ##self.fc3 = nn.Linear(256,64)
        #self.fc4 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3) 
        self.fc5 = nn.Linear(64, num_classes) 

    def forward(self, mean_representation):
        x = torch.relu(self.fc1(mean_representation))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        self.dropout = nn.Dropout(0.3)
        #x = torch.relu(self.fc3(x))
        #x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x