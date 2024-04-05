import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))  # 1st layer
        x = self.fc2(x)  # 2nd layer
        return x

# Beispiel: Zustandsdimensionen
state_dim = 61  # Beispielwert, ändere dies entsprechend deiner Daten

# Erstelle ein Neuronales Netzwerk mit den korrekten Dimensionen
model = NeuralNetwork(input_size=state_dim, hidden_size=101, output_size=6)
