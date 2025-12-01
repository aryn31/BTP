# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Define the Neural Network
class HealthClassifier(nn.Module):
    def __init__(self):
        super(HealthClassifier, self).__init__()
        # Input: 7 features (Age, BMI, HR, SpO2, Sys, Dia, Stress)
        # Hidden Layer: 16 neurons
        # Output: 4 classes (Optimal, Elevated, High, Critical)
        self.fc1 = nn.Linear(7, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # No softmax here, CrossEntropyLoss handles it
        return x

# 2. Helper to train the model (The "Logic" you asked for)
def train(net, trainloader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()         # 1. Clear gradients
            outputs = net(images)         # 2. Forward pass
            loss = criterion(outputs, labels) # 3. Calculate Loss
            loss.backward()               # 4. Backward pass (Gradient calculation)
            optimizer.step()              # 5. Update weights

# 3. Helper to test the model
def test(net, testloader):
    criterion = nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return loss / len(testloader.dataset), correct / total