import torch
import torch.nn as nn
import torch.nn.functional as F

class HealthClassifier(nn.Module):
    def __init__(self):
        super(HealthClassifier, self).__init__()
        #input 7 features (Age, BMI, HR, SpO2, Sys, Dia, Stress)
        self.fc1 = nn.Linear(7, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) 
        return x

def train(net, trainloader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    
    print(f"Starting training for {epochs} epochs...") 
    net.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Calculate average loss for this epoch
        avg_loss = running_loss / len(trainloader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}") 

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
            
    if len(testloader.dataset) == 0: return 0.0, 0.0
    return loss / len(testloader.dataset), correct / total