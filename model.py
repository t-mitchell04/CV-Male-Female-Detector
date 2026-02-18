import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
import os
import splitfolders

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm


class Dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    @property
    def classes(self):
        return self.data.classes
    
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
dataset = Dataset(data_dir='./dataset/train', transform=transform)
target_to_class = {v: k for k, v in ImageFolder('./dataset/train').class_to_idx.items()}
print(target_to_class)

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
for image, label in dataset:
    break

for images, labels in dataloader:
    break

class GenderClassifer(nn.Module):
    def __init__(self, num_classes=2):
        super(GenderClassifer, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=False)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        enet_out_size = 1280
        # Make a classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enet_out_size, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output
    
model = GenderClassifer(num_classes=2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_folder = dataset
test_folder = Dataset('./dataset/test', transform=transform)

train_loader = DataLoader(train_folder, batch_size=16, shuffle=True)
test_loader = DataLoader(test_folder, batch_size=16, shuffle=True)

num_epochs = 5
train_losses = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = GenderClassifer(num_classes=2)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

torch.cuda.empty_cache()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc='Training loop'):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)
    print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss}")

plt.plot(train_losses, label='Training loss')
plt.legend()
plt.title("Loss over epochs")
plt.show()

model.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing loop"):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item() * labels.size(0)

        _, predicted = torch.max(outputs, dim=1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
avg_test_loss = test_loss / len(test_loader.dataset)
accuracy = correct / total
print(f"Test loss: {avg_test_loss:.4f}  |  Test accuracy: {accuracy*100:.2f}%")

torch.save({
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "epoch": epoch
}, "model.pth")