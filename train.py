import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore", message="Palette images with Transparency expressed in bytes should be converted to RGBA images")

# Use GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

# Load datasets
train_data = datasets.ImageFolder("dataset/train", transform=transform)
test_data = datasets.ImageFolder("dataset/test", transform=transform)

# DataLoaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


# Model Definition
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
num_classes = len(train_data.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

load = input("Load Weights from model.pth? (Y/N) ")
if load.upper() == "Y":
    model.load_state_dict(torch.load("model.pth"))

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training Loop
model.train()

num_epochs = 10

for epoch in range(num_epochs):
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)

    print(f"Epoch [{epoch + 1} / {num_epochs}], Loss: {avg_loss:.4f}")


# Testing Loop
model.eval()

class_correct = [0] * num_classes
class_total = [0] * num_classes

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        correct_predictions = (predicted == labels)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        for i in range(len(labels)):
            label = labels[i].item()
            class_correct[label] += correct_predictions[i].item()
            class_total[label] += 1

accuracy = 100 * correct / total

print(f"Test Accuracy: {accuracy:.2f}%")


# Per-Class Accuracies
class_accuracies = []

for i in range(num_classes):
    accuracy = 100 * class_correct[i] / class_total[i]
    pokemon_name = train_data.classes[i]
    class_accuracies.append((pokemon_name, accuracy))

class_accuracies.sort(key=lambda x: x[1])

print("\nTop 5 Accuracies:")
for name, acc in reversed(class_accuracies[-5:]):
    print(f"{name}: {acc:.2f}%")


print("\nLowest 5 Accuracies:")
for name, acc in class_accuracies[:5]:
    print(f"{name}: {acc:.2f}%")


save = input("\nSave Weights to model.pth? (Y/N) ")
if save.upper() == "Y":
    torch.save(model.state_dict(), "model.pth")