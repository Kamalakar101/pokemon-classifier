import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from torchvision.models import ResNet18_Weights
from PIL import Image

import warnings
warnings.filterwarnings("ignore", message="Palette images with Transparency expressed in bytes should be converted to RGBA images")

# Use GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.Lambda(lambda img: img.convert("RGB")), transforms.Resize((128, 128)), transforms.ToTensor()])
train_data = datasets.ImageFolder("dataset/train")

# Model Definition
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
num_classes = len(train_data.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Load Weights
model.load_state_dict(torch.load("model.pth"))
model = model.to(device)

model.eval()

# Load Image
image_name = input("What is the image name? ")
image = Image.open(image_name)
image = transform(image)
image = image.unsqueeze(0)

# Prediction
with torch.no_grad():
    outputs = model(image.to(device))
    _, predicted = torch.max(outputs, 1)

classes = train_data.classes

pokemon_name = classes[predicted.item()]
print(f"Prediction: {pokemon_name}")