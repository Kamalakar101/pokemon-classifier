import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from torchvision.models import ResNet18_Weights
from PIL import Image
import streamlit as st

image_size = 224

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load class names
train_data = datasets.ImageFolder("dataset/train")
classes = train_data.classes

# Create model
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(classes))

# Load trained weights
model.load_state_dict(torch.load("model.pth", map_location=device))

model.to(device)
model.eval()

transform = transforms.Compose([transforms.Lambda(lambda img: img.convert("RGB")), transforms.Resize((image_size, image_size)), transforms.ToTensor()])

st.title("Pokémon Classifier (Gen 1)")
st.write("Upload an image and the model will predict which Pokémon it is.")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    # Prediction
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", width="stretch")

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor.to(device))
        probs = torch.softmax(outputs, dim=1)
        top5_prob, top5_idx = torch.topk(probs, 5)

    best_idx = top5_idx[0][0].item()

    st.success(f"Prediction: {classes[best_idx]}")

    st.subheader("Top 5 Predictions")

    for i in range(5):
        idx = top5_idx[0][i].item()
        prob = top5_prob[0][i].item() * 100
        st.write(f"{classes[idx]} ({prob:.2f}%)")
        st.progress(prob / 100)