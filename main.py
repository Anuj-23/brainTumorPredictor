from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
import torch
from PIL import Image
import numpy as np
import io

from model.model import SOHViT

app = FastAPI()

# Allow frontend access (adjust origin as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = SOHViT()
model.load_state_dict(torch.load("model/soh_vit_model.pth", map_location="cpu"))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Prediction route
@app.post("/predict")
async def predict(file: UploadFile = File(...), use_wavelet: bool = False):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(image)


    image_tensor = transform(image_np).unsqueeze(0)  # Shape: (1, 3, 224, 224)

    class_labels = ['glioma', 'meningioma', 'no tumor', 'pituitary']

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted = class_labels[predicted.item()]
        confidence_score = round(confidence.item(), 4)

    
    return {
    "prediction": predicted,
    "confidence": confidence_score
    }

