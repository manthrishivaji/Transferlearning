from fastapi import FastAPI, UploadFile, File
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for all origins or specify the allowed origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, or you can specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the Transfer learned model
model_path = "model/alexnet_model.pth"

# Define class labels
class_labels = {
    0: "Cat",
    1: "Dog",
    }

NO_OF_CLASSES = 2

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Pretrained Model Structure
model = models.alexnet(pretrained=False)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, NO_OF_CLASSES)

#Load Saved Weights
model.load_state_dict(torch.load(model_path, map_location=device))

model.to(device)
model.eval()

print("Model Loaded Successfully.")


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Accepts an image and returns the predicted class."""

    # Read image file
    # Here use.convert("RGB") to convert 4 channels image like png to 3 channel
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    # Define image transformation
    transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    
    image = transform(image).unsqueeze(0).to(device)

    # Make predictions
    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output, dim=1).item()
    
    return {"Predicted Class":class_labels[predicted_class]}


@app.get("/")
def Home():
    """Home page."""
    return {"message": "Welcome to the Image Classification API"}



if __name__ == "__main__":
   
    uvicorn.run(app, host="0.0.0.0", port=8000)