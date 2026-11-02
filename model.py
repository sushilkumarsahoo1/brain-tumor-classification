
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from pathlib import Path

# -------------------------
# Configuration
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "brain_tumor_mobilenetv3_large.pth"

DEVICE = torch.device("cpu")


# -------------------------
# Load trained checkpoint
# -------------------------
checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True,
)

CLASS_NAMES = checkpoint["class_names"]
IMAGE_SIZE = checkpoint["image_size"]
NUM_CLASSES = checkpoint["num_classes"]

if checkpoint["architecture"] != "mobilenet_v3_large":
    raise ValueError(
        f"Unexpected architecture: {checkpoint['architecture']}"
    )

if len(CLASS_NAMES) != NUM_CLASSES:
    raise ValueError("Class names and number of classes do not match.")


# -------------------------
# Rebuild model architecture
# -------------------------
model = models.mobilenet_v3_large(weights=None)

in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()


# -------------------------
# Image preprocessing
# Matches notebook evaluation:
# Resize((224, 224)), ToTensor(),
# ImageNet normalization
# -------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# -------------------------
# Prediction function
# -------------------------
def predict_image(image: Image.Image) -> dict:
    """
    Predict one MRI image.

    Returns the predicted class, confidence percentage,
    and probabilities for all classes.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL Image.")

    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_index = int(probabilities.argmax().item())

    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": round(
            float(probabilities[predicted_index].item()) * 100,
            2,
        ),
        "probabilities": {
            class_name: round(float(probabilities[i].item()) * 100, 2)
            for i, class_name in enumerate(CLASS_NAMES)
        },
    }


# -------------------------
# Basic verification
# -------------------------
if __name__ == "__main__":
    print("Model loaded successfully!")
    print("Architecture: MobileNetV3-Large")
    print("Classes:", CLASS_NAMES)
    print("Image size:", IMAGE_SIZE)
    print("Device:", DEVICE)

    # Dummy inference checks output shape only.
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    with torch.inference_mode():
        output = model(dummy_input)

    assert output.shape == (1, NUM_CLASSES)
    print("Inference shape check passed:", tuple(output.shape))
