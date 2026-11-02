
import torch
import torch.nn as nn
from torchvision import models

MODEL_PATH = "brain_tumor_mobilenetv3_large.pth"

# Load checkpoint
checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

# Read metadata saved with the checkpoint
CLASSES = checkpoint["class_names"]
IMAGE_SIZE = checkpoint["image_size"]

print("Architecture:", checkpoint["architecture"])
print("Classes:", CLASSES)
print("Image size:", IMAGE_SIZE)

# Recreate the model architecture
model = models.mobilenet_v3_large(weights=None)

in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, checkpoint["num_classes"])

# Load the actual trained weights
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("\nModel loaded successfully!")

# Test inference with a dummy image
dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

with torch.inference_mode():
    output = model(dummy_input)
    probabilities = torch.softmax(output, dim=1)

predicted_index = probabilities.argmax(dim=1).item()

print("Output shape:", output.shape)
print("Predicted class:", CLASSES[predicted_index])
print(f"Confidence: {probabilities[0, predicted_index].item():.2%}")
