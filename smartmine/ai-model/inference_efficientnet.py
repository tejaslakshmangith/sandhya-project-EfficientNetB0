import json
import os
import torch
from PIL import Image
from torchvision import transforms
from models.efficientnet_model import SmartMineEfficientNet

# Load class names dynamically from the JSON saved during training.
# Fall back to the default mineral/ore/rock list if the file is absent.
_CLASS_NAMES_PATH = "models/class_names.json"
if os.path.isfile(_CLASS_NAMES_PATH):
    with open(_CLASS_NAMES_PATH, "r") as _f:
        class_names = json.load(_f)
else:
    class_names = ["mineral", "ore", "rock"]

# ImageNet normalization — must match the transform used during training
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

model = SmartMineEfficientNet(len(class_names))
model.load_state_dict(torch.load("models/efficientnet_smartmine.pth", map_location="cpu"))
model.eval()


def predict_image(image_path: str) -> dict:
    image = Image.open(image_path).convert("RGB")
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)

    confidence, pred = torch.max(probs, 1)

    return {
        "class": class_names[pred.item()],
        "confidence": round(float(confidence), 4)
    }
