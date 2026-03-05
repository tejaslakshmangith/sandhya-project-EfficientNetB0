import torch
from PIL import Image
from torchvision import transforms
from models.efficientnet_model import SmartMineEfficientNet

class_names = ["safe", "unsafe"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
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
