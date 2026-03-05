import torch
import torch.nn as nn
from torchvision import models


class SmartMineEfficientNet(nn.Module):

    def __init__(self, num_classes):
        super(SmartMineEfficientNet, self).__init__()

        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        # Freeze feature extractor
        for param in self.model.features.parameters():
            param.requires_grad = False

        in_features = self.model.classifier[1].in_features

        self.model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)
