import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class GoResNet(nn.Module):
    def __init__(self, num_input_planes=17, num_classes=361):
        super(GoResNet, self).__init__()
        
        # 1. Load Standard ResNet-18 (No pre-trained weights)
        self.resnet = resnet18(weights=None)
        
        # 2. Modify Input Layer (17 channels instead of 3)
        self.resnet.conv1 = nn.Conv2d(
            in_channels=num_input_planes,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        # 3. Modify Output Layer (361 moves instead of 1000 classes)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # Pass through ResNet
        x = self.resnet(x)
        
        # 4. LogSoftmax for Probability Distribution
        return F.log_softmax(x, dim=1)

if __name__ == "__main__":
    # Test Architecture
    model = GoResNet()
    dummy = torch.randn(1, 17, 19, 19)
    print(f" Model Output Shape: {model(dummy).shape} (Should be [1, 361])")