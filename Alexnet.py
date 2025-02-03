import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

import torch.utils.model_zoo as model_zoo
import torch.onnx

import torch.nn.init as init

# AlexNet 모델 정의
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 모델 생성
model = AlexNet()

# optimizer 정의
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# loss function 정의
criterion = nn.CrossEntropyLoss()

# 학습 루프 (예시)
for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 더미 입력 생성 (배치 크기 1, 3채널, 224x224 이미지)
dummy_input = torch.randn(1, 3, 224, 224)

# ONNX로 변환
torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True)

from google.colab import files
files.download('alexnet.onnx')

'''
<Google 드라이브 저장 방식>
from google.colab import drive
drive.mount('/content/drive')

# ONNX 파일을 구글 드라이브로 복사
!cp alexnet.onnx /content/drive/My\ Drive/

print("파일이 구글 드라이브에 저장되었습니다.")
'''