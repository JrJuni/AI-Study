import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 수술 환자 데이터를 불러옵니다.
Data_set = np.loadtxt("./data/ThoraricSurgery3.csv", delimiter=",")  # 수술 환자 데이터를 불러옵니다.
X = Data_set[:,0:16]                                                 # 환자의 진찰 기록을 X로 지정합니다.
y = Data_set[:,16]                                                   # 수술 후 사망/생존 여부를 y로 지정합니다.


# NumPy 배열을 PyTorch 텐서로 변환합니다.
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# 데이터셋과 데이터로더를 정의합니다.
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 모델 정의
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(16, 30)
        self.fc2 = nn.Linear(30, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# 모델 인스턴스 생성
model = SimpleNN()

# 손실 함수와 옵티마이저 정의
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 학습
num_epochs = 5

for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        # 순전파
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)

        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# 모델 평가
model.eval()
with torch.no_grad():
    outputs = model(X_tensor)
    predicted = (outputs.squeeze() >= 0.5).float()
    accuracy = (predicted == y_tensor).float().mean()
    print(f'Accuracy: {accuracy.item()*100:.2f}%')

'''
[해설]

데이터 로드 및 전처리:

np.loadtxt를 사용하여 CSV 파일에서 데이터를 로드합니다.
데이터를 PyTorch 텐서로 변환합니다.
TensorDataset과 DataLoader를 사용하여 데이터셋과 데이터로더를 정의합니다.
모델 정의:

nn.Module을 상속받아 모델 클래스를 정의합니다.
두 개의 완전 연결층과 활성화 함수를 사용합니다.
손실 함수와 옵티마이저 정의:

BCELoss를 사용하여 이진 교차 엔트로피 손실을 정의합니다.
Adam 옵티마이저를 사용하여 모델 파라미터를 최적화합니다.
모델 학습:

에포크 수와 배치 크기를 설정하여 모델을 학습시킵니다.
각 배치에 대해 순전파, 손실 계산, 역전파, 최적화를 수행합니다.
모델 평가:

학습이 완료된 모델을 평가 모드로 전환합니다.
예측값과 실제값을 비교하여 정확도를 계산합니다.
'''