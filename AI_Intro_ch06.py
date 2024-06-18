import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# 데이터 준비
x = np.array([2, 4, 6, 8, 10, 12, 14])
y = np.array([0, 0, 0, 1, 1, 1, 1])

# PyTorch 모델 정의
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.fc = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

# 모델 인스턴스 생성
model = BinaryClassifier()

# 손실 함수와 최적화기 정의
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 모델 학습
for epoch in range(5000):
    inputs = torch.from_numpy(x.reshape(-1, 1)).float()
    targets = torch.from_numpy(y.reshape(-1, 1)).float()

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

# 결과 시각화
plt.scatter(x, y)
plt.plot(x, outputs.detach().numpy(), 'r')
plt.show()

# 예측
hour = 7
prediction = model(torch.tensor([hour], dtype=torch.float32)).item()
print(f"{hour}시간을 공부할 경우, 합격 예상 확률은 {prediction*100:.01f}%입니다.")

'''
1. PyTorch에서는 nn.Module을 상속받아 모델을 정의합니다. 이 경우 BinaryClassifier 클래스를 만들었습니다.
2. 모델 내부에 nn.Linear 레이어를 사용하여 입력 크기 1, 출력 크기 1의 완전 연결 레이어를 정의했습니다. 그리고 nn.Sigmoid 활성화 함수를 사용했습니다.
3. 손실 함수로 nn.BCELoss를 사용했고, 최적화기로 optim.SGD를 사용했습니다.
4. 모델 학습 시 torch.from_numpy를 사용하여 NumPy 배열을 PyTorch 텐서로 변환했습니다.
5. 예측 시 model(torch.tensor([hour], dtype=torch.float32)).item()을 사용하여 예측 결과를 얻었습니다.
'''

'''
<원본>
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array([2, 4, 6, 8, 10, 12, 14])
y = np.array([0, 0, 0, 1, 1, 1, 1]) 

model = Sequential()
model.add(Dense(1, input_dim=1, activation='sigmoid'))

# 교차 엔트로피 오차 함수를 이용하기 위하여 'binary_crossentropy'로 설정합니다. 
model.compile(optimizer='sgd' ,loss='binary_crossentropy')
model.fit(x, y, epochs=5000)

# 그래프로 확인해 봅니다.
plt.scatter(x, y)
plt.plot(x, model.predict(x),'r')
plt.show()

# 임의의 학습 시간을 집어넣어 합격 예상 확률을 예측해 보겠습니다.
hour = 7
prediction = model.predict([hour])

print("%.f시간을 공부할 경우, 합격 예상 확률은 %.01f%%입니다" % (hour, prediction * 100))
'''