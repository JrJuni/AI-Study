# 선형 회귀 모델 실습
import numpy as np
import matplotlib.pyplot as plt

x = np.array([2,3,6,8])
y = np.array([81,93,91,97])

plt.scatter(x,y)
plt.show()

a = 0
b = 0

lr = 0.02

epochs = 2500

n = len(x)

for i in range(epochs):
    y_pred = a * x + b
    error = y - y_pred

    a_diff = (2/n)*sum(-x * error)
    b_diff = (2/n)*sum(-error)

    a = a - lr * a_diff
    b = b - lr * b_diff

    if i % 100 == 0:
        print("epoch=%.f, slope=%.04f, intercept=%.04f" % (i, a, b))

y_pred = a * x  + b

plt.scatter(x,y)
plt.plot(x, y_pred, 'r')
plt.show()