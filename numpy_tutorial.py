import numpy as np

# version check
# print(np.__version__)
# print(np.__name__)

# array
myarray = np.array([[1,2,3,],[1,2,4,]])

myarray.shape # 크기 확인
myarray.ndim # 차원 확인
myarray.size # 총 요소 수 확인
myarray.dtype # 데이터 타입 확인
myarray.T # transpose

# array generate
array_ar = np.arange(1,10,2)
print(array_ar)

# linspace()
array_lin = np.linspace(1, 10, 3)  # 마지막 구간 갯수

# zero one full
np.zeros(shape=(3,3))
np.ones(shape=(2,3))
test_arr =np.full(shape=(3,3), fill_value=3)

np.eye(4)  # identical matrix
np.random.randn(3,3) # random matrix

# reshape
print(test_arr.size)
test_arr.reshape(9)
test_arr.reshape(-1)  # size 자동 계산 "-1"

# Boolean Indexing
test_b = np.arange(12).reshape(3,4)
bool_b = test_b > 5
print(test_b[bool_b])

# 배열 참조 - 슬라이싱 후 인덱스의 값 변경 시 원본 데이터 변경 // 방지법
cop = np.arange(1,11)
c1 = cop[:4].copy()
c2 = cop[np.arange(5)]

# where
# np.where(condition, a, b)
test_where = np.arange(1,13).reshape(3,4)
print(np.where(test_where > 6, '크다', '작다'))

# np.sum(axis=0,1)  np.mean(axis=0,1)  0 -> row 각 열의 연산  1 -> column 각 행의 연산
arr = np.array([[1, 2, 3],
                [4, 5, 6]])
print(np.sum(arr, axis = 0))
print(np.mean(arr, axis = 1))

# arr.min  max  var  std

# save
np.save('test_arr', arr)
