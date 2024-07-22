# 240722

# 1. Tensor
import torch
import numpy as np

# tensor initializing
data = [[1,2],[3,4]]
x_data = torch.tensor(data)
print(x_data.dtype)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

x_ones = torch.ones_like(x_data)  #retains properties like shape, datatype
x_rand = torch.rand_like(x_data, dtype=torch.float)

# rand or constant initializing
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

# attribute
t_att = torch.rand(3,4)
print(f"Shape of tensor: {t_att.shape}")
print(f"Datatype of tensor: {t_att.dtype}")
print(f"Device tensor is stroed on: {t_att.device}")
