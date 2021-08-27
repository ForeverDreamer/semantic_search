import torch
import numpy as np

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(f'x_data: {x_data}\n')

np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(f'x_np: {x_np}\n')

x_ones = torch.ones_like(x_data)  # retains the properties of x_data
print(f"Ones Tensor: {x_ones}\n")

x_rand = torch.rand_like(x_data, dtype=torch.float)  # overrides the datatype of x_data
print(f"Random Tensor: {x_rand}\n")

shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: {rand_tensor}")
print(f"Ones Tensor: {ones_tensor}")
print(f"Zeros Tensor: {zeros_tensor}\n")

tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}\n")

tensor = torch.ones(4, 4)
print('First row: ', tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1])
tensor[:, 1] = 0
print(tensor)
print()

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
print()

# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)
print('矩阵乘法：\n')
print(y1)
print(y2)
print(y3)


# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print('点乘法：\n')
print(z1)
print(z2)
print(z3)
print()

agg = tensor.sum()
agg_item = agg.item()
print('Single-element tensors')
print(agg_item, type(agg_item))
print()

print('In-place operations')
print(tensor, "\n")
tensor.add_(5)
print(tensor)
print()

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")
print()

n = np.ones(5)
t = torch.from_numpy(n)
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
