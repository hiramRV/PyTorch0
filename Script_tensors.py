import torch
import numpy as np

#Tensors are in essence ndarrays, but they can run in GPU's and other hardware acceleratos. 

# Initialize it from scratch

data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
print(x_data)

# Initialize it with a numpy array. 
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)

# From another tensor
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

# Create a tensor with a given shape and different values
shape = (2,3,)
rand_tensor = 10*torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)+2

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

## Attributes of a tensor. 
tensor = torch.rand(3,4)

#Shape, datatype, where is stored. 
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


## Operations on Tensors. 

# We move our tensor to the current accelerator if available
if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())
    print("tensor was moved!")

print(f"Device tensor is stored on: {tensor.device}")

# Some operations. Similar to numpy
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 22
print(tensor)

# Concat a pair of tensors
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

t2 = torch.stack((tensor, tensor), dim=1)
print(t2)

#Aritmetic operations
# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor@tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)
print(y3)

# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print(z3)

# Generate a single value
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))


# In place operations. Be cautions with them when using diff
print(f"{tensor} \n")
tensor.add_(100)
print(tensor)