# Source: # Data transformation is a common operation applied to data before it is used to train a model.

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

# The FashionMNIST features are in PIL Image format, and the labels are integers.
# We need to transform them to Tensors for training.
# transform: The features are normalized into the range of [0, 1]
# target_transform: The labels are one-hot encoded
ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

# Print the label of the fist 10 samples
for i in range(10):
    print(f'One hot encode: {ds[i][1]}')
