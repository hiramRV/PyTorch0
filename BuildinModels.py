# Example to build a neural network using PyTorch 

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Train on GPU if available
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Define a neural network class with three linear layers and ReLU activations
# Also add the forward method to define the forward pass
class NeuralNetwork(nn.Module):
    # Constructor to initialize the layers
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    # Method to define the forward pass
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
# Create an instance of the neural network
model = NeuralNetwork().to(device)
print(model)

# Create a random input tensor with the shape of (1, 28, 28)
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
# Apply softmax to get the predicted probabilities and get the predicted class
pred_probab = nn.Softmax(dim=1)(logits)
# Print the probabilities and the predicted class
print(f"Predicted probabilities: {pred_probab}")
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}, value: {pred_probab[0][y_pred].item()}")


# Exploring model layers step by step:


# Generate a random input image
input_image = torch.rand(1,28,28)
print(input_image.size())

# Flattening the input image to a vector
flatten = nn.Flatten()
flat_image = flatten(input_image)
#print(flat_image.size())

# Creating a linear layer with 28*28 input features and 20 output features
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
#print(hidden1.size())

# Applying ReLU activation function
#print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
#print(f"After ReLU: {hidden1}")

# Creating a sequential model: flatten -> layer1 -> ReLU -> Linear
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
#input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

# Last layer: Applying softmax to get the predicted probabilities
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted probabilities: {pred_probab}")
print(f'Predicted class: {y_pred}')

print(f"Model structure: {model}\n\n")
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")