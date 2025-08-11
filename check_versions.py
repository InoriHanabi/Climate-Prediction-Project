import torch
import numpy
import sys

print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("NumPy version:", numpy.__version__)

# Test a very simple PyTorch operation
try:
    x = torch.rand(2, 2, device=torch.device("cpu")) # Explicitly on CPU
    y = torch.rand(2, 2, device=torch.device("cpu"))
    z = x + y
    print("Simple PyTorch CPU operation successful:", z)
except Exception as e:
    print("Error during simple PyTorch test:", e)