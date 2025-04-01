import torch

if torch.cuda.is_available():
    print("CUDA is available!  ✨")
    device = torch.device("cuda")  # Set device to GPU
    x = torch.randn(10, device=device)  # Create a tensor on the GPU
    print(x)
else:
    print("CUDA is not available. 😔")