import torch

if torch.cuda.is_available():
    print("CUDA is available!")
    print("CUDA version:", torch.version.cuda)
else:
    print("CUDA is not available!")
    print("CUDA version:", torch.version.cuda)
    
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(1))

print(torch.cuda.get_device_name(0))