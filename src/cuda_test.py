import torch


print('CUDA version:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
