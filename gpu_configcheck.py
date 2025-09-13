import torch
import time

print("=== OPTIMIZATION CHECK ===")
print(f"MKL Available: {torch.backends.mkl.is_available()}")
print(f"OpenMP Available: {torch.backends.openmp.is_available()}")
print(f"cuDNN Available: {torch.backends.cudnn.is_available()}")
print(f"Built with CUDA: {torch.version.cuda is not None}")

print("\n=== PERFORMANCE TEST ===")
device = torch.device('cuda')
x = torch.randn(5000, 5000, device=device)
y = torch.randn(5000, 5000, device=device)

torch.cuda.synchronize()
start = time.time()
z = torch.mm(x, y)
torch.cuda.synchronize()
end = time.time()

print(f"5000x5000 matrix multiply: {end-start:.3f}s")
print(f"Expected: ~0.1-0.3s for good performance")

del x, y, z
torch.cuda.empty_cache()