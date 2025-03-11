# %%
import numpy as np
import torch
from data import KappaDataGenerator

# %%
N = 128
h = 1 / N

dataset = KappaDataGenerator(N, N)
dataset.load_data(data="cifar10")
kappa = dataset.generate_kappa().to(torch.float32)

data_kappa = np.zeros((16000, 1, N, N))
for i in range(16000):
    data_kappa[i, 0] = dataset.generate_kappa().to(torch.float32).reshape(N, N)

np.save(f"kappa{N}", data_kappa)

src = [N // 2, N // 2]
b = np.zeros((1, 1, N, N), dtype=np.complex64)
b[0, 0, src[0], src[1]] = 1.0 / h**2
np.save(f"b{N}", b)

# %%
N = 256
h = 1 / N

dataset = KappaDataGenerator(N, N)
dataset.load_data(data="cifar10")
kappa = dataset.generate_kappa().to(torch.float32)

data_kappa = np.zeros((10000, 1, N, N))
for i in range(10000):
    data_kappa[i, 0] = dataset.generate_kappa().to(torch.float32).reshape(N, N)
np.save(f"kappa{N}", data_kappa)

src = [N // 2, N // 2]
b = np.zeros((1, 1, N, N), dtype=np.complex64)
b[0, 0, src[0], src[1]] = 1.0 / h**2

np.save(f"b{N}", b)


# %%
N = 512
h = 1 / N

dataset = KappaDataGenerator(N, N)
dataset.load_data(data="cifar10")
kappa = dataset.generate_kappa().to(torch.float32)

data_kappa = np.zeros((6000, 1, N, N))
for i in range(6000):
    data_kappa[i, 0] = dataset.generate_kappa().to(torch.float32).reshape(N, N)
np.save(f"kappa{N}", data_kappa)

src = [N // 2, N // 2]
b = np.zeros((1, 1, N, N), dtype=np.complex64)
b[0, 0, src[0], src[1]] = 1.0 / h**2

np.save(f"b{N}", b)
