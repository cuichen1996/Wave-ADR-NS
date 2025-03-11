# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from data.load_data import CreateTestLoader
from model.model import *
from utils.misc import *

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = "cuda" if torch.cuda.is_available() else "cpu"
print('Using GPU, %s' % torch.cuda.get_device_name()) if device == 'cuda' else print('Using CPU')
# %%
DATASET = "stl10"
max_iter  = 50
restart   = 20
gamma_val = 0.0

N = 128
h = 1 / N
m = n = N
omega = 20*np.pi
max_level = 6
src = [m // 2, n // 2]
batch = 1

test_loader = CreateTestLoader(N, batch, DATASET, src)
for b, kappa in test_loader:
    b, kappa = b, kappa

gamma = absorbing_layer(gamma_val, kappa[0, 0].cpu(), [16, 16], omega)
kappa, gamma, b = kappa.to(device), gamma.to(device), b.to(device)

# %%
src1 = [m // 2, n // 2]
b1 = torch.zeros(1, 1, m, n, dtype=torch.cfloat, device=device)
b1[:,:, src1[0],src1[1]] = 1.0 / h**2

src2 = [m // 4, n // 4]
b2 = torch.zeros(1, 1, m, n, dtype=torch.cfloat, device=device)
b2[:,:, src2[0],src2[1]]  = 1.0 / h**2

b3 =  torch.randn_like(kappa, dtype=torch.cfloat)
# %%
config = {}
config["M"]              = 1
config["NO_Type"]        = "FNO" 
config["modes"]          = [12,12,12,12]
config["depths"]         = [3,3,9,3]
config["dims"]           = [36,36,36,36]
config["drop_path_rate"] = 0.3
config["drop"]           = 0.
config["padding"]        = 9
config["act"]            = "gelu"
config["xavier_init"]    = 1e-2

config["max_iter_num"]    = 100
config["error_threshold"] = 1e-6

model = WaveADR(config).to(device)
total_params = count_parameters(model)
print("Total parameters: ", total_params)

checkpoint = "expriments/checkpoint/model.pth"
model.load_state_dict(torch.load(checkpoint, map_location=device))
model.eval()

# %%
lam_max = model.setup_cheby(kappa, gamma, omega, max_level)

with torch.no_grad():
    T, Tx, Ty, LT, alphas = model.setup(down(kappa.float()), omega, src1, max_level)
    T, Tx, Ty, LT = T.repeat(1, 8, 1, 1), Tx.repeat(1, 8, 1, 1), Ty.repeat(1, 8, 1, 1), LT.repeat(1, 8, 1, 1)
    x, ress1, gmres_time = model.test(
        b1, kappa, omega, max_level, gamma_val, T, Tx, Ty, LT, lam_max, alphas, restart, max_iter
    )

with torch.no_grad():
    T, Tx, Ty, LT, alphas = model.setup(down(kappa.float()), omega, src2, max_level)
    T, Tx, Ty, LT = T.repeat(1, 8, 1, 1), Tx.repeat(1, 8, 1, 1), Ty.repeat(1, 8, 1, 1), LT.repeat(1, 8, 1, 1)
    x, ress2, gmres_time = model.test(
        b2, kappa, omega, max_level, gamma_val, T, Tx, Ty, LT, lam_max, alphas, restart, max_iter
    )

with torch.no_grad():
    T, Tx, Ty, LT, alphas = model.setup(down(kappa.float()), omega, src1, max_level)
    T, Tx, Ty, LT = T.repeat(1, 8, 1, 1), Tx.repeat(1, 8, 1, 1), Ty.repeat(1, 8, 1, 1), LT.repeat(1, 8, 1, 1)
    x, ress3, gmres_time = model.test(
        b3, kappa, omega, max_level, gamma_val, T, Tx, Ty, LT, lam_max, alphas, restart, max_iter
    )


plt.semilogy(ress1,  label="Centered Point Source")
plt.semilogy(ress2,  label="Random Point Source")
plt.semilogy(ress3,  label="Ramdom Source")
plt.xlabel('Iteration', fontsize=20)
plt.ylabel('Relative Residual', fontsize=20)
plt.grid()
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig("expriments/image/" + DATASET + "_rhs.png", dpi=300, bbox_inches='tight')
plt.close()

