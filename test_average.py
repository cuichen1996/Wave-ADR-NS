# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.misc import *
from model.model import *
from data.load_data import CreateTestLoader

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
device = "cuda" if torch.cuda.is_available() else "cpu"
print('Using GPU, %s' % torch.cuda.get_device_name()) if device == 'cuda' else print('Using CPU')
setup_seed(12345)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=128)
args = parser.parse_args()

DATASET = "stl10"
max_iter  = 50
restart   = 20
gamma_val = 0.0
batch = 10

N = args.n
if N == 128:
    omega = 20 * np.pi
    max_level = 6
elif N == 256:
    omega = 40 * np.pi
    max_level = 7
elif N == 512:
    omega = 80 * np.pi
    max_level = 8
elif N == 1024:
    omega = 160 * np.pi
    max_level = 9
elif N == 2048:
    omega = 320 * np.pi
    max_level = 10
elif N == 4096:
    omega = 640 * np.pi
    max_level = 11

m = n = N
h = 1/m
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

model = WaveADR(config).to(device)
checkpoint = "expriments/checkpoint/model.pth"
model.load_state_dict(torch.load(checkpoint, map_location=device))
model.eval()

src = [m // 2, n // 2]
test_loader = CreateTestLoader(N, batch, DATASET, src)

iters = []
setup_times = []
solve_times = []
ray_iters = []
ray_times = []

i = 0
for b, kappa in test_loader:
    gamma = absorbing_layer(gamma_val, kappa[0, 0].cpu(), [16, 16], omega)
    kappa, gamma, b = kappa.to(device), gamma.to(device), b.to(device)

    # setup
    tic = time.time()
    with torch.no_grad():
        lam_max = model.setup_cheby(kappa, gamma, omega, max_level)
        T, Tx, Ty, LT, alphas = model.setup(down(kappa.float()), omega, src, max_level)
        T, Tx, Ty, LT = T.repeat(1, 8, 1, 1), Tx.repeat(1, 8, 1, 1), Ty.repeat(1, 8, 1, 1), LT.repeat(1, 8, 1, 1)
    setup_time = time.time() - tic

    # solve
    with torch.no_grad():
        x, ress, gmres_time = model.test(
            b, kappa, omega, max_level, gamma_val, T, Tx, Ty, LT, lam_max, alphas, restart, max_iter
        )

    # wave ray
    T, Tx, Ty, LT = model.generate_T_equal(down(kappa.float()), 8)
    with torch.no_grad():
        x, ress_ray, ray_time = model.test(
            b, kappa, omega, max_level, gamma_val, T, Tx, Ty, LT, lam_max, alphas, restart, max_iter
        )

    print("WaveADR iters =", len(ress)-1, "setup time = %f" % setup_time, "solve time = %f" % gmres_time, "|| WaveRay iters =", len(ress_ray)-1, "ray time = %f" % ray_time)
    iters.append(len(ress)-1)
    setup_times.append(setup_time)
    solve_times.append(gmres_time)
    ray_iters.append(len(ress_ray)-1)
    ray_times.append(ray_time)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5)) 
    im = ax1.imshow(kappa[0,0].cpu(), cmap="jet", extent=[0,1,0,1])
    ax1.set_title(r'$1/c$', fontsize=12)
    cbar = fig.colorbar(im, ax=ax1) 
    im = ax2.imshow(x.reshape(N, N).real.cpu().detach().numpy(), cmap="jet")
    ax2.set_title(r'$u$', fontsize=12)
    cbar = fig.colorbar(im, ax=ax2) 
    ax3.semilogy(ress, label="adr", markersize=6)
    ax3.semilogy(ress_ray, label="ray", markersize=6)
    ax3.grid()
    ax3.legend()
    plt.savefig(f"expriments/image/" + DATASET+ f"_{N}_{i}.png", dpi=300, bbox_inches='tight')
    plt.close()
    i = i+1


# average
print("WaveADR iters = ", np.mean(iters), "setup time = %f" % np.mean(setup_times), "solve time = %f" % np.mean(solve_times))
print("WaveRay iters = ", np.mean(ray_iters), "waveRay time = %f" % np.mean(ray_times))