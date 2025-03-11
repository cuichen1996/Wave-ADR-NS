import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_
import time
from model.torch_fgmres import fgmres_res
from model.unet import UNet, CNN
from model.FNO import sFNO_epsilon_v2

def leb_shuffle_2n(n):
    if n == 1:
        return np.array([0], dtype=int)
    else:
        prev = leb_shuffle_2n(n // 2)
        ans = np.zeros(n, dtype=int)
        ans[::2] = prev
        ans[1::2] = n - 1 - prev
        return ans

def get_grid2D(shape, device):
    batchsize, size_x, size_y = shape[0], shape[2], shape[3]
    if size_x == size_y:
        x = np.linspace(0, 1, size_y, endpoint=True, dtype=np.float32)  
    else:
        x = np.linspace(0, 3, size_y, endpoint=True, dtype=np.float32)  
    y = np.linspace(0, 1, size_x, endpoint=True, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    gridx = torch.from_numpy(X).repeat([batchsize, 1, 1, 1])
    gridy = torch.from_numpy(Y).repeat([batchsize, 1, 1, 1])
    return gridx.to(device), gridy.to(device)

def absorbing_layer(gamma_val, kappa, pad, ABLamp, NeumannAtFirstDim=False):
    n = kappa.T.size()
    gamma = gamma_val * ABLamp * torch.ones_like(kappa.T)
    b_bwd1 = ((torch.arange(pad[0], 0, -1)) ** 2) / pad[0] ** 2
    b_bwd2 = ((torch.arange(pad[1], 0, -1)) ** 2) / pad[1] ** 2
    b_fwd1 = ((torch.arange(1, pad[0] + 1)) ** 2) / pad[0] ** 2
    b_fwd2 = ((torch.arange(1, pad[1] + 1)) ** 2) / pad[1] ** 2
    I1 = torch.arange(n[0] - pad[0], n[0])
    I2 = torch.arange(n[1] - pad[1], n[1])
    if not NeumannAtFirstDim:
        gamma[:, : pad[1]] += torch.outer(torch.ones(n[0]), b_bwd2) * ABLamp
        gamma[: pad[0], : pad[1]] -= torch.outer(b_bwd1, b_bwd2) * ABLamp
        gamma[I1, : pad[1]] -= torch.outer(b_fwd1, b_bwd2) * ABLamp
    gamma[:, I2] += torch.outer(torch.ones(n[0]), b_fwd2) * ABLamp
    gamma[: pad[0], :] += torch.outer(b_bwd1, torch.ones(n[1])) * ABLamp
    gamma[I1, :] += torch.outer(b_fwd1, torch.ones(n[1])) * ABLamp
    gamma[: pad[0], I2] -= torch.outer(b_bwd1, b_fwd2) * ABLamp
    gamma[I1.view(-1, 1), I2.view(1, -1)] -= torch.outer(b_fwd1, b_fwd2) * ABLamp
    return gamma.T.unsqueeze(0).unsqueeze(0)


class Helmholtz(nn.Module):
    def __init__(self, kappa: torch.Tensor, gamma: torch.Tensor, omega: float):
        super().__init__()
        self.degree = 16
        self.kappa = kappa
        self.omega = omega
        self.gamma = gamma.to(kappa.device, dtype=kappa.dtype)
        self.h = 1 / kappa.shape[-2]
        self.device = kappa.device
        self.laplace_kernel = (1.0 / (self.h**2)) * torch.tensor(
            [[[[0, -1.0, 0], [-1, 4, -1], [0, -1, 0]]]]
        )

    def generate_helmholtz_matrix(self):
        return ((self.kappa**2) * self.omega) * (self.omega - self.gamma * 1j)

    def generate_shifted_laplacian(self, alpha=0.5):
        return ((self.kappa**2) * self.omega) * (
            (self.omega - self.gamma * 1j) - (1j * self.omega * alpha)
        )

    def matvec(self, X: torch.Tensor, SL=False) -> torch.Tensor:
        convolution = F.conv2d(X, self.laplace_kernel.to(X.device, X.dtype), padding=1)
        if SL:
            return convolution - self.generate_shifted_laplacian() * X
        return convolution - self.generate_helmholtz_matrix() * X

    def matvec_conj(self, X: torch.Tensor) -> torch.Tensor:
        original_shape = X.shape
        if X.dim() == 1:
            X = X.reshape_as(self.kappa)
        if X.dim() == 2:
            X = X.unsqueeze(0).unsqueeze(0)
        convolution = F.conv2d(X, self.laplace_kernel.to(X.device, X.dtype), padding=1)
        matrix = self.generate_helmholtz_matrix()
        return (convolution - torch.conj(matrix) * X).reshape(original_shape)

    def forward(self, x, SL=False):
        batch, c, m, n = self.kappa.shape
        dims = x.dim()
        if dims == 2:
            x = x.view(m, n, c, batch).permute(3, 2, 0, 1)
        elif dims == 1:
            x = x.view(1, 1, m, n)
        if SL:
            y = self.matvec(x, SL)
        else:
            y = self.matvec(x)

        if dims == 2:
            return y.permute(2, 3, 1, 0).reshape(-1, batch)
        elif dims == 1:
            return y.flatten()
        else:
            return y

    def helm_normal(self, x):
        x = self.matvec_conj(self.matvec(x))
        return x

    def generate_adr_matrix(self, Tx, Ty, LT):
        reac = 1j * self.omega * (self.gamma * self.kappa * self.kappa + LT)
        eik = (Tx**2 + Ty**2 - self.kappa**2) * self.omega**2
        return reac + eik

    def ad_diff(self, x, Tx, Ty):
        # * 1-st upwind
        a2 = -1 * Ty - abs(Ty)
        a4 = -1 * Tx - abs(Tx)
        a5 = 2 * (abs(Tx) + abs(Ty))
        a6 = Tx - abs(Tx)
        a8 = Ty - abs(Ty)

        X = F.pad(x, (1, 1, 1, 1))  
        m, n = x.shape[-2], x.shape[-1]
        u2 = a2 * X[:, :, :m, 1 : n + 1]
        u4 = a4 * X[:, :, 1 : m + 1, :n]
        u5 = a5 * X[:, :, 1 : m + 1, 1 : n + 1]
        u6 = a6 * X[:, :, 1 : m + 1, 2 : n + 2]
        u8 = a8 * X[:, :, 2 : m + 3, 1 : n + 1]
        advec = 2j * self.omega * (u2 + u4 + u5 + u6 + u8) / (2 * self.h)
        diff = F.conv2d(x, self.laplace_kernel.to(x.device, x.dtype), padding=1)
        return advec + diff
    
    def adr_forward(self, x, Tx, Ty, LT):
        batch, m, n = Tx.shape[0], Tx.shape[-2], Tx.shape[-1]
        dims = x.dim()
        if dims == 2:
            x = x.view(m, n, 1, batch).permute(3, 2, 0, 1)
        AD = self.ad_diff(x, Tx, Ty)
        R = self.generate_adr_matrix(Tx, Ty, LT)
        y = AD + R * x
        if dims == 2:
            return y.permute(2, 3, 1, 0).reshape(-1, batch)
        else:
            return y

    def power_method(self):
        with torch.no_grad():
            b_k = torch.randn(self.kappa.shape, dtype=torch.cfloat, device=self.device)
            for i in range(50):
                b_k1 = self.helm_normal(b_k)
                b_k = b_k1 / torch.norm(b_k1)
            mu = torch.inner(
                self.helm_normal(b_k).flatten(), b_k.flatten()
            ) / torch.inner(b_k.flatten(), b_k.flatten())
        return mu.item().real

    def chebysemi(self, x, b, max_iter, lam_max, alpha):
        lam_min = lam_max / alpha
        roots = [np.cos((np.pi * (2 * i + 1)) / (2 * self.degree)) for i in range(self.degree)]
        good_perm_even = leb_shuffle_2n(self.degree)
        taus = [2 / (lam_max + lam_min - (lam_min - lam_max) * r) for r in roots]
        b = self.matvec_conj(b)
        for i in range(max_iter):
            r = b - self.helm_normal(x)
            x = x + taus[good_perm_even[i]].unsqueeze(-1).unsqueeze(-1) * r
        return x

    def jacobi(self, x, b, max_iter, w, SL=False):
        D = 4 / self.h**2
        if SL:
            R = (self.omega**2 * self.kappa**2) * (1-0.5j)
        else:
            R = self.omega**2 * self.kappa**2
        Dinv = 1 / (D - R)
        for i in range(max_iter):
            r = b - self.forward(x, SL)
            x = x + w * Dinv * r
        return x

    def adr_correction(self, r, T, Tx, Ty, LT, max_level):
        r_adr = r * torch.exp(1j * self.omega * T)
        ah = v_cycle_adr(r_adr, self.kappa, self.gamma, self.omega, Tx, Ty, LT, 1, max_level)
        e = ah * torch.exp(-1j * self.omega * T)
        return e

# * MG SOLVER * #
def down(matrix: torch.Tensor):
    down_kernel = torch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) * 1.0 / 16
    return F.conv2d(
        matrix,
        down_kernel.to(device=matrix.device, dtype=matrix.dtype),
        stride=2,
        padding=1,
    )

def up(matrix: torch.Tensor):
    up_kernel = torch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) * 1.0 / 4
    return F.conv_transpose2d(
        matrix,
        up_kernel.to(matrix.device, matrix.dtype),
        stride=2,
        padding=1,
        output_padding=1,
    )

def fgmres_helm(A, B, restrt=20, max_iter=10, tol=1e-6, M=None, SL=False, x0=None, verbose=False):
    batch, m, n = B.shape[0], B.shape[-2], B.shape[-1]
    N = m * n
    device = B.device
    if x0 is None:
        x0 = torch.zeros_like(B, device=device, dtype=B.dtype)
    B = B.permute(2, 3, 1, 0).reshape(N, batch)
    x0 = x0.permute(2, 3, 1, 0).reshape(N, batch)

    if M is None:
        def M(v, tmp=None):
            return v

    r0 = B - A(x0, SL)
    beta = torch.linalg.norm(r0, dim=0)

    res1 = torch.mean(beta)
    ress = [1.0]
    for i in range(max_iter):
        rhs = torch.zeros(restrt + 1, batch, device=device, dtype=B.dtype)
        rhs[0] = beta
        Q = torch.zeros((N, restrt + 1, batch), device=device, dtype=B.dtype)
        # Preconditioned basis
        Z = torch.zeros_like(Q, device=device, dtype=B.dtype)
        # Hessenberg matrix
        H = torch.zeros((restrt + 1, restrt, batch), device=device, dtype=B.dtype)
        for k in range(restrt):
            if k == 0:
                Q[:, k] = r0 / beta
            else:
                Q[:, k] = ww
            Z[:, k] = M(Q[:, k])
            ww = A(Z[:, k], SL)

            # Modified Gram Schmidt Orthogonalize
            for j in range(k + 1):
                H[j, k] = torch.inner(
                    ww.transpose(1, 0), Q[:, j].transpose(1, 0).conj()
                ).diag()
                ww = ww - H[j, k] * Q[:, j]
            beta = torch.linalg.norm(ww, dim=0)
            ww = ww / beta
            H[k + 1, k] = beta

        # Build the upper-triangular system to solve
        U = H.permute(2, 0, 1)
        rhs_ = rhs.permute(1, 0)
        y = torch.linalg.lstsq(U, rhs_, rcond=None)[0]

        # Update the solution estimate
        x0 = x0 + torch.einsum(
            "ijk,ikh->ijh", Z[:, :restrt].permute(2, 0, 1), y.unsqueeze(-1)
        ).squeeze(-1).permute(1, 0)
        r0 = B - A(x0, SL)
        beta = torch.linalg.norm(r0, dim=0)
        res = torch.mean(beta) / res1
        ress.append(res.cpu().item())
        if verbose:
            print(f"res: {ress[-1]}")
        if res < tol:
            break
    return x0.view(m, n, 1, batch).permute(3, 2, 0, 1), ress


# **  ADR SOLVER **
def v_cycle_adr(b, kappa, gamma, omega, Tx, Ty, LT, level, max_level, x=None):
    if x is None:
        x = torch.zeros_like(b, dtype=b.dtype, device=b.device)
    Helm = Helmholtz(kappa, gamma, omega)

    if level == max_level:
        x = fgmres(Helm.adr_forward, Tx, Ty, LT, b, restrt=10, max_iter=1, x0=x)[0]
        return x
    else:
        x = fgmres(Helm.adr_forward, Tx, Ty, LT, b, restrt=3, max_iter=1, x0=x)[0]
        r = b - Helm.adr_forward(x, Tx, Ty, LT)
        rc = down(r)
        kappa = down(kappa)
        gamma = down(gamma)
        Tx = down(Tx)
        Ty = down(Ty)
        LT = down(LT)
        ec = v_cycle_adr(rc, kappa, gamma, omega, Tx, Ty, LT, level + 1, max_level)
        e = up(ec)
        x = x + e
        Tx = up(Tx)
        Ty = up(Ty)
        LT = up(LT)
        x = fgmres(Helm.adr_forward, Tx, Ty, LT, b, restrt=3, max_iter=1, x0=x)[0]
        del Helm, r, rc, ec, e
    return x


def fgmres(A, Tx, Ty, LT, B, restrt=20, max_iter=10, tol=1e-6, M=None, x0=None):
    # * FGMRES for ADR equation
    batch, m, n = Tx.shape[0], Tx.shape[-2], Tx.shape[-1]
    N = m * n
    device = B.device
    if x0 is None:
        x0 = torch.zeros_like(B, device=device, dtype=B.dtype)
    B = B.permute(2, 3, 1, 0).reshape(N, batch)
    x0 = x0.permute(2, 3, 1, 0).reshape(N, batch)

    def mv(v):
        return A(v, Tx, Ty, LT)

    if M is None:
        def M(v):
            return v

    r0 = B - mv(x0)
    beta = torch.linalg.norm(r0, dim=0)

    res1 = torch.mean(beta)
    ress = [res1]
    for i in range(max_iter):
        rhs = torch.zeros(restrt + 1, batch, device=device, dtype=B.dtype)
        rhs[0] = beta
        Q = torch.zeros((N, restrt + 1, batch), device=device, dtype=B.dtype)
        # Preconditioned basis
        Z = torch.zeros_like(Q, device=device, dtype=B.dtype)
        # Hessenberg matrix
        H = torch.zeros((restrt + 1, restrt, batch), device=device, dtype=B.dtype)
        for k in range(restrt):
            if k == 0:
                Q[:, k] = r0.clone() / beta
            else:
                Q[:, k] = ww.clone()
            Z[:, k] = M(Q[:, k].clone())  # preconditioner step
            ww = mv(Z[:, k].clone())
            # Modified Gram Schmidt Orthogonalize
            for j in range(k + 1):
                H[j, k] = torch.inner(
                    ww.clone().transpose(1, 0), Q[:, j].clone().transpose(1, 0).conj()
                ).diag()
                ww = ww - H[j, k].clone() * Q[:, j].clone()
            beta = torch.linalg.norm(ww, dim=0)
            ww = ww / beta
            H[k + 1, k] = beta.clone()

        # Build the upper-triangular system to solve
        U = H.permute(2, 0, 1)
        rhs_ = rhs.permute(1, 0)
        y = torch.linalg.lstsq(U, rhs_, rcond=None)[0]

        # Update the solution estimate
        x0 = x0 + torch.einsum(
            "ijk,ikh->ijh", Z[:, :restrt].permute(2, 0, 1), y.unsqueeze(-1)
        ).squeeze(-1).permute(1, 0)
        r0 = B - mv(x0)
        beta = torch.linalg.norm(r0, dim=0)
        res = torch.mean(beta) / res1
        ress.append(res)
        if res < tol:
            break
    return x0.view(m, n, 1, batch).permute(3, 2, 0, 1), ress


def vcycle_wave_adr(b, kappa, gamma, omega, level, max_level, T, Tx, Ty, LT, lams, alphas, plot=False, x=None):
    if x is None:
        x = torch.zeros_like(b, dtype=b.dtype, device=b.device)
    Helm = Helmholtz(kappa, gamma, omega)
    # * pre-smoother
    if level == 1:
        w = 2/3
        x = Helm.jacobi(x, b, 3, w)
    elif level == max_level:
        x = Helm.chebysemi(x, b, 10, lams[level-1], alphas[level-3])
        return x
    elif level == 3:
        x = x
    else:
        x = Helm.chebysemi(x, b, 5, lams[level-1], alphas[level-2])
    # * coarse grid correction
    r = b - Helm(x)
    rc = down(r)
    kappa = down(kappa)
    gamma = down(gamma)
    ec = vcycle_wave_adr(rc, kappa, gamma, omega, level+1, max_level, T, Tx, Ty, LT, lams, alphas, plot)
    e = up(ec)
    x = x + e
    # * post-smoother
    if level == 1:
        x = Helm.jacobi(x, b, 3, w)
    elif level == 2:
        x = Helm.chebysemi(x, b, 5, lams[level-1], alphas[level-2])
        # * ADR correction
        r = b - Helm(x)
        e = Helm.adr_correction(r, T[:,0:1], Tx[:,0:1], Ty[:,0:1], LT[:,0:1], max_level-level)
        x = x + e
        #* multiplicative correction
        for i in range(1, T.shape[1]):
            r = b - Helm(x)
            e = Helm.adr_correction(r, T[:,i:i+1], Tx[:,i:i+1], Ty[:,i:i+1], LT[:,i:i+1], max_level-level)
            x = x + e
    elif level == 3:
        x = x
    else:
        x = Helm.chebysemi(x, b, 5, lams[level-1], alphas[level-2])
        del Helm, r, rc, ec, e
    return x


class WaveADR(nn.Module):
    def __init__(self, config):
        super(WaveADR, self).__init__()
        self.act = config["act"]
        self.M = config["M"]

        if config["NO_Type"] == "UNet":
            self.neural_operator = UNet(3, 4, 3, config["act"])
            print("Using UNet")
        elif config["NO_Type"] == "FNO":
            self.neural_operator = sFNO_epsilon_v2(config, 3, 4)
            print("Using Fourier Neural Operator")

        self.meta_alpha = CNN(8, 3, "relu")
        
        self.xavier_init = config["xavier_init"]
        if self.xavier_init > 0:
            self._reset_parameters()

    def forward(self, f, kappa, u):
        N = kappa.shape[-2]
        if N == 128:
            omega = 20 * np.pi
            max_level = 6
        elif N == 256:
            omega = 40 * np.pi
            max_level = 7
        else:
            omega = 80 * np.pi
            max_level = 8
        gamma_val = 0.0
        gamma = absorbing_layer(gamma_val, kappa[0, 0].cpu(), [16, 16], omega)
        Helm = Helmholtz(kappa, gamma, omega)

        T, Tx, Ty, LT, alphas = self.setup(down(kappa), omega, [N//2, N//2], max_level)
        lams = self.setup_cheby(kappa, gamma, omega, max_level)
        x = torch.zeros_like(u, dtype=u.dtype, device=u.device)
        for i in range(3):
            x = vcycle_wave_adr(f, kappa, gamma, omega, 1, max_level, T, Tx, Ty, LT, lams, alphas, False, x)
        r = f - Helm(x)
        res = torch.norm(r) / torch.norm(f)
        return res

    def constant_T(self, m, n, src):
        h = 1 / (m - 1)
        source1 = (src[0]) * h
        source2 = (src[1]) * h
        X1, X2 = torch.meshgrid(torch.arange(0, m) * h - source1, torch.arange(0, n) * h - source2)

        T = torch.sqrt(X1**2 + X2**2)
        L = 1 / T
        L[src[0], src[1]] = (
            2* (
                h * torch.asinh(torch.tensor([1.0]))
                + h * torch.asinh(torch.tensor([1.0]))
            )
            / (h**2)
        )
        G1 = X1 * L
        G1[src[0], src[1]] = 1 / 2**0.5
        G2 = X2 * L
        G2[src[0], src[1]] = 1 / 2**0.5

        T = T.unsqueeze(0).unsqueeze(0)
        if m == n:
            Tx = G1.T.unsqueeze(0).unsqueeze(0)
            Ty = G2.T.unsqueeze(0).unsqueeze(0)
        else:
            Tx = G1.unsqueeze(0).unsqueeze(0)
            Ty = G2.unsqueeze(0).unsqueeze(0) 
        LT = L.unsqueeze(0).unsqueeze(0)
        return T, Tx, Ty, LT
    
    def generate_T_equal(self, kappa, M):
        m, n = kappa.shape[-2], kappa.shape[-1]
        if m==n:
            X, Y = np.meshgrid(
                np.linspace(0, 1, n, endpoint=True), np.linspace(0, 1, m, endpoint=True)
            )
        else:
            X, Y = np.meshgrid(
                np.linspace(0, 3, n, endpoint=True), np.linspace(0, 1, m, endpoint=True)
            )
        Ts = []
        Txs = []
        Tys = []
        LTs = []
        thetas = [2 * m * np.pi / M for m in range(M)] # Equidistant Nodes
        # thetas = (np.cos(np.pi * (np.arange(M) + 0.5) / M) + 1) * np.pi # Chebyshev Nodes
        for theta in thetas:
            T = np.cos(theta) * X + np.sin(theta) * Y
            Tx = np.cos(theta) * np.ones_like(X)
            Ty = np.sin(theta) * np.ones_like(Y)
            Ts.append(torch.from_numpy(T).unsqueeze(0).unsqueeze(0))
            Txs.append(torch.from_numpy(Tx).unsqueeze(0).unsqueeze(0))
            Tys.append(torch.from_numpy(Ty).unsqueeze(0).unsqueeze(0))
            LTs.append(torch.from_numpy(np.zeros_like(T)).unsqueeze(0).unsqueeze(0))
        Ts = torch.cat(Ts, dim=1).to(kappa.device)
        Txs = torch.cat(Txs, dim=1).to(kappa.device)
        Tys = torch.cat(Tys, dim=1).to(kappa.device)
        LTs = torch.cat(LTs, dim=1).to(kappa.device)
        return Ts, Txs, Tys, LTs
    
    def setup_cheby(self, kappa, gamma, omega, max_level):
        lam_max = []
        for i in range(max_level):
            Helm = Helmholtz(kappa, gamma, omega)
            lam_max.append(Helm.power_method())
            kappa = down(kappa)
            gamma = down(gamma)
        return lam_max
    
    def setup(self, kappa, omega, src, max_level):
        gridx, gridy = get_grid2D(kappa.shape, kappa.device)
        no_input = torch.cat((kappa, gridx, gridy), 1)
        y = self.neural_operator(no_input)
        T1, Tx1, Ty1, LT1 = y[:,0:1,:,:], y[:,1:2,:,:], y[:,2:3,:,:], y[:,3:4,:,:]

        m, n = kappa.shape[-2], kappa.shape[-1]
        i, j = src[0]//2, src[1]//2
        T0, Tx0, Ty0, LT0 = self.constant_T(m, n, [i, j])
        Tx0[:,:,i, j] = 0
        Ty0[:,:,i, j] = 0
        T0 = T0.to(kappa.device)
        Tx0 = Tx0.to(kappa.device)
        Ty0 = Ty0.to(kappa.device)
        LT0 = LT0.to(kappa.device)

        T = T0 * T1
        Tx = Tx0 * T1 + T0 * Tx1
        Ty = Ty0 * T1 + T0 * Ty1
        LT = LT0 * T1 + 2 * (Tx0 * Tx1 + Ty0 * Ty1) + T0 * LT1

        alphas = []
        for i in range(max_level-2):
            N = kappa.shape[-1]
            alpha = self.meta_alpha(kappa.float(), omega, N)
            alphas.append(alpha)
            kappa = down(kappa)
        return T, Tx, Ty, LT, alphas
        

    def test(self, f, kappa, omega, max_level, gamma_val, T, Tx, Ty, LT, lams, alphas, restart, max_iter):
        m, n = f.shape[-2:]
        gamma = absorbing_layer(gamma_val, kappa[0, 0].cpu(), [16, 16], omega)
        Helm = Helmholtz(kappa, gamma, omega)

        def M(v):
            e = vcycle_wave_adr(v.view(1,1,m,n), kappa, gamma, omega, 1, max_level-1, T, Tx, Ty, LT, lams, alphas)
            return e.flatten()
        
        tic = time.time()
        haha = fgmres_res(Helm, f.flatten(), rel_tol=1e-6, max_restarts=restart, max_iter=max_iter, flexible=True, precond=M)
        ress_gmres = [1.] + haha.residual_norms
        PGMRES_time = time.time() - tic
        return haha.solution, ress_gmres, PGMRES_time

    def _reset_parameters(self):
        for param in self.meta_alpha.parameters():
            if param.ndim > 1:
                xavier_uniform_(param, gain=self.xavier_init)
            else:
                constant_(param, 1e-2)

        for param in self.neural_operator.parameters():
            if param.ndim > 1:
                xavier_uniform_(param, gain=self.xavier_init)
            else:
                constant_(param, 1e-2)
