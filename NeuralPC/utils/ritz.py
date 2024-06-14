import torch
import torch.nn as nn


class Ritz_estimator(nn.Module):
    """
    Ritz estimator class

    Args:
        lin_opt: linear operator
        lam: current E-value
        psi_all: current E-vector
        N_eig: number of E-vectors
        Rsd_r: target relative residual
        Rsd_a: target absolute residual
        zero_cutoff: if ev-slip is smaller than this, it is considered zero
        n_renorm: number of renormalizations
        n_min: minimum number of iterations
        n_max: maximum number of iterations
        MaxGC: maximum number of gradient calculations
        ProjApsiP: projection of A (?)
        n_count: no iters actually done (?)
        final_grad: final gradient norm
        Kalk_Sim: Are we in Kalk-Simma mode?
        delta_cycle: initial error estimate (KS mode)
        gamma_factor: convergence factor
    """

    def __init__(
        self,
        lin_opt,
        lam,
        psi_all,
        N_eig,
        Rsd_r,
        Rsd_a,
        zero_cutoff,
        n_renorm,
        n_min,
        n_max,
        MaxGC,
        ProjApsiP,
        n_count,
        final_grad,
        Kalk_Sim,
        delta_cycle,
        gamma_factor,
    ):
        super(Ritz_estimator, self).__init__()
        self.lin_opt = lin_opt
        self.lam = lam
        self.psi_all = psi_all
        self.N_eig = N_eig
        self.Rsd_r = Rsd_r
        self.Rsd_a = Rsd_a
        self.zero_cutoff = zero_cutoff
        self.n_renorm = n_renorm
        self.n_min = n_min
        self.n_max = n_max
        self.MaxGC = MaxGC
        self.ProjApsiP = ProjApsiP
        self.n_count = n_count
        self.final_grad = final_grad
        self.Kalk_Sim = Kalk_Sim
        self.delta_cycle = delta_cycle
        self.gamma_factor = gamma_factor
