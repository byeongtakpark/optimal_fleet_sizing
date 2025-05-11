import numpy as np
import gurobipy as gp
from gurobipy import GRB

def solve_rebalancing_lp(
    T: np.ndarray,
    lambda_vec: np.ndarray,
    P: np.ndarray,
    n_vpt: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the optimal rebalancing problem (Zhang & Pavone, 2016) using Gurobi.

    Parameters:
        T: Travel time matrix (n_vpt x n_vpt)
        lambda_vec: Arrival rate vector (n_vpt,)
        P: Choice probability matrix (n_vpt x n_vpt)
        n_vpt: Number of vertiports

    Returns:
        psi: Total outbound rebalancing flow per node (n_vpt,)
        alpha: Routing proportions matrix (n_vpt x n_vpt)
        beta_mat: Full rebalancing flow matrix (n_vpt x n_vpt)
    """
    m = gp.Model("Rebalancing")
    m.Params.OutputFlag = 0

    beta_vars = {}
    for i in range(n_vpt):
        for j in range(n_vpt):
            if i != j:
                beta_vars[i, j] = m.addVar(lb=0.0, name=f"beta_{i}_{j}")

    # Objective
    m.setObjective(gp.quicksum(T[i, j] * beta_vars[i, j] for (i, j) in beta_vars), GRB.MINIMIZE)

    # Flow conservation
    for i in range(n_vpt):
        outflow = gp.quicksum(beta_vars[i, j] for j in range(n_vpt) if j != i)
        inflow = gp.quicksum(beta_vars[j, i] for j in range(n_vpt) if j != i)
        rhs = -lambda_vec[i] + float(np.dot(P[:, i], lambda_vec))
        m.addConstr(outflow - inflow == rhs, name=f"flow_{i}")

    m.optimize()

    beta_mat = np.zeros((n_vpt, n_vpt))
    for (i, j), var in beta_vars.items():
        beta_mat[i, j] = var.X

    psi = beta_mat.sum(axis=1)
    alpha = np.zeros_like(beta_mat)
    for i in range(n_vpt):
        if psi[i] > 0:
            alpha[i] = beta_mat[i] / psi[i]

    return psi, alpha, beta_mat

def compute_composite_arrival_routing(
    lambda_vec: np.ndarray,
    psi: np.ndarray,
    P: np.ndarray,
    alpha: np.ndarray,
    n_vpt: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute composite arrival rates and mixed routing probabilities.

    Returns:
    - lambda_tilde: composite arrival vector (lambda_vec and psi)
    - P_tilde: mixed routing probability matrix
    """
    lambda_tilde = lambda_vec + psi

    if np.any(lambda_tilde <= 0):
        zero_idx = np.where(lambda_tilde <= 0)[0]
        raise ValueError(f"Composite arrival rate zero or negative at indices {zero_idx}")

    P_tilde = np.zeros((n_vpt, n_vpt))
    for i in range(n_vpt):
        for j in range(n_vpt):
            if i != j:
                P_tilde[i, j] = (lambda_vec[i] * P[i, j] + psi[i] * alpha[i, j]) / lambda_tilde[i]
        P_tilde[i, i] = 0.0

    # print(P_tilde)
    # print(P_tilde.shape)

    return lambda_tilde, P_tilde

