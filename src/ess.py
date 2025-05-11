import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

def run_exact_solution_search(
    node_list: List,
    node_idx: dict,
    loc_idx: dict,
    locs: List,
    n_nodes: int,
    n_vpt: int,
    lambda_vec: np.ndarray,
    mu_1: np.ndarray,
    pi_vec: np.ndarray,
    fare_vec: np.ndarray,
    c_usage_vec: np.ndarray,
    c_mnt: float,
    c_penalty: float,
    max_m: int = 500
) -> Tuple[int, float, List[int], List[float], List[float], List[float], List[float], np.ndarray]:

    obj_history, n_history = [], []
    avgA_history, avgL_history, avgW_history = [], [], []
    L_prev = np.zeros(n_nodes)
    obj_prev = -np.inf
    m_star = -1
    A_final = None

    for n in range(1, max_m + 1):
        W = np.zeros(n_nodes)
        for node in node_list:
            idx = node_idx[node]
            if isinstance(node, tuple):
                W[idx] = 1.0 / mu_1[idx]
            else:
                W[idx] = (1 + L_prev[idx]) / lambda_vec[loc_idx[node]]

        Xn = n / np.dot(pi_vec, W)
        L = Xn * pi_vec * W

        A = np.array([
            Xn * pi_vec[node_idx[loc]] / lambda_vec[loc_idx[loc]]
            for loc in locs
        ])
        avgA = A.mean()
        avgL = L[:n_vpt].mean()
        avgW = (W[:n_vpt] * 60).mean()

        lost_rate = lambda_vec * (1 - A)
        penalty_term = c_penalty * lost_rate.sum()

        Lambda_v = Xn * pi_vec[:n_vpt]
        obj = (
            fare_vec.dot(L[n_vpt:])
            - c_usage_vec.dot(Lambda_v)
            - c_mnt * n
            - penalty_term
        )

        n_history.append(n)
        obj_history.append(obj)
        avgA_history.append(avgA)
        avgL_history.append(avgL)
        avgW_history.append(avgW)

        L_prev = L
        
        if obj < obj_prev:
            m_star = n - 1
            print(f"Optimal fleet size m* = {m_star}")
            print(f"Obj_val={obj_prev}")
            break
        obj_prev = obj
        A_final = A

    return m_star, obj_prev, n_history, obj_history, avgA_history, avgL_history, avgW_history, A_final

