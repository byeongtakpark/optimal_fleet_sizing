import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from typing import List, Tuple, Dict

def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    r = 6371
    return r * c

def compute_distance_matrix(df):
    latitudes = df['Lat'].tolist()
    longitudes = df['Lon'].tolist()

    distance_matrix = []
    for i in range(len(df)):
        row = []
        for j in range(len(df)):
            dist = haversine(longitudes[i], latitudes[i], longitudes[j], latitudes[j])
            row.append(dist)
        distance_matrix.append(row)

    distance_matrix = np.array(distance_matrix)
    return distance_matrix

def compute_traffic_flow(array):
    outflow = array.sum(axis=1)
    inflow = array.sum(axis=0) / 24
    outflow_ratio = array / outflow[:, np.newaxis]
    return inflow, outflow_ratio

def is_irreducible(matrix: np.ndarray) -> bool:
    n = matrix.shape[0]
    reachability = np.copy(matrix)
    for _ in range(2, n+1):
        reachability = np.dot(reachability, matrix)
        reachability[reachability > 0] = 1
    if np.all(reachability > 0):
        print("This Markov chain is irreducible.")
    else:
        print("This Markov chain is not irreducible.")

def compute_routing_matrix(locations, P):
    flight_corridors = [(i, j) for i in locations for j in locations if i != j]
    nodes = locations + flight_corridors
    node_idx = {node: idx for idx, node in enumerate(nodes)}

    R = np.zeros((len(nodes), len(nodes)))
    for i_node in nodes:
        for j_node in nodes:
            i_idx = node_idx[i_node]
            j_idx = node_idx[j_node]
            if isinstance(i_node, str) and isinstance(j_node, tuple):
                entry, exit = j_node
                if i_node == entry:
                    R[i_idx, j_idx] = P[locations.index(entry), locations.index(exit)]
            elif isinstance(i_node, tuple) and isinstance(j_node, str):
                entry, exit = i_node
                if j_node == exit:
                    R[i_idx, j_idx] = 1.0
    return R

def compute_travel_time_matrix(distances,
                            cruise_speed_kmph: float = 241,
                            takeoff_time_min: float = 1,
                            landing_time_min: float = 1,
                            taxi_time_min: float = 2) -> np.ndarray:
    procedure_time = (takeoff_time_min + landing_time_min + 2 * taxi_time_min) / 60
    T = procedure_time + (distances / cruise_speed_kmph)
    return T

def compute_relative_throughput_vector(R) -> np.ndarray:
    evals, evecs = np.linalg.eig(R.T)
    idx = np.argmin(np.abs(evals - 1))
    pi = np.real(evecs[:, idx])
    return pi

def scale_arrival_rate(lambda_vec: np.ndarray, lambda_total: float) -> np.ndarray:
    """
    Scale the arrival rate vector to match the total system demand.
    """
    lambda_props = lambda_vec / np.sum(lambda_vec)
    return lambda_props * lambda_total

def compute_service_rate(
    lambda_vec: np.ndarray,
    T: np.ndarray,
    flight_corridors: List[Tuple[str, str]],
    loc_index: Dict[str, int],
    n_tot: int,
    n_vpt: int
) -> np.ndarray:
    """
    Compute service rate vector for nodes.
    """
    mu1 = np.zeros(n_tot)
    mu1[:n_vpt] = lambda_vec
    for f_idx, (j, k) in enumerate(flight_corridors, start=n_vpt):
        mu1[f_idx] = 1.0 / T[loc_index[j], loc_index[k]]
    return mu1

def compute_cost_terms(
    D: np.ndarray,
    loc_index: Dict[str, int],
    flight_corridors: List[Tuple[str, str]],
    c_fare: float,
    c_usage: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute fare and usage cost vectors.
    """
    c_fare_vec = np.array([
        D[loc_index[i], loc_index[j]] * c_fare for i, j in flight_corridors
    ])
    c_usage_vec = np.full(len(loc_index), c_usage)
    return c_fare_vec, c_usage_vec