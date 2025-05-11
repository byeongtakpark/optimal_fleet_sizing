import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Union

def load_uam_network(filepath_vpt_loc: str = "data/vpt_locations.csv") -> Tuple[
    List[str],
    List[Tuple[str, str]],
    List[Union[str, Tuple[str, str]]],
    Dict[Union[str, Tuple[str, str]], int],
    Dict[str, int],
    int,
    int
]:
    """
    Load vertiport locations and generate network nodes and index mappings.

    Parameters:
        filepath_vpt_loc: Path to the CSV file containing 'Location' column.

    Returns:
        locations: List of vertiport names.
        flight_corridors: List of (origin, destination) pairs.
        nodes: Combined list of locations and corridors.
        node_idx: Dictionary mapping each node to a unique index.
        loc_index: Dictionary mapping each vertiport to its index.
        n_tot: Total number of nodes (locations + corridors).
        n_vpt: Number of vertiports.
    """
    df = pd.read_csv(filepath_vpt_loc)
    locations = df['Location'].tolist()

    flight_corridors = [(i, j) for i in locations for j in locations if i != j]
    nodes = locations + flight_corridors

    node_idx = {node: idx for idx, node in enumerate(nodes)}
    loc_index = {loc: idx for idx, loc in enumerate(locations)}

    n_tot = len(nodes)
    n_vpt = len(locations)

    return locations, flight_corridors, nodes, node_idx, loc_index, n_tot, n_vpt

def load_input_data(
    filepath_routing: str = "data/routing_matrix.csv",
    filepath_time: str = "data/travel_time_matrix.csv",
    filepath_dist: str = "data/distance_matrix.csv",
    filepath_arrival: str = "data/arrival_rate_vector.csv",
    filepath_relative_throughput: str = "data/relative_throughput_vector.csv"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load key matrices and vectors for the UAM network model.

    Parameters:
        filepath_routing: Path to routing_matrix.csv
        filepath_time: Path to travel_time_matrix.csv
        filepath_dist: Path to distance_matrix.csv
        filepath_arrival: Path to arrival_rate_vector.csv
        filepath_relative_throughput: Path to relative_throughput_vector.csv

    Returns:
        R: Routing matrix (|N| x |N|)
        T: Travel-time matrix (|V| x |V|)
        D: Distance matrix (|V| x |V|)
        lambda_vec: Arrival rate vector (|V|,)
        pi_vec: Relative throughput vector (|N|,)
    """
    R = np.loadtxt(filepath_routing, delimiter=',')
    T = np.loadtxt(filepath_time, delimiter=',')
    D = np.loadtxt(filepath_dist, delimiter=',')
    lambda_vec = np.loadtxt(filepath_arrival, delimiter=',')
    pi_vec = np.loadtxt(filepath_relative_throughput, delimiter=',')

    return R, T, D, lambda_vec, pi_vec