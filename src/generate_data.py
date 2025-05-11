import pandas as pd
import numpy as np

from typing import List, Tuple, Dict
from src.util import *

def generate_distance_matrix(input_csv_path: str, output_csv_path: str):
    df = pd.read_csv(input_csv_path)
    distance_matrix = compute_distance_matrix(df)
    np.savetxt(output_csv_path, distance_matrix, delimiter=",", fmt='%.6f')

def generate_traffic_flow(demand_csv_path: str, inflow_path: str, choice_matrix_path: str):
    array = np.loadtxt(demand_csv_path, delimiter=',')
    inflow, outflow_ratio = compute_traffic_flow(array)

    np.savetxt(inflow_path, inflow, delimiter=',', fmt='%.5f')
    np.savetxt(choice_matrix_path, outflow_ratio, delimiter=',', fmt='%.5f')

def generate_routing_matrix(locations_csv_path: str,
                         choice_matrix_csv_path: str,
                         routing_matrix_output_path: str) -> np.ndarray:
    df = pd.read_csv(locations_csv_path)
    locations = df["Location"].to_list()
    P = np.loadtxt(choice_matrix_csv_path, delimiter=',')
    R = compute_routing_matrix(locations, P)

    np.savetxt(routing_matrix_output_path, R, delimiter=',', fmt='%.5f')

    is_irreducible(P)
    
def generate_travel_time_matrix(distance_matrix_path: str,
                                   output_path: str,
                                   cruise_speed_kmph: float = 241,
                                   takeoff_time_min: float = 1,
                                   landing_time_min: float = 1,
                                   taxi_time_min: float = 2) -> np.ndarray:
    distances = np.loadtxt(distance_matrix_path, delimiter=',')
    
    T = compute_travel_time_matrix(distances,
                                   cruise_speed_kmph,
                                   takeoff_time_min,
                                   landing_time_min,
                                   taxi_time_min)
    np.savetxt(output_path, T, delimiter=',', fmt='%.5f')

def generate_relative_throughput_vector(routing_matrix_path: str, output_path: str) -> np.ndarray:
    R = np.loadtxt(routing_matrix_path, delimiter=',')
    pi = compute_relative_throughput_vector(R) 
    
    np.savetxt(output_path, pi, delimiter=',', fmt='%.5f')