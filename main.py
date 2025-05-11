import argparse
import numpy as np
import os
import json 

from src.generate_data import *
from src.util import *
from src.data_loader import load_uam_network, load_input_data
from src.rebalancing import solve_rebalancing_lp, compute_composite_arrival_routing
from src.ess import run_exact_solution_search
from src.visualization import plot_bar_final_availability, plot_ess_summary

def main():
    parser = argparse.ArgumentParser(description="Optimal Fleet Sizing")
    parser.add_argument("--step", type=str, default="all",
                        choices=["all", "rebalancing_lp"],
                        help="Which step to run")
    args = parser.parse_args()

    # Generate Basic Data
    generate_distance_matrix("data/vpt_locations.csv", "data/distance_matrix.csv")
    generate_traffic_flow("data/demand.csv", "data/arrival_rate_vector.csv", "data/choice_probability_matrix.csv")
    generate_routing_matrix("data/vpt_locations.csv", "data/choice_probability_matrix.csv", "data/routing_matrix.csv")
    generate_travel_time_matrix("data/distance_matrix.csv", "data/travel_time_matrix.csv")
    generate_relative_throughput_vector("data/routing_matrix.csv", "data/relative_throughput_vector.csv")

    locations, flight_corridors, nodes, node_idx, loc_index, n_total, n_vpt = load_uam_network("data/vpt_locations.csv")
    R, T, D, lambda_vec, pi_vec = load_input_data("data/routing_matrix.csv",
                                                    "data/travel_time_matrix.csv",
                                                    "data/distance_matrix.csv",
                                                    "data/arrival_rate_vector.csv",
                                                    "data/relative_throughput_vector.csv")


    if args.step in ["rebalancing_lp", "all"]:
        print("Optimal Rebalancing Policy")
        P = np.loadtxt("data/choice_probability_matrix.csv", delimiter=",")
        psi, alpha, beta = solve_rebalancing_lp(T, lambda_vec, P, n_vpt)
        lambda_vec, P_tilde = compute_composite_arrival_routing(lambda_vec, psi, P, alpha, n_vpt)
        # print("[Rebalancing] ψ =", psi)
        # print("[Rebalancing] λ~ =", lambda_tilde)
        R = compute_routing_matrix(locations, P_tilde)
        pi_vec = compute_relative_throughput_vector(R)        

    os.makedirs("result/figures", exist_ok=True)

    # Load config
    with open("config.json", "r") as f:
        config = json.load(f)

    c_fare = config["c_fare"]
    c_usage = config["c_usage"]
    c_penalty = config["c_penalty"]
    c_mnt = config["c_mnt"]
    lambda_total = config["lambda_total"]

    lambda_vec = scale_arrival_rate(lambda_vec, lambda_total)
    c_fare_vec, c_usage_vec = compute_cost_terms(D, loc_index, flight_corridors, c_fare, c_usage)
    mu_1 = compute_service_rate(lambda_vec, T, flight_corridors, loc_index, n_total, n_vpt)

    m_star, _, n_hist, obj_history, avg_availability_history, avg_L_history, avg_W_history, A_fianl = run_exact_solution_search(
        node_list=nodes,
        node_idx=node_idx,
        loc_idx=loc_index,
        locs=locations,
        n_nodes=n_total,
        n_vpt=n_vpt,
        lambda_vec=lambda_vec,
        mu_1=mu_1,
        pi_vec=pi_vec,
        fare_vec=c_fare_vec,
        c_usage_vec=c_usage_vec,
        c_mnt=c_mnt,
        c_penalty=c_penalty
    )

    plot_bar_final_availability(
        locations,
        A_fianl,
        save_path="result/figures/fig_bar_final_availability.png"
    )

    plot_ess_summary(
        m_star,
        n_hist,
        obj_history,
        avg_availability_history,
        avg_L_history,
        avg_W_history,
        save_dir="result/figures"
    )

if __name__ == "__main__":
    main()
