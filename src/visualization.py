import matplotlib.pyplot as plt
import os

def plot_bar_final_availability(locations, availability, save_path):
    plt.figure()
    plt.bar(locations, availability)
    plt.xlabel('Vertiport')
    plt.ylabel('Availability')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()

def plot_ess_summary(m_star, n_history, obj_history, avgA, avgL, avgW, save_dir="result/figures"):
    os.makedirs(save_dir, exist_ok=True)

    # Average vehicle availability
    plt.figure()
    plt.plot(n_history, avgA, marker='o', linewidth=3, markersize=8)
    plt.xlabel('Fleet size (m)', fontsize=16)
    plt.ylabel('Average vehicle availability', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linewidth=1.2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig_avg_veh_availability.png"), dpi=600)
    plt.close()

    # Average queue length
    plt.figure()
    plt.plot(n_history, avgL, marker='o', linewidth=3, markersize=8)
    plt.xlabel('Fleet size (m)', fontsize=16)
    plt.ylabel('Average queue length at vertiports', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linewidth=1.2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig_avg_queue_length.png"), dpi=600)
    plt.close()

    # Average response time
    plt.figure()
    plt.plot(n_history, avgW, marker='o', linewidth=3, markersize=8)
    plt.xlabel('Fleet size (m)', fontsize=16)
    plt.ylabel('Average response time at vertiports (min)', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linewidth=1.2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig_avg_response_time.png"), dpi=600)
    plt.close()

    # Objective function curve
    plt.figure()
    plt.plot(n_history, obj_history, linewidth=4)
    plt.xlabel('Fleet size (m)', fontsize=16)
    plt.ylabel('Objective function obj(m)', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linewidth=1.2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig_obj_function.png"), dpi=600)
    plt.close()