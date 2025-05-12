# Optimal fleet sizing
This repository contains code and experiments for paper  

Optimal Fleet Sizing for On-demand Urban Air Mobility using Queueing-Theoretical Approach  
[Byeong Tak Park](https://sites.google.com/view/btpark)  
Submiteed to [IEEE ITSC 2025](https://ieee-itsc.org/2025/)  

---

## Installation

### 1. Clone the Repository
Clone the repository and install the required packages. 

```
# Clone the repository
git clone https://github.com/byeongtakpark/optimal_fleet_sizing.git

# Install required packages
pip install -r requirements.txt
```

### 2. Define parameters
Edit the `config.json` file:

```json
{
    "c_fare": 2000,
    "c_usage": 5000,
    "c_penalty": 0,
    "c_mnt": 2500,
    "lambda_total": 200    
}
```

### 3. Run main.py 
Run the `main.py` to analyze the optimal fleet sizing

```
python main.py --step all
```

To analyze the optimal fleet sizing including the vehicle rebalancing policy schemes, run the `main.py`

```
python main.py --step rebalancing
```

---

## Directory Structure

```
├── main.py     # Entry point script
├── config.json # Experimental parameters
├── data/       # Input files (locations, demand, etc.)
├── result/
│ ├── figures/  # Saved plots
├── src/
│ ├── generate_data.py 
│ ├── data_loader.py    
│ ├── calculations.py   
│ ├── rebalancing.py 
│ ├── ess.py # ESS algorithm
│ └── visualization.py 
```
