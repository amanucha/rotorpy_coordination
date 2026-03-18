import os
import sys
import numpy as np
from coordination_with_mpc_automated import run_simulation
from rotorpy.config import *

def main():
    k_values = [1,5,10,25] 
    agent_counts = [5,10,15]
    trajectory_types = ['circular']
    
    from rotorpy.config import delays
    base_params = {
        'delays': delays,
    }

    print("Starting simulation batch...")
    for traj in trajectory_types: 
        for n_agents in agent_counts: 
            for k in k_values:
                    print(f"\n--- Running: K={k}, num_agents={n_agents}, traj={traj} ---")
                    params = base_params.copy()
                    params.update({
                    'K': k,
                    'num_agents': n_agents,
                    })
                
                    run_simulation(params)

    print("\nBatch simulation completed.")

if __name__ == "__main__":
    main()
