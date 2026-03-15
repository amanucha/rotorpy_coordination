import os
import sys
import numpy as np
from coordination_with_mpc_automated import run_simulation
from rotorpy.config import *

def main():
    k_values = [1,10] 
    agent_counts = [4]
    trajectory_types = ['tunnel']
    from rotorpy.config import delays, sequential
    base_params = {
        'delays': delays,
        'sequential': sequential,
    }

    print("Starting simulation batch...")
    for i in range(15):
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
