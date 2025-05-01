from experiment import Run_experiment
from experiment_gridworld import run_gridworld_experiment
import os

def batch_run():
    # Create output folder
    os.makedirs('results', exist_ok=True)
    
    ### 1. Synthetic MDPs: LinCombLock (Chain MDP)
    print("\n==== Running Synthetic MDP Experiments (LinCombLock) ====\n")

    grid = [
        (10, 5, 10, 50, "standard"),  # 10 states, 5 actions, 10 horizon, d=50
        (10, 5, 10, 50, "group"),     # Same setup, group features
        (20, 10, 15, 100, "standard"), # 20 states, 10 actions, 15 horizon, d=100
        (20, 10, 15, 100, "group"),    # Same setup, group features
    ]

    for num_state, num_action, horizon, dimension, feature_type in grid:
        print(f"Running LinCombLock | feature_type={feature_type} | S={num_state}, A={num_action}, H={horizon}, d={dimension}")
        
        # Update experiment.py manually to use feature_type dynamically if needed
        
        # Here I assume you modify Run_experiment to accept feature_type (I'll show below)
        Run_experiment(
            num_state=num_state, 
            num_action=num_action, 
            horizon=horizon, 
            dimension=dimension, 
            num_epi=500, 
            num_trial=3,
            mis_prob=0,
            feature_type=feature_type
        )

    ### 2. Gridlock MDPs (Gridworld)

    print("\n==== Running Gridlock Experiments ====\n")

    for grid_size in [4, 6, 8]:  # grid sizes
        print(f"Running Gridworld Grid {grid_size}x{grid_size} with standard and group features")
        
        run_gridworld_experiment(grid_size=grid_size, horizon=10, num_epi=500)

if __name__ == "__main__":
    batch_run()
