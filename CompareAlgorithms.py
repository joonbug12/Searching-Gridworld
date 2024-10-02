import os
import numpy as np
import random
import time
import csv
from AStarAlgorithms import (
    AdaptiveAStar,
    HigherG_RepeatedAStar,
    LowerG_RepeatedAStar,
    RepeatedBackwardsAStar,
    RepeatedAStar
)

#This file imports the A star algorithms from AStarAlgorithms.py and runs tests on the gridworlds to see
#the different runtimes and expanded nodes per search. Saves the results onto a csv file to be analyzed
#Authors: Joon Song, Anay Kothana

# Constants
GRID_SIZE = 101
NUM_GRIDWORLDS = 50
GRIDWORLD_DIR = '/Users/joonsong/Desktop/Intro-to-AI/Project1/gridworlds'  
OUTPUT_FILE = 'a_star_comparison_results.csv'

# Define the list of algorithm classes to compare
ALGORITHMS = {
    'AdaptiveAStar': AdaptiveAStar,
    'HigherG_RepeatedAStar': HigherG_RepeatedAStar,
    'LowerG_RepeatedAStar': LowerG_RepeatedAStar,
    'RepeatedBackwardsAStar': RepeatedBackwardsAStar,
    'RepeatedForwardAStar': RepeatedAStar
}

def select_start_goal(grid):
    """Select random start and goal positions ensuring they are unblocked."""
    unblocked_positions = list(zip(*np.where(grid == 0)))
    if len(unblocked_positions) < 2:
        raise ValueError("Not enough unblocked cells to select start and goal.")
    
    start, goal = random.sample(unblocked_positions, 2)
    return tuple(start), tuple(goal)

def run_algorithm(algorithm_class, grid, start, goal):
    """Instantiate and run the algorithm, returning runtime and expanded nodes."""
    astar_instance = algorithm_class(grid.copy(), start, goal)  # Use a copy to prevent side-effects
    
    start_time = time.perf_counter()
    success = astar_instance.run()
    end_time = time.perf_counter()
    
    elapsed_time = end_time - start_time
    expanded_nodes = astar_instance.expanded_nodes
    
    return elapsed_time, expanded_nodes, success

def main():
    # Prepare the output CSV
    with open(OUTPUT_FILE, mode='w', newline='') as csv_file:
        fieldnames = ['Gridworld_ID', 'Algorithm', 'Runtime_Seconds', 'Expanded_Nodes', 'Success']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        
        # Iterate through each gridworld
        for i in range(NUM_GRIDWORLDS):
            grid_file = os.path.join(GRIDWORLD_DIR, f'gridworld_{i}.npy')
            if not os.path.exists(grid_file):
                print(f"Gridworld file {grid_file} does not exist. Skipping.")
                continue
            
            grid = np.load(grid_file)
            
            try:
                start, goal = select_start_goal(grid)
            except ValueError as ve:
                print(f"Gridworld {i} skipped: {ve}")
                continue
            
            print(f"\nGridworld {i}: Start={start}, Goal={goal}")
            
            # Iterate through each algorithm
            for algo_name, algo_class in ALGORITHMS.items():
                print(f"  Running {algo_name}...")
                
                try:
                    runtime, expanded, success = run_algorithm(algo_class, grid, start, goal)
                except Exception as e:
                    print(f"    Error running {algo_name}: {e}")
                    runtime, expanded, success = None, None, False
                
                # Write the results to CSV
                writer.writerow({
                    'Gridworld_ID': i,
                    'Algorithm': algo_name,
                    'Runtime_Seconds': runtime if runtime is not None else 'Error',
                    'Expanded_Nodes': expanded if expanded is not None else 'Error',
                    'Success': success
                })
                
                print(f"    Runtime: {runtime:.4f} seconds, Expanded Nodes: {expanded}, Success: {success}")
    
    print(f"\nComparison complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
