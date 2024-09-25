import random
import numpy as np
import matplotlib.pyplot as plt
import os

GRID_SIZE = 101
BLOCK_PROBABILITY = 0.3
DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

# Initialize a grid with all cells marked as unvisited (-1)
def create_empty_grid(size):
    return np.full((size, size), -1)  # -1 means unvisited

# Check if a position is inside the grid and unvisited
def is_valid(x, y, grid):
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and grid[x, y] == -1

# Depth-First Search (DFS) function to generate the grid
def depth_first_search(grid, start_x, start_y):
    stack = [(start_x, start_y)]  # Initialize stack with starting position

    while stack:
        x, y = stack.pop()

        # Mark the current cell as unblocked (0)
        grid[x, y] = 0

        # Shuffle directions for random movement
        random.shuffle(DIRECTIONS)

        neighbors = []
        for direction in DIRECTIONS:
            nx, ny = x + direction[0], y + direction[1]

            # Check if the neighboring cell is valid and unvisited
            if is_valid(nx, ny, grid):
                neighbors.append((nx, ny))

        # If we have neighbors, push the current cell back onto the stack
        if neighbors:
            stack.append((x, y))  # Push the current cell back for future backtracking

            # Choose a random neighbor to visit
            nx, ny = random.choice(neighbors)

            # 30% chance to block the cell
            if random.random() < BLOCK_PROBABILITY:
                grid[nx, ny] = 1  # Blocked cell (1)
            else:
                stack.append((nx, ny))  # Add the neighbor to the stack for exploration


# Generate a gridworld using DFS
def generate_gridworld():
    grid = create_empty_grid(GRID_SIZE)
    start_x, start_y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
    
    # Run iterative DFS from a random starting point
    depth_first_search(grid, start_x, start_y)
    
    return grid

# Save the generated grid as a .npy file
def save_grid(grid, filename):
    np.save(filename, grid)
    print(f"Grid saved to {filename}")

# Main function to create and save 50 gridworlds
def generate_multiple_gridworlds(save_directory):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    for i in range(50):
        grid = generate_gridworld()
        filename = os.path.join(save_directory, f'gridworld_{i}.npy')
        save_grid(grid, filename)

# Run the code to generate and save multiple gridworlds
if __name__ == "__main__":
    # Specify the directory where you want to save the .npy files
    save_directory = '/Users/joonsong/Desktop/Intro-to-AI/Project1'  # You can customize this path

    generate_multiple_gridworlds(save_directory)
