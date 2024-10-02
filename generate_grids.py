import random
import numpy as np
import matplotlib.pyplot as plt
import os

#this file creates 50 101x101 gridworlds
#Authors: Joon Song, Anay Kothana

GRID_SIZE = 101
BLOCK_PROBABILITY = 0.3
DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

# initialize a grid with all cells marked as unvisited (-1)
def create_empty_grid(size):
    return np.full((size, size), -1)  # -1 means unvisited

# check if a position is inside the grid and unvisited (ignores blocked status)
def is_valid(x, y, grid):
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and grid[x, y] == -1

# dfs function to generate the grid
def depth_first_search(grid, start_x, start_y):
    stack = [(start_x, start_y)]  # Initialize stack with starting position

    while stack:
        x, y = stack.pop()

        # mark the current cell as visited (0)
        grid[x, y] = 0

        # shuffle directions for random movement
        random.shuffle(DIRECTIONS)

        neighbors = []
        for direction in DIRECTIONS:
            nx, ny = x + direction[0], y + direction[1]

            # check if the neighboring cell is valid and unvisited
            if is_valid(nx, ny, grid):
                neighbors.append((nx, ny))

        # if we have neighbors, push the current cell back onto the stack
        if neighbors:
            stack.append((x, y))  # push the current cell back for future backtracking

            # Ccoose a random neighbor to visit
            nx, ny = random.choice(neighbors)
            stack.append((nx, ny))  # add the neighbor to the stack for exploration

# function to randomly block cells after DFS exploration
def block_cells(grid):
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if grid[x, y] == 0 and random.random() < BLOCK_PROBABILITY:
                grid[x, y] = 1  # Block the cell with a 30% probability

# generate a gridworld using DFS and block random cells afterward
def generate_gridworld():
    grid = create_empty_grid(GRID_SIZE)
    start_x, start_y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
    
    # run iterative DFS from a random starting point
    depth_first_search(grid, start_x, start_y)

    # randomly block some cells after DFS
    block_cells(grid)
    
    return grid

# save the generated grid as a .npy file
def save_grid(grid, filename):
    np.save(filename, grid)
    print(f"Grid saved to {filename}")

# function to create and save 50 gridworlds
def generate_multiple_gridworlds(save_directory):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    for i in range(50):
        grid = generate_gridworld()
        filename = os.path.join(save_directory, f'gridworld_{i}.npy')
        save_grid(grid, filename)

# run the code to generate and save multiple gridworlds
if __name__ == "__main__":
    # change this to the directory you want
    save_directory = '/Users/joonsong/Desktop/Intro-to-AI/Project1/gridworlds'  

    generate_multiple_gridworlds(save_directory)
