import numpy as np
import os
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from BinaryHeap import BinaryHeap

#this file implements BACKWARDS repeated a star !!
#Authors: Joon Song, Anay Kothana

# Constants and cardinal directions
GRID_SIZE = 101
DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

class RepeatedBackwardsAStar:
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal

        # Agent's knowledge of the grid: -1 = unknown, 0 = unblocked, 1 = blocked
        self.known_grid = np.full((GRID_SIZE, GRID_SIZE), -1)
        self.known_grid[start] = 0  # Agent knows the start cell is unblocked

        # Initialize g-values, f-values, and parent pointers
        self.gvalues = {}
        self.fvalues = {}
        self.parent = {}

        # Initialize open and closed lists
        self.open_list = BinaryHeap()
        self.closed_list = set()

        # Initialize the agent's path
        self.path = []

        # Initialize all g-values and f-values to infinity
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                self.gvalues[(x, y)] = float('inf')
                self.fvalues[(x, y)] = float('inf')

        # Set the g-value for the goal (which is the start in backward search)
        self.gvalues[self.goal] = 0
        self.fvalues[self.goal] = self.heuristic(self.goal)  # Corrected initialization

        # Insert the goal node into the open list with proper tie-breaking
        self.open_list.insert((self.fvalues[self.goal], -self.gvalues[self.goal], self.goal))

    def heuristic(self, cell):
        # Heuristic is the Manhattan distance from the current cell to the start
        x, y = cell
        start_x, start_y = self.start
        return abs(x - start_x) + abs(y - start_y)

    def get_neighbors(self, cell):
        x, y = cell
        neighbors = []
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            # Ensure neighbor is within grid bounds
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                # Treat unknown cells (-1) as unblocked for planning purposes
                if self.known_grid[nx, ny] != 1:
                    neighbors.append((nx, ny))
        return neighbors

    def search(self):
        # Reset open and closed lists for each search
        self.open_list = BinaryHeap()
        self.closed_list = set()

        # Reset g-values, f-values, and parent pointers
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                self.gvalues[(x, y)] = float('inf')
                self.fvalues[(x, y)] = float('inf')
        self.gvalues[self.goal] = 0
        self.fvalues[self.goal] = self.heuristic(self.goal)
        self.open_list.insert((self.fvalues[self.goal], -self.gvalues[self.goal], self.goal))
        self.parent = {}

        while not self.open_list.is_empty():
            f_value, neg_g_value, current = self.open_list.extract()

            if current == self.start:
                return self.reconstruct_path(current)

            self.closed_list.add(current)

            for neighbor in self.get_neighbors(current):
                if neighbor in self.closed_list:
                    continue

                temp_g = self.gvalues[current] + 1  # Cost from current to neighbor

                if temp_g < self.gvalues[neighbor]:
                    self.gvalues[neighbor] = temp_g
                    self.fvalues[neighbor] = temp_g + self.heuristic(neighbor)
                    self.parent[neighbor] = current

                    # Insert with tie-breaking
                    self.open_list.insert((self.fvalues[neighbor], -self.gvalues[neighbor], neighbor))

        # No path found
        return None

    def reconstruct_path(self, current):
        path = []
        while current in self.parent:
            path.append(current)
            current = self.parent[current]
        path.append(current) 
        path.reverse()
        return path

    def observe_adjacent_cells(self, position):
        x, y = position
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                if self.known_grid[nx, ny] == -1:
                    self.known_grid[nx, ny] = self.grid[nx, ny]

    def is_path_blocked(self, path):
        for cell in path:
            x, y = cell
            if self.known_grid[x, y] == 1:
                return True
        return False

    def run(self):
        current_position = self.goal  # Start from the goal for backward search

        while current_position != self.start:
            self.observe_adjacent_cells(current_position)

            # Plan a path from the current position to the goal oop actually the start
            self.path = self.search()

            if not self.path:
                print("Unable to find a path to the start.")
                return False

            for step in self.path[1:]:  
                current_position = step
                print(f"Agent moving to: {current_position}")

                self.observe_adjacent_cells(current_position)

                remaining_path = self.path[self.path.index(step):]
                if self.is_path_blocked(remaining_path):
                    print(f"Path is blocked at {current_position}. Replanning...")
                    break  

                # If reached the goal i mean start
                if current_position == self.start:
                    print("Start reached!")
                    return True
            else:
                continue  

        print("Start reached!")
        return True

def load_gridworld(filename):
    return np.load(filename)

def visualize_grid(grid, known_grid, path, start, goal):
    # Create a visual grid to represent the agent's knowledge
    visual_grid = np.full(grid.shape, -1)  # Unknown cells

    # Update visual grid with known information
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if known_grid[x, y] != -1:
                visual_grid[x, y] = known_grid[x, y]

    # Mark the path on the visual grid
    for x, y in path:
        visual_grid[x, y] = 0.5  # Use a different value to represent the path

    # Define a custom colormap
    cmap = mcolors.ListedColormap(['white', 'gray', 'black', 'green'])
    bounds = [-1.5, -0.5, 0.25, 0.75, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(8, 8))
    plt.imshow(visual_grid, cmap=cmap, norm=norm, origin='upper')

    # Mark the start and goal positions
    plt.plot(start[1], start[0], 'o', color='blue', markersize=8, label='Start')
    plt.plot(goal[1], goal[0], 'X', color='red', markersize=8, label='Goal')

    plt.legend(loc='upper right')
    plt.title("Agent's Known Grid and Path")
    plt.show()

def run_BackwardsAStar(grid_directory, grid_file):
    grid = load_gridworld(os.path.join(grid_directory, grid_file))

    # Random start and goal positions
    start = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
    goal = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))

    # Ensure start and goal are unblocked
    while grid[start] == 1:
        start = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
    while grid[goal] == 1:
        goal = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))

    print(f"Start: {start}")
    print(f"Goal: {goal}")  # In backward search, planning is from goal to start

    # Create an instance of RepeatedBackwardsAStar
    astar = RepeatedBackwardsAStar(grid, start, goal)

    # Run the agent
    success = astar.run()

    if success:
        print("Success! Path found.")
        visualize_grid(grid, astar.known_grid, astar.path, start, goal)
    else:
        print("No path was found to reach the start.")

if __name__ == "__main__":
    grid_directory = '/Users/joonsong/Desktop/Intro-to-AI/Project1/gridworlds'
    grid_file = 'gridworld_0.npy'  # replace with your grid file
    run_BackwardsAStar(grid_directory, grid_file)
