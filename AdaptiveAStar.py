import numpy as np
import os
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from BinaryHeap import BinaryHeap

#constants and cardinal directions
GRID_SIZE = 101
DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

class AdaptiveAStar:
    
    #constructor for the class
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.gvalues = {}  # Stores cost from start to a node
        self.hvalues = {}  # Heuristic values (can be adaptive)
        self.fvalues = {}
        self.parent = {}
        self.open_list = BinaryHeap()
        self.closed_list = set()
        self.path = []

        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                self.gvalues[(x, y)] = float('inf')
                self.hvalues[(x, y)] = self.heuristic((x, y))  # Initialize heuristic values
                self.fvalues[(x, y)] = float('inf')

        self.gvalues[self.start] = 0
        self.fvalues[self.start] = self.hvalues[self.start]
        self.open_list.insert((self.fvalues[self.start], self.start))

    #to calculate heuristic value, use the Manhattan distance from the current cell to the goal.
    # This will be updated in future searches in Adaptive A*
    def heuristic(self, cell):
        x, y = cell
        goal_x, goal_y = self.goal
        return abs(x - goal_x) + abs(y - goal_y)

    #gets the neighbors of a cell
    def get_neighbors(self, cell):
        x, y = cell
        neighbors = []
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            #ensures the neighbor is on the grid
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and self.grid[nx, ny] != 1:
                neighbors.append((nx, ny))
        return neighbors

    #main A* search implementation
    def search(self):
        while not self.open_list.is_empty():
            f_value, current = self.open_list.extract()

            if current == self.goal: 
                return self.reconstruct_path(current)

            self.closed_list.add(current)

            for neighbor in self.get_neighbors(current):
                if neighbor in self.closed_list:
                    continue

                temp_g = self.gvalues[current] + 1

                if temp_g < self.gvalues[neighbor]:
                    self.gvalues[neighbor] = temp_g
                    self.fvalues[neighbor] = temp_g + self.hvalues[neighbor]
                    self.parent[neighbor] = current

                    if not self.open_list.contains(neighbor):
                        self.open_list.insert((self.fvalues[neighbor], neighbor))

        return None  # If no path is found

    #rebuilds the path from the start to goal node after a* finds the goal
    def reconstruct_path(self, current):
        path = []
        while current in self.parent:
            path.append(current)
            current = self.parent[current]

        path.append(self.start)
        path.reverse()
        return path

    # Update heuristic values using g-values after a successful search
    def update_heuristics(self):
        g_goal = self.gvalues[self.goal]
        for node in self.closed_list:
            # Update heuristic to improve future searches
            self.hvalues[node] = g_goal - self.gvalues[node]

    def run(self):
        curr_position = self.start

        while curr_position != self.goal:
            self.path = self.search()

            if not self.path:
                print("Unable to find a path to goal")
                return False

            self.update_heuristics()  # Update heuristics based on this search
            
            for step in self.path:
                curr_position = step
                print(f"agent moving to: {curr_position}")

                if self.observe_blockage(curr_position):
                    print(f"There is a blockage. Replanning from {curr_position}")
                    break

        print("goal reached!")
        return True

    def observe_blockage(self, position):
        x, y = position
        blocked = False
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                if self.grid[nx, ny] == 1 and self.gvalues.get((nx, ny)) != float('inf'):
                    blocked = True
                    self.grid[nx, ny] = 1
                    self.gvalues[(nx, ny)] = float('inf')

        return blocked

def load_gridworld(filename): return np.load(filename)

def visualize_grid(grid, path, start, goal):
    visual_grid = grid.copy()

    for x, y in path:
        visual_grid[x, y] = 2

    cmap = mcolors.ListedColormap(['white', 'gray', 'black'])

    plt.plot(start[1], start[0], 'o', color='green', markersize=10, label='Start') 
    plt.plot(goal[1], goal[0], 'X', color='blue', markersize=10, label='Goal') 

    plt.grid(True, which='both', color='black', linestyle='-', linewidth=0.5)
    plt.xticks(np.arange(0, grid.shape[1], 1))
    plt.yticks(np.arange(0, grid.shape[0], 1))

    plt.legend(loc='upper right')

    plt.imshow(visual_grid, cmap=cmap, origin='upper')
    plt.title("Gridworld path")
    plt.colorbar()
    plt.show()

def run_AdaptiveAStar(grid_directory, grid_file):
    grid = load_gridworld(os.path.join(grid_directory, grid_file))

    start = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
    goal = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))

    if grid[start] == 1:
        while grid[start] == 1:
            start = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
    
    if grid[goal] == 1:
        while grid[goal] == 1:
            goal = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))

    print(f"Start: {start}")
    print(f"Goal: {goal}")

    astar = AdaptiveAStar(grid, start, goal)

    success = astar.run()

    if success:
        print("success! path found")
        visualize_grid(grid, astar.path, start, goal)
    else:
        print("no path was found to reach the goal")

if __name__ == "__main__":
    grid_directory = '/Users/joonsong/Desktop/Intro-to-AI/Project1'
    grid_file = 'zgridworld_0.npy'
    run_AdaptiveAStar(grid_directory, grid_file)

