import numpy as np
import os
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from BinaryHeap import BinaryHeap 

# Constants and cardinal directions
GRID_SIZE = 101
DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

#this class is used to run adaptive a star
#authors: Joon Song, Anay Kothana
class AdaptiveAStar:
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal

        # agent knowledge of the grid: -1 = unknown, 0 = unblocked, 1 = blocked
        self.known_grid = np.full((GRID_SIZE, GRID_SIZE), -1)
        self.known_grid[start] = 0  # Agent knows the start cell is unblocked

        # initialize stuff
        self.gvalues = {}
        self.hvalues = {}
        self.fvalues = {}
        self.parent = {}

        # initialize open and closed lists
        self.open_list = BinaryHeap()
        self.closed_list = set()

        # initialize the agent's path
        self.path = []

        # initialize all g-values and f-values to infinity, and h-values using the heuristic (inf is easily identifiable)
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                self.gvalues[(x, y)] = float('inf')
                self.hvalues[(x, y)] = self.heuristic((x, y))
                self.fvalues[(x, y)] = float('inf')

        # Set the g-value for the start node
        self.gvalues[self.start] = 0
        self.fvalues[self.start] = self.hvalues[self.start]

        # Insert the start node into the open list with proper tie-breaking
        self.open_list.insert((self.fvalues[self.start], -self.gvalues[self.start], self.start))

    #this function calculates the heuristic value by using the manhattan distance (absolute differnce of x + absolute difference y)
    #returns the heuristic value
    def heuristic(self, cell):
        x, y = cell
        goal_x, goal_y = self.goal
        return abs(x - goal_x) + abs(y - goal_y)

    #this function gets the neighbors and returns an array
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

    #this is the main search method
    #returns the path if goal is found, none otherwise
    def search(self):
        # Reset open and closed lists for each search
        self.open_list = BinaryHeap()
        self.closed_list = set()

        # Reset g-values, f-values, and parent pointers
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                self.gvalues[(x, y)] = float('inf')
                self.fvalues[(x, y)] = float('inf')
        self.gvalues[self.start] = 0
        self.fvalues[self.start] = self.hvalues[self.start]
        self.open_list.insert((self.fvalues[self.start], -self.gvalues[self.start], self.start))
        self.parent = {}

        while not self.open_list.is_empty():
            f_value, neg_g_value, current = self.open_list.extract()

            if current == self.goal:
                return self.reconstruct_path(current)

            self.closed_list.add(current)

            for neighbor in self.get_neighbors(current):
                if neighbor in self.closed_list:
                    continue

                # Cost to move is 1
                temp_g = self.gvalues[current] + 1

                if temp_g < self.gvalues[neighbor]:
                    self.gvalues[neighbor] = temp_g
                    self.fvalues[neighbor] = temp_g + self.hvalues[neighbor]
                    self.parent[neighbor] = current

                    # Insert with tie-breaking (higher g-values prioritized)
                    self.open_list.insert((self.fvalues[neighbor], -self.gvalues[neighbor], neighbor))

        # No path found
        return None

    #this function flipflops the path so its not reversed. Return the array of the nodes in order
    def reconstruct_path(self, current):
        path = []
        while current in self.parent:
            path.append(current)
            current = self.parent[current]
        path.append(current)  # add the start node
        path.reverse()
        return path

    #this function updates the heuristic value after searching
    def update_heuristics(self):
        g_goal = self.gvalues[self.goal]
        if g_goal == float('inf'):
            return  # Goal was not reached, so heuristic is unchanged

        for node in self.closed_list:
            if self.gvalues[node] < float('inf'):
                self.hvalues[node] = g_goal - self.gvalues[node]

    #this function helps the agent gain information of the gridworld by looking in each direction
    def observe_adjacent_cells(self, position):
        x, y = position
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                if self.known_grid[nx, ny] == -1:
                    # Observe the actual grid
                    self.known_grid[nx, ny] = self.grid[nx, ny]

    #this function checks if the path is blocked
    #returns true if blocked, false otherwise
    def is_path_blocked(self, path):
        for cell in path:
            x, y = cell
            if self.known_grid[x, y] == 1:
                return True
        return False

    #runs the search. 
    #return true if the path is found, false otherwise
    def run(self):
        current_position = self.start

        while current_position != self.goal:
            # observe adjacent cells
            self.observe_adjacent_cells(current_position)

            # plan a path from the current position to the goal
            self.path = self.search()

            if not self.path:
                print("Unable to find a path to the goal.")
                return False

            # update heuristics based on this search
            self.update_heuristics()

            # follow the planned path until a blocked cell is encountered
            for step in self.path[1:]:  # Skip the current position
                current_position = step
                print(f"Agent moving to: {current_position}")

                # observe adjacent cells at the new position
                self.observe_adjacent_cells(current_position)

                # if the current path is now blocked, need to replan
                remaining_path = self.path[self.path.index(step):]
                if self.is_path_blocked(remaining_path):
                    print(f"Path is blocked at {current_position}. Replanning...")
                    break  # Exit the for-loop to replan

                # if reached the goal
                if current_position == self.goal:
                    print("Goal reached!")
                    return True
            else:
                # completed the path without blockage; continue to plan the next path
                continue  # continue to the while-loop to plan the next path

        print("Goal reached!")
        return True

#loads gridworld and returns gridworld
def load_gridworld(filename):
    return np.load(filename)

#
def visualize_grid(grid, known_grid, path, start, goal):
    # create a visual grid to represent the agent's knowledge
    visual_grid = np.full(grid.shape, -1)  # unknown cells

    # update visual grid with known information
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if known_grid[x, y] != -1:
                visual_grid[x, y] = known_grid[x, y]

    # mark the path on the visual grid
    for x, y in path:
        visual_grid[x, y] = 0.5  # use a different value to represent the path

    # Define a custom colormap
    cmap = mcolors.ListedColormap(['white', 'gray', 'black', 'green'])
    bounds = [-1.5, -0.5, 0.25, 0.75, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(8, 8))
    plt.imshow(visual_grid, cmap=cmap, norm=norm, origin='upper')

    # mark the start and goal positions
    plt.plot(start[1], start[0], 'o', color='blue', markersize=8, label='Start')
    plt.plot(goal[1], goal[0], 'X', color='red', markersize=8, label='Goal')

    plt.legend(loc='upper right')
    plt.title("Agent's Known Grid and Path")
    plt.show()

def run_AdaptiveAStar(grid_directory, grid_file):
    grid = load_gridworld(os.path.join(grid_directory, grid_file))

    # random start and goal positions
    start = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
    goal = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))

    # ensure start and goal are unblocked
    while grid[start] == 1:
        start = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
    while grid[goal] == 1:
        goal = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))

    print(f"Start: {start}")
    print(f"Goal: {goal}")

    # create an instance of AdaptiveAStar
    astar = AdaptiveAStar(grid, start, goal)

    # run the agent
    success = astar.run()

    if success:
        print("Success! Path found.")
        visualize_grid(grid, astar.known_grid, astar.path, start, goal)
    else:
        print("No path was found to reach the goal.")

if __name__ == "__main__":
    grid_directory = '/Users/joonsong/Desktop/Intro-to-AI/Project1/gridworlds'
    grid_file = 'gridworld_0.npy'  # replace with your grid file (0,1,2,3.....etc)
    run_AdaptiveAStar(grid_directory, grid_file)
