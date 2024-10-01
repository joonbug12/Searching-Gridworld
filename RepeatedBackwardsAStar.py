import numpy as np
import os
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from BinaryHeap import BinaryHeap

#constants and cardinal directions
GRID_SIZE = 101
DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

class RepeatedBackwardsAStar:
    
    #constructor for the class
    def __init__(self, grid, start, goal):
        self.grid=grid
        self.start=start
        self.goal=goal
        self.gvalues = {}
        self.fvalues = {}
        self.parent = {}
        self.open_list = BinaryHeap()
        self.closed_list= set()
        self.path = []

        for x in range (GRID_SIZE):
            for y in range (GRID_SIZE):
                self.gvalues[(x,y)] = float('inf')
                self.fvalues[(x,y)] = float('inf')

        #change from self.start to self.goal
        self.gvalues[self.goal] = 0
        self.fvalues[self.goal] = self.heuristic(self.start)
        self.open_list.insert((self.fvalues[self.goal], self.goal))     #changed from start to goal

    #to calculate heuristic value, use the manhattan distance from the current cell to the goal.
    # Calculate by finding the absolute differnce in x and y and adding them together.
    # This is going to be reversed now since its backwards :D
    def heuristic(self, cell):
        x, y = cell
        start_x, start_y = self.goal
        return abs(x-start_x) + abs(y-start_y)

    #gets the neighbors of a cell
    def get_neighbors(self, cell):
        x, y = cell
        neighbors=[]
        for dx, dy in DIRECTIONS:
            nx, ny = x+dx, y+dy
            #ensures the neighbor is on the grid
            if 0<=nx<GRID_SIZE and 0<=ny<GRID_SIZE and self.grid[nx,ny] !=1:
                neighbors.append((nx, ny))
        return neighbors
        
    #main A* search implementation
    def search(self):
        
        #main loop for the search, and keeps looping until the goal (actually the start now) is found, or the list becomes empty aka there is no path
        while not self.open_list.is_empty():
            f_value, current = self.open_list.extract()

            #changed it to self.start :))))
            if current == self.start: return self.reconstruct_path(current)

            self.closed_list.add(current)

            #loop to check the neighbors
            for neighbor in self.get_neighbors(current):
                if neighbor in self.closed_list:
                    continue

                #cost to move is 1
                temp_g = self.gvalues[current] + 1

                #checking to see if the new g value is better than the g value of the neighbor.
                # If it is, update the g, f, and parent accordingly
                if temp_g < self.gvalues[neighbor]:
                    self.gvalues[neighbor] = temp_g
                    self.fvalues[neighbor] = temp_g + self.heuristic(neighbor)
                    self.parent[neighbor] = current

                    #if the neighbor isnt in the open list, add it.
                    if not self.open_list.contains(neighbor):
                        self.open_list.insert((self.fvalues[neighbor], neighbor))

        #if there is no path
        return None
        
    #rebuilds the path from the start to goal node after a* finds the goal, actually the start
    def reconstruct_path(self, current):
        path=[]
        #this loop is backtracking until it reaches the start
        while current in self.parent:
            path.append(current)
            current = self.parent[current]

        #add the start node and then reverse the array to get the path start to finish instead of having it backwards
        path.append(self.goal)  #start to goal woah
        path.reverse()
        return path
        

    def run(self):
        curr_position = self.goal   #starting from goal now hehehheh

        while curr_position != self.start:      #you see how i changed from goal to start
            self.path = self.search()

            if not self.path:
                print("Unable to find a path to start")
                return False
                
            for step in self.path:
                curr_position = step
                print(f"agent moving to: {curr_position}")

                if self.observe_blockage(curr_position):
                    print(f"There is a blockage. Replanning from {curr_position}")
                    break

        print("start reached!")
        return True
        

    def observe_blockage(self,position):
        x, y = position
        blocked = False
        for dx, dy in DIRECTIONS:
            nx, ny = x+dx, y+dy
                
            if 0<=nx<GRID_SIZE and 0<=ny<GRID_SIZE:
                if self.grid[nx,ny] == 1 and self.gvalues.get((nx, ny)) != float('inf'):
                    blocked = True
                    self.grid[nx, ny] = 1
                    self.gvalues[(nx, ny)] = float('inf')

        return blocked


def load_gridworld(filename): return np.load(filename)

def visualize_grid(grid, path, start, goal):
    visual_grid = grid.copy()

    for x,y in path:
        visual_grid[x,y] = 2

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

def run_BackwardsAStar(grid_directory, grid_file):
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
    print(f"Goal: {goal}")      #i almost swapped them but then i realized that would defeat the purpose

    #filling in the values for the RepeatedBackwardsAStar instance
    astar = RepeatedBackwardsAStar(grid, start, goal)

    #running the A star algorithm
    success = astar.run()

    if success: 
        print("success! path found")
        visualize_grid(grid, astar.path, start, goal)
    else:
        print("no path was found to reach the start")

if __name__ == "__main__":
    grid_directory = '/Users/joonsong/Desktop/Intro-to-AI/Project1'
    grid_file = 'zgridworld_0.npy'
    run_BackwardsAStar(grid_directory, grid_file)



