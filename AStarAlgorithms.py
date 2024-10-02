import numpy as np
from BinaryHeap import BinaryHeap 
from HigherG_BinaryHeap import HigherG_BinaryHeap 
from LowerG_BinaryHeap import LowerG_BinaryHeap

#This file just merges all of the AStar implementations into one file to make comparing them easier
#Authors: Joon Song, Anay Kothana

# Constants and cardinal directions
GRID_SIZE = 101
DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

class RepeatedAStar:
    def __init__(self, grid, start, goal):
        self.grid = grid  # The actual grid (unknown to the agent)
        self.start = start
        self.goal = goal
        self.expanded_nodes = 0

        # agent's knowledge of the grid: -1 = unknown, 0 = unblocked, 1 = blocked
        self.known_grid = np.full((GRID_SIZE, GRID_SIZE), -1)
        self.known_grid[start] = 0  # agent knows the start cell is unblocked

        # initialize g-values, f-values, and parent pointers
        self.gvalues = {}
        self.fvalues = {}
        self.parent = {}

        # initialize open and closed lists
        self.open_list = BinaryHeap()
        self.closed_list = set()

        # initialize the agent's path
        self.path = []

    def heuristic(self, cell):
        x, y = cell
        goal_x, goal_y = self.goal
        return abs(x - goal_x) + abs(y - goal_y)

    def get_neighbors(self, cell):
        x, y = cell
        neighbors = []
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                if self.known_grid[nx, ny] != 1:
                    neighbors.append((nx, ny))
        return neighbors

    def search(self, start):
        self.open_list = BinaryHeap()
        self.closed_list = set()

        self.gvalues = {start: 0}
        self.fvalues = {start: self.heuristic(start)}
        self.parent = {}

        # insert the start node into the open list
        self.open_list.insert((self.fvalues[start], -self.gvalues[start], start))

        while not self.open_list.is_empty():
            f_value, neg_g_value, current = self.open_list.extract()

            self.expanded_nodes += 1

            if current == self.goal:
                return self.reconstruct_path(current)

            self.closed_list.add(current)

            for neighbor in self.get_neighbors(current):
                if neighbor in self.closed_list:
                    continue

                temp_g = self.gvalues[current] + 1  # cost from start to neighbor

                if neighbor not in self.gvalues or temp_g < self.gvalues[neighbor]:
                    self.gvalues[neighbor] = temp_g
                    self.fvalues[neighbor] = temp_g + self.heuristic(neighbor)
                    self.parent[neighbor] = current

                    # tie-breaking in favor of larger g-values (smaller -g)
                    self.open_list.insert((self.fvalues[neighbor], -self.gvalues[neighbor], neighbor))

        # No path found
        return None

    def reconstruct_path(self, current):
        path = []
        while current in self.parent:
            path.append(current)
            current = self.parent[current]
        path.append(current)  # Add the start node
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
        current_position = self.start

        while current_position != self.goal:
            self.observe_adjacent_cells(current_position)

            self.path = self.search(current_position)

            if not self.path:
                print("Unable to find a path to the goal.")
                return False

            for step in self.path[1:]:  
                current_position = step
                print(f"Agent moving to: {current_position}")

                self.observe_adjacent_cells(current_position)

                if self.is_path_blocked(self.path[self.path.index(step):]):
                    print(f"Path is blocked at {current_position}. Replanning...")
                    break  
                if current_position == self.goal:
                    print("Goal reached!")
                    return True
            else:
                continue  
        print("Goal reached!")
        return True

class HigherG_RepeatedAStar:
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.expanded_nodes=0

        # agent's knowledge of the grid: -1 = unknown, 0 = unblocked, 1 = blocked
        self.known_grid = np.full((GRID_SIZE, GRID_SIZE), -1)
        self.known_grid[start] = 0  # Agent knows the start cell is unblocked

        # initialize g-values, f-values, and parent pointers
        self.gvalues = {}
        self.fvalues = {}
        self.parent = {}

        # initialize open and closed lists
        self.open_list = HigherG_BinaryHeap()
        self.closed_list = set()

        # initialize the agent's path
        self.path = []

        # initialize all g-values and f-values to infinity
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                self.gvalues[(x, y)] = float('inf')
                self.fvalues[(x, y)] = float('inf')

        # set the g-value for the start node
        self.gvalues[self.start] = 0
        self.fvalues[self.start] = self.heuristic(self.start)

        # insert the start node into the open list with proper tie-breaking
        self.open_list.insert((self.fvalues[self.start], -self.gvalues[self.start], self.start))

    def heuristic(self, cell):
        x, y = cell
        goal_x, goal_y = self.goal
        return abs(x - goal_x) + abs(y - goal_y)

    def get_neighbors(self, cell):
        x, y = cell
        neighbors = []
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                if self.known_grid[nx, ny] != 1:
                    neighbors.append((nx, ny))
        return neighbors

    def search(self):
        self.open_list = HigherG_BinaryHeap()
        self.closed_list = set()

        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                self.gvalues[(x, y)] = float('inf')
                self.fvalues[(x, y)] = float('inf')
        self.gvalues[self.start] = 0
        self.fvalues[self.start] = self.heuristic(self.start)
        self.open_list.insert((self.fvalues[self.start], -self.gvalues[self.start], self.start))
        self.parent = {}

        while not self.open_list.is_empty():
            f_value, neg_g_value, current = self.open_list.extract()

            self.expanded_nodes+=1

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
                    self.fvalues[neighbor] = temp_g + self.heuristic(neighbor)
                    self.parent[neighbor] = current

                    # Insert with tie-breaking (higher g-values prioritized)
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
        current_position = self.start

        while current_position != self.goal:
            self.observe_adjacent_cells(current_position)

            self.path = self.search()

            if not self.path:
                print("Unable to find a path to the goal.")
                return False

            for step in self.path[1:]:  # Skip the current position
                current_position = step
                print(f"Agent moving to: {current_position}")

                self.observe_adjacent_cells(current_position)

                remaining_path = self.path[self.path.index(step):]
                if self.is_path_blocked(remaining_path):
                    print(f"Path is blocked at {current_position}. Replanning...")
                    break  

                if current_position == self.goal:
                    print("Goal reached!")
                    return True
            else:
                continue  

        print("Goal reached!")
        return True
    
class LowerG_RepeatedAStar:
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.expanded_nodes=0

        self.known_grid = np.full((GRID_SIZE, GRID_SIZE), -1)
        self.known_grid[start] = 0  # Agent knows the start cell is unblocked

        self.gvalues = {}
        self.fvalues = {}
        self.parent = {}

        self.open_list = LowerG_BinaryHeap()
        self.closed_list = set()

        self.path = []

        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                self.gvalues[(x, y)] = float('inf')
                self.fvalues[(x, y)] = float('inf')

        self.gvalues[self.start] = 0
        self.fvalues[self.start] = self.heuristic(self.start)

        self.open_list.insert((self.fvalues[self.start], self.gvalues[self.start], self.start))

    def heuristic(self, cell):
        x, y = cell
        goal_x, goal_y = self.goal
        return abs(x - goal_x) + abs(y - goal_y)

    def get_neighbors(self, cell):
        x, y = cell
        neighbors = []
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                if self.known_grid[nx, ny] != 1:
                    neighbors.append((nx, ny))
        return neighbors

    def search(self):
        self.open_list = LowerG_BinaryHeap()
        self.closed_list = set()

        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                self.gvalues[(x, y)] = float('inf')
                self.fvalues[(x, y)] = float('inf')
        self.gvalues[self.start] = 0
        self.fvalues[self.start] = self.heuristic(self.start)
        self.open_list.insert((self.fvalues[self.start], self.gvalues[self.start], self.start))
        self.parent = {}

        while not self.open_list.is_empty():
            f_value, g_value, current = self.open_list.extract()

            self.expanded_nodes += 1

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
                    self.fvalues[neighbor] = temp_g + self.heuristic(neighbor)
                    self.parent[neighbor] = current

                    # Insert with tie-breaking (lower g-values prioritized)
                    self.open_list.insert((self.fvalues[neighbor], self.gvalues[neighbor], neighbor))

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
        current_position = self.start

        while current_position != self.goal:
            self.observe_adjacent_cells(current_position)

            self.path = self.search()

            if not self.path:
                print("Unable to find a path to the goal.")
                return False

            for step in self.path[1:]:  
                current_position = step
                print(f"Agent moving to: {current_position}")

                self.observe_adjacent_cells(current_position)

                remaining_path = self.path[self.path.index(step):]
                if self.is_path_blocked(remaining_path):
                    print(f"Path is blocked at {current_position}. Replanning...")
                    break  

                if current_position == self.goal:
                    print("Goal reached!")
                    return True
            else:
                continue 

        print("Goal reached!")
        return True
    
class RepeatedBackwardsAStar:
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.expanded_nodes=0

        self.known_grid = np.full((GRID_SIZE, GRID_SIZE), -1)
        self.known_grid[start] = 0  # Agent knows the start cell is unblocked

        self.gvalues = {}
        self.fvalues = {}
        self.parent = {}

        self.open_list = BinaryHeap()
        self.closed_list = set()

        self.path = []

        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                self.gvalues[(x, y)] = float('inf')
                self.fvalues[(x, y)] = float('inf')

        self.gvalues[self.goal] = 0
        self.fvalues[self.goal] = self.heuristic(self.goal)  

        self.open_list.insert((self.fvalues[self.goal], -self.gvalues[self.goal], self.goal))

    def heuristic(self, cell):
        x, y = cell
        start_x, start_y = self.start
        return abs(x - start_x) + abs(y - start_y)

    def get_neighbors(self, cell):
        x, y = cell
        neighbors = []
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                if self.known_grid[nx, ny] != 1:
                    neighbors.append((nx, ny))
        return neighbors

    def search(self):
        self.open_list = BinaryHeap()
        self.closed_list = set()

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

            self.expanded_nodes += 1

            if current == self.start:
                return self.reconstruct_path(current)

            self.closed_list.add(current)

            for neighbor in self.get_neighbors(current):
                if neighbor in self.closed_list:
                    continue

                temp_g = self.gvalues[current] + 1  

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
        current_position = self.goal  

        while current_position != self.start:
            self.observe_adjacent_cells(current_position)

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

                if current_position == self.start:
                    print("Start reached!")
                    return True
            else:
                continue  

        print("Start reached!")
        return True
    
class AdaptiveAStar:
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.expanded_nodes=0

        self.known_grid = np.full((GRID_SIZE, GRID_SIZE), -1)
        self.known_grid[start] = 0  

        self.gvalues = {}
        self.hvalues = {}
        self.fvalues = {}
        self.parent = {}

        self.open_list = BinaryHeap()
        self.closed_list = set()

        self.path = []

        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                self.gvalues[(x, y)] = float('inf')
                self.hvalues[(x, y)] = self.heuristic((x, y))
                self.fvalues[(x, y)] = float('inf')

        self.gvalues[self.start] = 0
        self.fvalues[self.start] = self.hvalues[self.start]

        self.open_list.insert((self.fvalues[self.start], -self.gvalues[self.start], self.start))

    def heuristic(self, cell):
        x, y = cell
        goal_x, goal_y = self.goal
        return abs(x - goal_x) + abs(y - goal_y)

    def get_neighbors(self, cell):
        x, y = cell
        neighbors = []
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                if self.known_grid[nx, ny] != 1:
                    neighbors.append((nx, ny))
        return neighbors

    def search(self):
        self.open_list = BinaryHeap()
        self.closed_list = set()

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

            self.expanded_nodes+=1

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

                    self.open_list.insert((self.fvalues[neighbor], -self.gvalues[neighbor], neighbor))

        return None

    def reconstruct_path(self, current):
        path = []
        while current in self.parent:
            path.append(current)
            current = self.parent[current]
        path.append(current) 
        path.reverse()
        return path

    def update_heuristics(self):
        g_goal = self.gvalues[self.goal]
        if g_goal == float('inf'):
            return  

        for node in self.closed_list:
            if self.gvalues[node] < float('inf'):
                self.hvalues[node] = g_goal - self.gvalues[node]

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
        current_position = self.start

        while current_position != self.goal:
            self.observe_adjacent_cells(current_position)

            self.path = self.search()

            if not self.path:
                print("Unable to find a path to the goal.")
                return False

            self.update_heuristics()

            for step in self.path[1:]:  
                current_position = step
                print(f"Agent moving to: {current_position}")

                self.observe_adjacent_cells(current_position)

                remaining_path = self.path[self.path.index(step):]
                if self.is_path_blocked(remaining_path):
                    print(f"Path is blocked at {current_position}. Replanning...")
                    break  

                if current_position == self.goal:
                    print("Goal reached!")
                    return True
            else:
                continue  

        print("Goal reached!")
        return True


