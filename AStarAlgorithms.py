import numpy as np
import os
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from BinaryHeap import BinaryHeap 
from HigherG_BinaryHeap import HigherG_BinaryHeap 
from LowerG_BinaryHeap import LowerG_BinaryHeap


# Constants and cardinal directions
GRID_SIZE = 101
DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

class RepeatedAStar:
    def __init__(self, grid, start, goal):
        self.grid = grid  # The actual grid (unknown to the agent)
        self.start = start
        self.goal = goal
        self.expanded_nodes = 0

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

    def heuristic(self, cell):
        x, y = cell
        goal_x, goal_y = self.goal
        return abs(x - goal_x) + abs(y - goal_y)

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

    def search(self, start):
        # Initialize open and closed lists for each search
        self.open_list = BinaryHeap()
        self.closed_list = set()

        self.gvalues = {start: 0}
        self.fvalues = {start: self.heuristic(start)}
        self.parent = {}

        # Insert the start node into the open list
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

                temp_g = self.gvalues[current] + 1  # Cost from start to neighbor

                if neighbor not in self.gvalues or temp_g < self.gvalues[neighbor]:
                    self.gvalues[neighbor] = temp_g
                    self.fvalues[neighbor] = temp_g + self.heuristic(neighbor)
                    self.parent[neighbor] = current

                    # Tie-breaking in favor of larger g-values (smaller -g)
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
                    # Observe the actual grid
                    self.known_grid[nx, ny] = self.grid[nx, ny]

    def is_path_blocked(self, path):
        # Check if any cell in the path is now known to be blocked
        for cell in path:
            x, y = cell
            if self.known_grid[x, y] == 1:
                return True
        return False

    def run(self):
        current_position = self.start

        while current_position != self.goal:
            # Observe adjacent cells
            self.observe_adjacent_cells(current_position)

            # Plan a path from the current position to the goal
            self.path = self.search(current_position)

            if not self.path:
                print("Unable to find a path to the goal.")
                return False

            # Follow the planned path until a blocked cell is encountered
            for step in self.path[1:]:  # Skip the current position
                # Move to the next step
                current_position = step
                print(f"Agent moving to: {current_position}")

                # Observe adjacent cells at the new position
                self.observe_adjacent_cells(current_position)

                # If the current path is now blocked, need to replan
                if self.is_path_blocked(self.path[self.path.index(step):]):
                    print(f"Path is blocked at {current_position}. Replanning...")
                    break  # Exit the for-loop to replan
                # If reached the goal
                if current_position == self.goal:
                    print("Goal reached!")
                    return True
            else:
                # Completed the path without blockage; goal might not be reached yet
                continue  # Continue to the while-loop to plan the next path
        print("Goal reached!")
        return True

class HigherG_RepeatedAStar:
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.expanded_nodes=0

        # Agent's knowledge of the grid: -1 = unknown, 0 = unblocked, 1 = blocked
        self.known_grid = np.full((GRID_SIZE, GRID_SIZE), -1)
        self.known_grid[start] = 0  # Agent knows the start cell is unblocked

        # Initialize g-values, f-values, and parent pointers
        self.gvalues = {}
        self.fvalues = {}
        self.parent = {}

        # Initialize open and closed lists
        self.open_list = HigherG_BinaryHeap()
        self.closed_list = set()

        # Initialize the agent's path
        self.path = []

        # Initialize all g-values and f-values to infinity
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                self.gvalues[(x, y)] = float('inf')
                self.fvalues[(x, y)] = float('inf')

        # Set the g-value for the start node
        self.gvalues[self.start] = 0
        self.fvalues[self.start] = self.heuristic(self.start)

        # Insert the start node into the open list with proper tie-breaking
        self.open_list.insert((self.fvalues[self.start], -self.gvalues[self.start], self.start))

    def heuristic(self, cell):
        # Heuristic is the Manhattan distance from the current cell to the goal
        x, y = cell
        goal_x, goal_y = self.goal
        return abs(x - goal_x) + abs(y - goal_y)

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
        self.open_list = HigherG_BinaryHeap()
        self.closed_list = set()

        # Reset g-values, f-values, and parent pointers
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
        path.append(current)  # Append the start node
        path.reverse()
        return path

    def observe_adjacent_cells(self, position):
        x, y = position
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                if self.known_grid[nx, ny] == -1:
                    # Observe the actual grid
                    self.known_grid[nx, ny] = self.grid[nx, ny]

    def is_path_blocked(self, path):
        # Check if any cell in the path is now known to be blocked
        for cell in path:
            x, y = cell
            if self.known_grid[x, y] == 1:
                return True
        return False

    def run(self):
        current_position = self.start

        while current_position != self.goal:
            # Observe adjacent cells
            self.observe_adjacent_cells(current_position)

            # Plan a path from the current position to the goal
            self.path = self.search()

            if not self.path:
                print("Unable to find a path to the goal.")
                return False

            # Follow the planned path until a blocked cell is encountered
            for step in self.path[1:]:  # Skip the current position
                current_position = step
                print(f"Agent moving to: {current_position}")

                # Observe adjacent cells at the new position
                self.observe_adjacent_cells(current_position)

                # If the current path is now blocked, need to replan
                remaining_path = self.path[self.path.index(step):]
                if self.is_path_blocked(remaining_path):
                    print(f"Path is blocked at {current_position}. Replanning...")
                    break  # Exit the for-loop to replan

                # If reached the goal
                if current_position == self.goal:
                    print("Goal reached!")
                    return True
            else:
                # Completed the path without blockage; continue to plan the next path
                continue  # Continue to the while-loop to plan the next path

        print("Goal reached!")
        return True
    
class LowerG_RepeatedAStar:
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.expanded_nodes=0

        # Agent's knowledge of the grid: -1 = unknown, 0 = unblocked, 1 = blocked
        self.known_grid = np.full((GRID_SIZE, GRID_SIZE), -1)
        self.known_grid[start] = 0  # Agent knows the start cell is unblocked

        # Initialize g-values, f-values, and parent pointers
        self.gvalues = {}
        self.fvalues = {}
        self.parent = {}

        # Initialize open and closed lists
        self.open_list = LowerG_BinaryHeap()
        self.closed_list = set()

        # Initialize the agent's path
        self.path = []

        # Initialize all g-values and f-values to infinity
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                self.gvalues[(x, y)] = float('inf')
                self.fvalues[(x, y)] = float('inf')

        # Set the g-value for the start node
        self.gvalues[self.start] = 0
        self.fvalues[self.start] = self.heuristic(self.start)

        # Insert the start node into the open list with proper tie-breaking
        self.open_list.insert((self.fvalues[self.start], self.gvalues[self.start], self.start))

    def heuristic(self, cell):
        # Heuristic is the Manhattan distance from the current cell to the goal
        x, y = cell
        goal_x, goal_y = self.goal
        return abs(x - goal_x) + abs(y - goal_y)

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
        self.open_list = LowerG_BinaryHeap()
        self.closed_list = set()

        # Reset g-values, f-values, and parent pointers
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
        path.append(current)  # Append the start node
        path.reverse()
        return path

    def observe_adjacent_cells(self, position):
        x, y = position
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                if self.known_grid[nx, ny] == -1:
                    # Observe the actual grid
                    self.known_grid[nx, ny] = self.grid[nx, ny]

    def is_path_blocked(self, path):
        # Check if any cell in the path is now known to be blocked
        for cell in path:
            x, y = cell
            if self.known_grid[x, y] == 1:
                return True
        return False

    def run(self):
        current_position = self.start

        while current_position != self.goal:
            # Observe adjacent cells
            self.observe_adjacent_cells(current_position)

            # Plan a path from the current position to the goal
            self.path = self.search()

            if not self.path:
                print("Unable to find a path to the goal.")
                return False

            # Follow the planned path until a blocked cell is encountered
            for step in self.path[1:]:  # Skip the current position
                current_position = step
                print(f"Agent moving to: {current_position}")

                # Observe adjacent cells at the new position
                self.observe_adjacent_cells(current_position)

                # If the current path is now blocked, need to replan
                remaining_path = self.path[self.path.index(step):]
                if self.is_path_blocked(remaining_path):
                    print(f"Path is blocked at {current_position}. Replanning...")
                    break  # Exit the for-loop to replan

                # If reached the goal
                if current_position == self.goal:
                    print("Goal reached!")
                    return True
            else:
                # Completed the path without blockage; continue to plan the next path
                continue  # Continue to the while-loop to plan the next path

        print("Goal reached!")
        return True
    
class RepeatedBackwardsAStar:
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.expanded_nodes=0

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

            self.expanded_nodes += 1

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
        path.append(current)  # Append the goal node
        path.reverse()
        return path

    def observe_adjacent_cells(self, position):
        x, y = position
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                if self.known_grid[nx, ny] == -1:
                    # Observe the actual grid
                    self.known_grid[nx, ny] = self.grid[nx, ny]

    def is_path_blocked(self, path):
        # Check if any cell in the path is now known to be blocked
        for cell in path:
            x, y = cell
            if self.known_grid[x, y] == 1:
                return True
        return False

    def run(self):
        current_position = self.goal  # Start from the goal for backward search

        while current_position != self.start:
            # Observe adjacent cells
            self.observe_adjacent_cells(current_position)

            # Plan a path from the current position to the start
            self.path = self.search()

            if not self.path:
                print("Unable to find a path to the start.")
                return False

            # Follow the planned path until a blocked cell is encountered
            for step in self.path[1:]:  # Skip the current position
                current_position = step
                print(f"Agent moving to: {current_position}")

                # Observe adjacent cells at the new position
                self.observe_adjacent_cells(current_position)

                # If the current path is now blocked, need to replan
                remaining_path = self.path[self.path.index(step):]
                if self.is_path_blocked(remaining_path):
                    print(f"Path is blocked at {current_position}. Replanning...")
                    break  # Exit the for-loop to replan

                # If reached the start
                if current_position == self.start:
                    print("Start reached!")
                    return True
            else:
                # Completed the path without blockage; continue to plan the next path
                continue  # Continue to the while-loop to plan the next path

        print("Start reached!")
        return True
    
class AdaptiveAStar:
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.expanded_nodes=0

        # Agent's knowledge of the grid: -1 = unknown, 0 = unblocked, 1 = blocked
        self.known_grid = np.full((GRID_SIZE, GRID_SIZE), -1)
        self.known_grid[start] = 0  # Agent knows the start cell is unblocked

        # Initialize g-values, h-values, f-values, and parent pointers
        self.gvalues = {}
        self.hvalues = {}
        self.fvalues = {}
        self.parent = {}

        # Initialize open and closed lists
        self.open_list = BinaryHeap()
        self.closed_list = set()

        # Initialize the agent's path
        self.path = []

        # Initialize all g-values and f-values to infinity, and h-values using the heuristic
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

    def heuristic(self, cell):
        # Heuristic is the Manhattan distance from the current cell to the goal
        x, y = cell
        goal_x, goal_y = self.goal
        return abs(x - goal_x) + abs(y - goal_y)

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

    def reconstruct_path(self, current):
        path = []
        while current in self.parent:
            path.append(current)
            current = self.parent[current]
        path.append(current)  # Append the start node
        path.reverse()
        return path

    def update_heuristics(self):
        g_goal = self.gvalues[self.goal]
        if g_goal == float('inf'):
            return  # Goal was not reached in the last search; heuristics remain unchanged

        for node in self.closed_list:
            if self.gvalues[node] < float('inf'):
                self.hvalues[node] = g_goal - self.gvalues[node]

    def observe_adjacent_cells(self, position):
        x, y = position
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                if self.known_grid[nx, ny] == -1:
                    # Observe the actual grid
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
            # Observe adjacent cells
            self.observe_adjacent_cells(current_position)

            # Plan a path from the current position to the goal
            self.path = self.search()

            if not self.path:
                print("Unable to find a path to the goal.")
                return False

            # Update heuristics based on this search
            self.update_heuristics()

            # Follow the planned path until a blocked cell is encountered
            for step in self.path[1:]:  # Skip the current position
                current_position = step
                print(f"Agent moving to: {current_position}")

                # Observe adjacent cells at the new position
                self.observe_adjacent_cells(current_position)

                # If the current path is now blocked, need to replan
                remaining_path = self.path[self.path.index(step):]
                if self.is_path_blocked(remaining_path):
                    print(f"Path is blocked at {current_position}. Replanning...")
                    break  # Exit the for-loop to replan

                # If reached the goal
                if current_position == self.goal:
                    print("Goal reached!")
                    return True
            else:
                # Completed the path without blockage; continue to plan the next path
                continue  # Continue to the while-loop to plan the next path

        print("Goal reached!")
        return True


