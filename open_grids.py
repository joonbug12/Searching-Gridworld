import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Function to load and visualize a gridworld from a .npy file
def load_and_visualize_gridworld(filename):
    # Load the gridworld
    grid = np.load(filename)

    cmap=mcolors.ListedColormap(['black', 'white', 'gray'])

    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    # Visualize the gridworld using matplotlib
    plt.imshow(grid, cmap=cmap, norm=norm)
    plt.title(f'Gridworld: {filename}, White:Unblocked, Gray: Blocked, Black: Unvisited')
    plt.show()

# Main function to open a specific gridworld
if __name__ == "__main__":
    # Replace 'gridworld_0.npy' with the path to the gridworld file you want to open
    filename = 'zgridworld_0.npy'
    
    load_and_visualize_gridworld(filename)
