'''
Project: Schelling's Segregation Model
Author: Hunter Renard
'''

# Imports
import matplotlib.pyplot as plt
import numpy as np

# Global Variables
EMPTY = 0
RED = 1
BLUE = 2

# Teleport the person at cell (x,y) of this grid to a random empty square.
def teleport(grid, x, y):
    exc, eyc = np.nonzero(grid == EMPTY)
    hurnyak = np.random.choice(range(len(exc)))
    new_x = exc[hurnyak]
    new_y = eyc[hurnyak]
    grid[new_x,new_y] = grid[x,y]
    grid[x,y] = EMPTY
    

# Return the number of cells neighboring (x,y) that are the color specified.
def num_neighbors_of_color(grid,x,y,color, w,h):
    neighbors = 0
    if x < w-1 and grid[x+1,y] == color:                    # Right
        neighbors = neighbors + 1
    if x > 0 and grid[x-1,y] == color:                          # Left
        neighbors = neighbors + 1
    if y < h-1 and grid[x,y+1] == color:                   # Up
        neighbors = neighbors + 1
    if y > 0 and grid[x,y-1] == color:                          # Down
        neighbors = neighbors + 1
    if x < w-1 and y < h-1 and grid[x+1,y+1] == color: # Up-right
        neighbors = neighbors + 1
    if x > 0 and y > 0 and grid[x-1,y-1] == color:              # Lower-left
        neighbors = neighbors + 1
    if x < w-1 and y > 0 and grid[x+1,y-1] == color:        # Lower-right
        neighbors = neighbors + 1
    if x > 0 and y < h-1 and grid[x-1,y+1] == color:       # Upper-left
        neighbors = neighbors + 1
    return neighbors
 

# Return the fraction of person (x,y)'s neighbors that have the same color as
# that person. If there are no neighbors, return .5.
def frac_like_me(grid, x, y, w, h):
    if grid[x,y] == EMPTY:
        return .5
    my_color = grid[x,y]
    other_color = RED if my_color == BLUE else BLUE
    num_like_me = num_neighbors_of_color(grid, x, y, my_color, w, h)
    num_unlike_me = num_neighbors_of_color(grid, x, y, other_color, w, h)
    if num_like_me + num_unlike_me == 0:
        return .5
    return num_like_me / (num_like_me + num_unlike_me)


# Return the average fraction of people's neighbors that have the same color 
# as they do.
def avg_uniformity(grid, w, h):
    total = 0
    for x in range(w):
        for y in range(h):
            total += frac_like_me(grid, x, y, w, h)
    return total / (w*h)


# Return True only if the person at cell (x,y) of this grid has a high enough
# fraction of neighbors with the same color.
def happy(grid, x, y, t, w, h):
    if grid[x,y] == EMPTY:
        return True
    if frac_like_me(grid, x, y, w, h) >= t:
        return True
    else:
        return False


# Given a 2d array, return a list with the x-coordinates (in a list) and the
# y-coordinates (in another list) of the cells that have the color passed.
def points_for_grid(grid, color, w, h):
    xc = []
    yc = []
    for x in range(w):
        for y in range(h):
            if grid[x,y] == color:
                xc.append(x)
                yc.append(y)
    return [xc,yc]


# Simulation parameters.
def runsim(
PROB_RED = .3,
PROB_BLUE = .3,
# The fraction of a person's neighbors that must be the same color as the
# person in order for that person not to want to move away.
THRESHOLD = .3,
WIDTH = 50,
HEIGHT = 50,
NUM_GEN = 20,
animate = False,
plot = False
):

	# Create a random starting configuration with about PROB_RED of the cells
	# being red, PROB_BLUE being blue, and the rest empty.
	config = np.random.choice([RED,BLUE,EMPTY],
    		p=[PROB_RED,PROB_BLUE,1-PROB_RED-PROB_BLUE], size=WIDTH*HEIGHT)
	config.shape = (WIDTH, HEIGHT)

	# This 3d array will use the third coordinate as a generation number, and thus
	# represent the entire lifetime of the simulated model.
	cube = np.empty((WIDTH,HEIGHT,NUM_GEN))
	cube[:,:,0] = config


	# Run the simulation.
	for gen in range(1,NUM_GEN):
    		cube[:,:,gen] = cube[:,:,gen-1]
    		for x in range(WIDTH):
        		for y in range(HEIGHT):
            			if not happy(cube[:,:,gen],x,y, THRESHOLD, WIDTH, HEIGHT):
                			teleport(cube[:,:,gen],x,y)

	if animate:
		# Animate the results.
		for gen in range(0,NUM_GEN):
    			plt.clf()
    			xc, yc = points_for_grid(cube[:,:,gen],RED, WIDTH, HEIGHT)
    			plt.scatter(xc,yc,marker="s",color="red")
    			xc, yc = points_for_grid(cube[:,:,gen],BLUE, WIDTH, HEIGHT)
    			plt.scatter(xc,yc,marker="s",color="blue")
    			plt.title("Generation #" + str(gen))
    			plt.pause(.1)


	uniformities = []
	for gen in range(0,NUM_GEN):
    		uniformities.append(avg_uniformity(cube[:,:,gen], WIDTH, HEIGHT))

	# Plot the average uniformity (non-diversity) over time.
	if plot:
		plt.clf()
		plt.plot(uniformities)
		plt.title("Final avg uniformity: {:.2f}".format(uniformities[-1]))
		plt.xlabel("generation")
		plt.ylabel("Avg uniformity of all populated cells")
		plt.ylim(0, 1)
		plt.show()

	return uniformities[-1]

def parameter_sweep():
	tolerance_levels = np.arange(0, 1.1, .1)
	average_final_uniformity = np.zeros(len(tolerance_levels))

	for i in range(0, len(tolerance_levels)):
		print("tolerance: " + str(round(tolerance_levels[i], 2)))
		count = 0
		for j in range(0,50):
			count += runsim(THRESHOLD = round(tolerance_levels[i], 2))
		average_final_uniformity[i] = count / 50

	plt.clf()
	plt.ylim(0, 1)
	plt.xlabel("Tolerance")
	plt.ylabel("Average Uniformity in Final Generation")
	plt.suptitle("Tolerance VS. Average Uniformity in Final Generation")
	plt.plot(tolerance_levels, average_final_uniformity)
	plt.show()

parameter_sweep()

