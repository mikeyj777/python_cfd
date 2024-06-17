import numpy as np

spacing = 6
timesteps = 100

grid = np.zeros(spacing)
grid[-1] = 1

scoring = []

for t in range(timesteps):
    for i in range(1, grid.shape[0]-1):
        grid[i] = (grid[i+1] + grid[i-1]) / 2
    print(f'{t} | {grid}')
    scoring.append(grid)
    if len(scoring) > 5:
        scoring = scoring[-5:]
