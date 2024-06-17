import numpy as np

spacing = 10
timesteps = 100
u = 1


grid = np.zeros(spacing)
grid[-1] = 1

scoring = []

for t in range(timesteps):
    for i in range(1, grid.shape[0]-1):
        grid[i] = (0.5 + (u / 4)) * grid[i+1] + (0.5 - (u / 4)) * grid[i-1]
    print(f'{t} | {grid}')
    scoring.append(grid)
    if len(scoring) > 5:
        scoring = scoring[-5:]
