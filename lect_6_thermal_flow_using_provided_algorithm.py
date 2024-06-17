import numpy as np

spacing = 6
epsilon_mag = 10
epsilon = 10**(-1 * epsilon_mag)
max_iter = 1000

T = np.zeros(spacing)
T[-1] = 1

T_new = T.copy()

err = np.inf

iter = 0

while err > epsilon and iter <= max_iter:
    iter += 1
    for i in range(1, T.shape[0]-1):
        T_new[i] = (T_new[i+1] + T_new[i-1]) / 2
    diff = np.abs(T - T_new)
    err = diff.sum()
    T = T_new.copy()
    print(f'iter: {iter} | err: {err} | T: {T}')

apple = 1