import numpy as np
import matplotlib.pyplot as plt

# grid points
N = 11

# domain size
L = 1

# grid spacing
h = np.float64(L / (N-1))

# iterations
iteration = 0
max_iter = 1000

# temperature array
T = np.zeros(N)
T[-1] = 1

# iterated temperature array
T_new = T.copy()

#error-related
epsilon_mag = 10
epsilon = 1e-8
numerical_error = np.inf

while numerical_error > epsilon:
    for i in range(1, N-1):
        T_new[i]= 0.5 * (T[i-1] + T[i+1])
    diff = np.abs(T - T_new)
    numerical_error = diff.sum()
    iteration += 1
    T = T_new.copy()
    print(f'iter: {iteration} | err: {numerical_error} | T: {T}')

# plot results
x_dom = np.arange(N) * h
plt.plot(x_dom, T, 'gx--', linewidth=2)
plt.grid(True, color = 'k')

plt.xlabel("Position", size=20)
plt.ylabel("Tenperature", size=20)
plt.title("T(x)")
plt.show()

apple = 1
