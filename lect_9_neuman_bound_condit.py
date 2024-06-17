import numpy as np
import matplotlib.pyplot as plt

# bound given dT/dx = 0 at i = 0

# grid points
N = 101

# domain size
L = 1

# grid spacing
h = np.float64(L / (N-1))

# thermal cond
k = 0.1

#x-sect area
A = 0.001

# iterations
iteration = 0
# max_iter = 1000

# temperature array
T = np.zeros(N)
T[-1] = 1

# iterated temperature array
T_new = T.copy()

#error-related
epsilon = 1e-8
numerical_error = np.inf

while numerical_error > epsilon:
    for i in range(N-1):
        a_e = np.float64(k*A / h)
        a_w = np.float64(k*A / h)
        if i == 0:
            # no flux at west-most face/point
            a_w = 0
        a_p = a_e + a_w
        T_new[i]= (a_e * T_new[i+1] + a_w * T[i-1]) / a_p
    diff = np.abs(T - T_new)
    numerical_error = diff.sum()
    iteration += 1
    T = T_new.copy()
    # print(f'iter: {iteration} | err: {numerical_error} | T: {T}')

# plot results
print(f'iterations: {iteration}')
x_dom = np.arange(N) * h
plt.plot(x_dom, T, 'gx--', linewidth=2)
plt.grid(True, color = 'k')

plt.xlabel("Position", size=20)
plt.ylabel("Tenperature", size=20)
plt.title("T(x)")
plt.show()

apple = 1
