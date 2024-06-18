import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from datetime import datetime as dt

# size of grid side
N = 21

# domain size
L = 1

# grid spacing
h = np.float64(L / (N-1))

# diffusion
gamma = np.float64(0.1)

# density
rho = np.float64(1)

# velocity
u = np.float64(0.25/0.5)

F = rho * u
D = gamma / h

Pe = F / D
print(f'Peclet Number: {Pe}')

# iterations
iteration = 0
# max_iter = 1000

# temperature array
T = np.zeros((N, N))
T[0, :] = 400
T[:, 0] = 400
T[-1, :] = 400
T[:, -1] = 400

# iterated temperature array
T_new = T.copy()

#error-related
epsilon = 1e-8
numerical_error = np.inf
errs = []

t0 = dt.now()
t1_prev = t0
while numerical_error > epsilon:
    for xi in range(1, N-1):
        for yi in range(1, N-1):
            a_e = np.float64(D - F/2)
            a_w = np.float64(D + F/2)
            a_n = np.float64(D - F/2)
            a_s = np.float64(D + F/2)
            a_p = a_e + a_w + a_n + a_s
            T_new[xi, yi]= (a_e * T_new[xi+1, yi] + a_w * T_new[xi-1, yi] + a_n * T[xi, yi-1]+ a_s * T[xi, yi+1]) / a_p
    diff = np.abs(T - T_new)
    numerical_error = diff.sum()
    errs.append(numerical_error)
    iteration += 1
    T = T_new.copy()
    if iteration % 100 == 0:
        t1 = dt.now()
        dt1 = t1-t1_prev
        t1_prev = t1
        dt1_0 = t1-t0
        print(f'iteration: {iteration}.  error: {numerical_error}. iter_time: {dt1}.  runtime: {dt1_0}')

# plot results
        
# plt.figure(10)
# plt.semilogy(errs, 'ko')

x_dom = np.arange(N) * h
y_dom = L - np.arange(N) * h

[X, Y] = np.meshgrid(x_dom, y_dom)

plt.figure(11)
plt.contourf(X, Y, T, 12)

plt.grid(True, color = 'k')

# legend
plt.colorbar(orientation='vertical')

plt.title("T(x, y)")
plt.show()

apple = 1