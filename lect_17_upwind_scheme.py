import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from datetime import datetime as dt

# Peclet number = F / D
# Peclet number > 2 causes issues with solutions using central differencing scheme

# size of grid side
N = 51

# domain size
L = 1

# grid spacing
h = np.float64(L / (N-1))

# diffusion
gamma = np.float64(0.1)

# density
rho = np.float64(1)

# velocity
u = np.float64(5/0.2)

F = rho * u
D = gamma / h

Pe = F / D
print(f'Peclet Number: {Pe}')

# iterations
iteration = 0
max_iters = 100000

# temperature array
T = np.zeros(N)
T[0] = 100.
T[-1] = 500.

# iterated temperature array
T_new = T.copy()

#error-related
epsilon = 1e-8
numerical_error = np.inf
errs = []

t0 = dt.now()
t1_prev = t0
while numerical_error > epsilon: # and iteration < max_iters:
    for i in range(1, N-1):
        a_w = D + max(0, F)
        a_e = D + max(0, -F)
        a_p = a_e + a_w # + F - F
        T_new[i]= (a_e * T_new[i+1] + a_w * T_new[i-1]) / a_p
    diff = np.abs(T - T_new)
    numerical_error = diff.sum()
    errs.append(numerical_error)
    iteration += 1
    T = T_new.copy()
    if iteration % 10 == 0:
        t1 = dt.now()
        dt1 = t1-t1_prev
        t1_prev = t1
        dt1_0 = t1-t0
        print(f'Pe: {Pe}. iteration: {iteration}.  error: {numerical_error}. iter_time: {dt1}.  runtime: {dt1_0}')

# plot results
        
# plt.figure(10)
# plt.semilogy(errs, 'ko')

x_dom = np.arange(N) * h
plt.figure(11)
plt.plot(x_dom, T, 'gx--', linewidth=2, markersize = 5)
plt.grid(True, color = 'k')

plt.xlabel("Position", size=20)
plt.ylabel("Tenperature", size=20)
plt.title("T(x)")
plt.show()

apple = 1
