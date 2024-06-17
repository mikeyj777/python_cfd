import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# size of grid side
N = 10
Nx = N
Ny = N

# domain size
L = 1

# grid spacing
hx = np.float64(L / (Nx-1))
hy = np.float64(L / (Ny-1))

# thermal cond
k = 0.1

#x-sect area
A = 0.001

# iterations
iteration = 0
# max_iter = 1000

# temperature array
T = np.zeros((Nx, Ny))
T[0,:] = 1

# iterated temperature array
T_new = T.copy()

#error-related
epsilon = 1e-8
numerical_error = np.inf

while numerical_error > epsilon:
    for xi in range(Nx-1):
        for yi in range(Ny-1):
            a_e = np.float64(k*A / hx)
            a_w = np.float64(k*A / hx)
            a_n = np.float64(k*A / hy)
            a_s = np.float64(k*A / hy)
            # if xi == 0:
            #     # no flux at west-most face/point
            #     a_w = 0
            a_p = a_e + a_w + a_n + a_s
            T_new[xi, yi]= (a_e * T_new[xi+1, yi] + a_w * T_new[xi-1, yi] + a_n * T[xi, yi-1]+ a_s * T[xi, yi+1]) / a_p
    diff = np.abs(T - T_new)
    numerical_error = diff.sum()
    iteration += 1
    T = T_new.copy()
    # print(f'iter: {iteration} | err: {numerical_error} | T: {T}')

# plot results
print(f'iterations: {iteration}')
x_dom = np.arange(Nx) * hx
y_dom = np.arange(Ny) * hy

fig = plt.figure()
ax = fig.gca(projection="3d")
ax.plot3D(x_dom, y_dom, T)

# plt.plot(x_dom, T, 'gx--', linewidth=2)
# plt.grid(True, color = 'k')

# plt.xlabel("Position", size=20)
# plt.ylabel("Tenperature", size=20)
plt.title("T(x)")
plt.show()

apple = 1
