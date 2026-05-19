import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from functions import Beta, Delta_U_vec
from colinear_lagrangian_points_T1 import L2_SE, L2_SM, mu_SE, mu_SM, L1_SE


x, y, z, mu = sp.symbols('x y z mu')
r1_mod = sp.sqrt((x + mu)**2 + y**2 + z**2)
r1_vec = sp.Matrix([x + mu, y, z]) / r1_mod
r2_mod = sp.sqrt((x + mu - 1)**2 + y**2 + z**2)
r2_vec = sp.Matrix([x + mu - 1, y, z]) / r2_mod

r_mod = sp.sqrt(x**2 + y**2 + z**2)
r_vec = sp.Matrix([x, y, z]) / r_mod

# Gravitational Potential
U = (1 - mu) / r1_mod + mu / r2_mod + 0.5 * (x**2 + y**2)
dU_dx = sp.diff(U, x)
dU_dy = sp.diff(U, y)
dU_dz = sp.diff(U, z)
d2U_dx2 = sp.diff(dU_dx, x)
d2U_dy2 = sp.diff(dU_dy, y)
d2U_dxdy = sp.diff(dU_dx, y)
d2U_dz2 = sp.diff(dU_dz, z)
d2U_dxdz = sp.diff(dU_dz, x)
d2U_dzdy = sp.diff(dU_dz, y)

# Solar Sail acceleration
beta = sp.symbols('beta') 
nx, ny, nz = sp.symbols('n_x n_y n_z')
n = sp.Matrix([nx, ny, nz]) # n must be expressed in x, y, z coord

a_s = beta * (1 - mu) / r_mod**2 * (r1_vec.dot(n))**2 * n
da_s_dx = sp.diff(a_s, x)
da_s_dy = sp.diff(a_s, y)
da_s_dz = sp.diff(a_s, z)

beta_SE = Beta([L2_SE - 0.002, 0, 0], mu_SE)
beta_SM = Beta([L2_SM - 0.002, 0, 0], mu_SM)

A_3D_ver = sp.Matrix([
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
    [d2U_dx2 + da_s_dx[0], d2U_dxdy + da_s_dy[0], d2U_dxdz + da_s_dz[0], 0, 2, 0],
    [d2U_dxdy + da_s_dx[1], d2U_dy2 + da_s_dy[1], d2U_dzdy + da_s_dz[1], -2, 0, 0],
    [d2U_dxdz + da_s_dx[2], d2U_dzdy + da_s_dy[2], d2U_dz2 + da_s_dz[2], 0, 0, 0]])

# eigenvalues of A for each system (classical and artificial)
A_SE_classical = A_3D_ver.subs({beta: 0, x: L2_SE, y: 0, z: 0, mu : mu_SE})
A_SE_classical = np.array(A_SE_classical).astype(np.float64)
eigenvals_SE_classical = np.linalg.eigvals(A_SE_classical)

Delta_U_SE = Delta_U_vec([L2_SE - 0.002, 0, 0], mu_SE)
n_SE_art = Delta_U_SE / np.linalg.norm(Delta_U_SE)
A_SE_artificial = A_3D_ver.subs({beta: beta_SE, nx: n_SE_art[0], ny: n_SE_art[1], nz: n_SE_art[2]})
A_SE_artificial = A_SE_artificial.subs({x: L2_SE - 0.002, y: 0, z: 0, mu: mu_SE}) # optimal orientation along r1
A_SE_artificial = np.array(A_SE_artificial).astype(np.float64)
eigenvals_SE_artificial = np.linalg.eigvals(A_SE_artificial)

A_SM_classical = A_3D_ver.subs({beta: 0, x: L2_SM, y: 0, z: 0, mu : mu_SM})
A_SM_classical = np.array(A_SM_classical).astype(np.float64)
eigenvals_SM_classical = np.linalg.eigvals(A_SM_classical)

Delta_U_SM = Delta_U_vec([L2_SM - 0.002, 0, 0], mu_SM)
n_SM_art = Delta_U_SM / np.linalg.norm(Delta_U_SM)
A_SM_artificial = A_3D_ver.subs({beta: beta_SM, nx: n_SM_art[0], ny: n_SM_art[1], nz: n_SM_art[2]})
A_SM_artificial = A_SM_artificial.subs({x: L2_SM - 0.002, y: 0, z: 0, mu: mu_SM})
A_SM_artificial = np.array(A_SM_artificial).astype(np.float64)
eigenvals_SM_artificial = np.linalg.eigvals(A_SM_artificial)

def format_complex(val, tol=1e-3):
    real = val.real if abs(val.real) > tol else 0.0
    imag = val.imag if abs(val.imag) > tol else 0.0
    if real == 0.0 and imag == 0.0:
        return "0.0"
    if imag == 0.0:
        return f"{real:+.4f}"  # Il '+' mostra sempre il segno
    if real == 0.0:
        return f"{imag:+.4f}j"
    return f"{real:+.4f} {imag:+.4f}j"

print("Eigenvalues at Sun-Earth L2 (classical):")
print([format_complex(val) for val in eigenvals_SE_classical])
print("Eigenvalues at Sun-Earth L2 (artificial):")
print([format_complex(val) for val in eigenvals_SE_artificial])
print("Eigenvalues at Sun-Mars L2 (classical):")
print([format_complex(val) for val in eigenvals_SM_classical])
print("Eigenvalues at Sun-Mars L2 (artificial):")
print([format_complex(val) for val in eigenvals_SM_artificial])

# CHECK that dU(r0) + a_s(r0) ~ 0
dU_r0 = np.array([dU_dx.subs({x: L2_SE - 0.002, y: 0, z: 0, mu: mu_SE}),
                  dU_dy.subs({x: L2_SE - 0.002, y: 0, z: 0, mu: mu_SE}),
                  0])
a_s_r0 = np.array([a_s.subs({beta: beta_SE, nx: 1, ny: 0, nz: 0, x: L2_SE - 0.002, y: 0, z: 0, mu: mu_SE})[0],
                   a_s.subs({beta: beta_SE, nx: 1, ny: 0, nz: 0, x: L2_SE - 0.002, y: 0, z: 0, mu: mu_SE})[1],
                   a_s.subs({beta: beta_SE, nx: 1, ny: 0, nz: 0, x: L2_SE - 0.002, y: 0, z: 0, mu: mu_SE})[2]])
check_sum = dU_r0 + a_s_r0
print(f"Sum of gravitational and solar sail accelerations at the artificial equilibrium point: {check_sum} (should be close to zero)")

# TASK 3.2 VERIFICATION

# TASK 3.2 VERIFICATION WITH PICCIRILLO DATASET
# ref: A. Piccirillo. “A comparison of Solar Sail actuation methods for parking orbits for solar
# sails”. MA thesis. Politecnico di Milano, 2019.
# from table (3.1)
x_v = 0.988503049085796
y_v = 0.000000000000000
z_v = 0.00264908459384856
delta = 0
gamma = 0.51730
# results of Piccirillo from table 3.3
# lambda1_2 = +−1.9988
# lambda3_4 = +−1.8898i
# lambda5_6 = +−1.5628i
# beta = 0.02

n_ver = [np.cos(gamma)*np.cos(delta), np.sin(delta), np.sin(gamma)] # already in x,y,z coordinates
beta_SE = Beta([x_v, y_v, z_v], mu_SE, n_ver)
print(f'Beta at Piccirillo verification point: {beta_SE:.4f} (paper value: 0.02)')


# eigenvalues of A for each system (classical and artificial)
A_ver_classical = A_3D_ver.subs({beta: 0, x: L1_SE, y: 0, z: 0, mu : mu_SE})
A_ver_classical = np.array(A_ver_classical).astype(np.float64)
eigenvals_ver_classical = np.linalg.eigvals(A_ver_classical)

A_ver_artificial = A_3D_ver.subs({beta: beta_SE, nx: n_ver[0], ny: n_ver[1], nz: n_ver[2]})
A_ver_artificial = A_ver_artificial.subs({x: x_v, y: y_v, z: z_v, mu: mu_SE})
A_ver_artificial = np.array(A_ver_artificial).astype(np.float64)
eigenvals_ver_artificial = np.linalg.eigvals(A_ver_artificial)


print("Eigenvalues at Sun-Earth L1 (classical):")
print([format_complex(val) for val in eigenvals_ver_classical])
print("Eigenvalues at Sun-Earth L1 (artificial):")
print([format_complex(val) for val in eigenvals_ver_artificial])





# extra for visualization 2D of stability regions

# A = sp.Matrix([
#     [0, 0, 1, 0],
#     [0, 0, 0, 1],
#     [d2U_dx2 + da_s_dx[0], d2U_dxdy + da_s_dy[0], 0, 2],
#     [d2U_dxdy + da_s_dx[1], d2U_dy2 + da_s_dy[1], -2, 0],])

# x_shift = np.linspace(0, -0.004, 10)
# y_shift = np.linspace(-0.002, 0.002, 10)
# X_shift, Y_shift = np.meshgrid(x_shift, y_shift)
# lambda_vec_y = X_shift.copy()
# for i in range(X_shift.shape[0]):
#     for j in range(X_shift.shape[0]):
#         beta_SE = Beta([L2_SE + X_shift[i, j], Y_shift[i, j], 0], mu_SE)
#         A_SE_artificial = A.subs({beta: beta_SE, nx: r1_vec[0], ny: r1_vec[1], nz: r1_vec[2]})
#         A_SE_artificial = A_SE_artificial.subs({x: L2_SE + X_shift[i, j], y: Y_shift[i, j], z: 0, mu: mu_SE}) # optimal orientation along r1
#         A_SE_artificial = np.array(A_SE_artificial).astype(np.float64)
#         eigenvals_SE_artificial = np.linalg.eigvals(A_SE_artificial)
#         lambda2 = np.real(np.abs(eigenvals_SE_artificial[0]))
#         lambda_vec_y[i, j] = lambda2

# x_shift = np.linspace(0, -0.004, 10)
# z_shift = np.linspace(-0.002, 0.002, 10)
# X_shift, Z_shift = np.meshgrid(x_shift, z_shift)
# lambda_vec_z = X_shift.copy()
# for i in range(X_shift.shape[0]):
#     for j in range(X_shift.shape[0]):
#         beta_SE = Beta([L2_SE + X_shift[i, j],0, Z_shift[i, j]], mu_SE)
#         A_SE_artificial = A.subs({beta: beta_SE, nx: r1_vec[0], ny: r1_vec[1], nz: r1_vec[2]})
#         A_SE_artificial = A_SE_artificial.subs({x: L2_SE + X_shift[i, j], y: 0, z: Z_shift[i, j], mu: mu_SE}) # optimal orientation along r1
#         A_SE_artificial = np.array(A_SE_artificial).astype(np.float64)
#         eigenvals_SE_artificial = np.linalg.eigvals(A_SE_artificial)
#         lambda2 = np.real(np.abs(eigenvals_SE_artificial[0]))
#         lambda_vec_z[i, j] = lambda2


# fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# axes[0].contourf(X_shift, Y_shift, lambda_vec_y, levels=50, cmap='viridis')
# cbar1 = plt.colorbar(axes[0].contourf(X_shift, Y_shift, lambda_vec_y, levels=50, cmap='viridis'), ax=axes[0])
# cbar1.set_label('Max Real Part of Eigenvalues')
# axes[0].set_title('Real positive eigenvalue in Sun-Earth system (X-Y plane)')
# axes[0].set_xlabel('x Shift')
# axes[0].set_ylabel('y Shift')

# axes[1].contourf(X_shift, Z_shift, lambda_vec_z, levels=50, cmap='viridis')
# cbar2 = plt.colorbar(axes[1].contourf(X_shift, Z_shift, lambda_vec_z, levels=50, cmap='viridis'), ax=axes[1])
# cbar2.set_label('Max Real Part of Eigenvalues')
# axes[1].set_title('Real positive eigenvalue in Sun-Earth system (X-Z plane)')
# axes[1].set_xlabel('x Shift')
# axes[1].set_ylabel('z Shift')

# plt.tight_layout()
# plt.show()

