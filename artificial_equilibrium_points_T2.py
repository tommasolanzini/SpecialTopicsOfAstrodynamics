import numpy as np
import matplotlib.pyplot as plt
import functions as f

from functions import Delta_U_vec
from colinear_lagrangian_points_T1 import L2_SE, L2_SM, mu_SE, mu_SM

# first definition of function "Beta"
def Beta(X, mu, n = None, n_rf_r1 = None):
    """ Lightness parameter for the solar sail"""
    r1 = ((X[0] + mu)**2 + X[1]**2 + X[2]**2)**0.5
    R1 = np.array([X[0] + mu, X[1], X[2]]) / r1
    Delta_U = Delta_U_vec(X, mu)
    if n is None and n_rf_r1 is None:
        n = Delta_U / np.linalg.norm(Delta_U)
    elif n_rf_r1 is not None:
        # Rotational matrix R (DCM)
        gamma = np.arctan2(X[1], X[0] + mu) # angle between the position vector and the x-axis
        i = np.arctan2(X[2], np.sqrt((X[0] + mu)**2 + X[1]**2)) # angle between the position vector and the xy-plane
        R = np.array([[np.cos(i)*np.cos(gamma), np.sin(gamma)*np.cos(i), np.sin(i)],
                  [-np.sin(gamma), np.cos(gamma), 0],
                  [-np.cos(gamma)*np.sin(i), -np.sin(gamma)*np.sin(i), np.cos(i)]])
        n = R @ n_rf_r1
    Beta = r1**2 / (1 - mu) * Delta_U @ n / (R1 @ n)**2
    return Beta


L_shift = 0.00200
X_SE = np.array ([L2_SE - L_shift, 0, 0])
print(f'Beta at S-E L2: {Beta(X_SE, mu_SE)}')   

X_SM = np.array ([L2_SM - L_shift, 0, 0])
print(f'Beta at S-M L2: {Beta(X_SM, mu_SM)}')

# orientation (task 2.1b)

Delta_U_SE = Delta_U_vec(X_SE, mu_SE)
n_SE = Delta_U_SE / np.linalg.norm(Delta_U_SE)
print(f'Optimal orientation SE (task 2.1b): {n_SE}')

Delta_U_SM = Delta_U_vec(X_SM, mu_SM)
n_SM = Delta_U_SM / np.linalg.norm(Delta_U_SM)
print(f'Optimal orientation SM (task 2.1b): {n_SM}')

# ACCELERATIONS VISUAL REPRESENTATION 
def acceleration_vector(X, mu):
    X = np.array(list(X) + [0,0,0] if len(X)==3 else list(X), dtype=float)
    r1 = ((X[0] + mu)**2 + X[1]**2 + X[2]**2)**0.5
    R1 = [X[0] + mu, X[1], X[2]]/r1
    r2 = ((X[0] + mu - 1)**2 + X[1]**2 + X[2]**2)**0.5
    R2 = [X[0] + mu - 1, X[1], X[2]]/r2
    a_gravity = - (1 - mu) * R1 / r1**2 - mu * R2 / r2**2
    ang_velocity = np.array([0, 0, 1]) # otherwise it would not be CR3BP
    a_centripetal = - np.cross(ang_velocity, np.cross(ang_velocity, X[0:2]))
    a_coriolis = -2 * np.cross(ang_velocity, X[3:5])
    return a_gravity, a_centripetal, a_coriolis
 
a_gravity_SE, a_centripetal_SE, a_coriolis_SE = acceleration_vector(X_SE, mu_SE)
a_gravity_SM, a_centripetal_SM, a_coriolis_SM = acceleration_vector(X_SM, mu_SM)

### For visualization, I will focus on the Sun-Earth L2 point
ag_SE, ac_SE, acor_SE = acceleration_vector(X_SE, mu_SE)

fig = plt.figure(figsize=(15, 6), facecolor='white')
ax = fig.add_subplot(111, projection='3d', facecolor="white")

earth_pos = np.array([1-mu_SE, 0, 0])
X_pos     = np.array(X_SE[:3])

ax.scatter(*earth_pos, s=200, color='deepskyblue', zorder=5, label='Earth')
ax.scatter(*X_pos,     s=50,  color='black',       zorder=5, label='X')

scale = 0.004
for label, vec, color in [('Gravity',     ag_SE,   '#ff4d4d'),
                          ('Centripetal', ac_SE,   'forestgreen'), # Scurito per sfondo bianco
                          ('Coriolis',    acor_SE, 'darkorange'),  # Scurito per sfondo bianco
                          ('Net Accel',   ag_SE+ac_SE+acor_SE, "#0044ff")]:
    current_scale = scale * 0.5 if label == 'Net Accel' else scale
    
    v = vec / np.linalg.norm(vec) * current_scale
    ax.quiver(*X_pos, *v, color=color, linewidth=2.5, arrow_length_ratio=0.4,
              label=f'{label}  |a|={np.linalg.norm(vec):.2e}')

for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
    pane.fill = False
ax.tick_params(colors='black', labelsize=7)
ax.set_xlabel('x', color='black', fontsize=8)
ax.set_ylabel('y',           color='black', fontsize=8)
ax.set_zlabel('z',           color='black', fontsize=8)
ax.set_title('Accelerations at X — Sun-Earth CR3BP', color='black', fontsize=11, pad=10)
ax.legend(loc='upper left', fontsize=7.5, facecolor='white', labelcolor='black', framealpha=0.9, edgecolor='gray')
ax.set_xlim([0.995, 1.01]); ax.set_ylim([-0.01, 0.01]); ax.set_zlim([-0.01, 0.01])

plt.tight_layout()
plt.savefig('accel_3d.png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()


# Verification of the net acceleration at L2
# computation of total acceleration on the shifted point, including the solar sail effect
def net_accel(X, mu, n_rf_r1, Beta):
    """"ATTENTION, n must be in r1, theta, nu reference frame, 
    otherwise the solar sail acceleration will not be correctly computed"""
    ag, ac, acor = acceleration_vector(X, mu)
    r1 = ((X[0] + mu)**2 + X[1]**2 + X[2]**2)**0.5
    r1_hat = np.array([X[0] + mu, X[1], X[2]]) / r1
    # Rotational matrix R (DCM)
    gamma = np.arctan2(X[1], X[0] + mu) # angle between the position vector and the x-axis
    i = np.arctan2(X[2], np.sqrt((X[0] + mu)**2 + X[1]**2)) # angle between the position vector and the xy-plane
    R = np.array([[np.cos(i)*np.cos(gamma), np.sin(gamma)*np.cos(i), np.sin(i)],
                  [-np.sin(gamma), np.cos(gamma), 0],
                  [-np.cos(gamma)*np.sin(i), -np.sin(gamma)*np.sin(i), np.cos(i)]])
    n = R @ n_rf_r1
    a_solar_sail = Beta * (1 - mu) / r1**2 * ( r1_hat @ n )**2 * n
    return ag + ac + acor + a_solar_sail

n_rf_r1 = np.array([1, 0, 0]) # optimal orientation is along r1
Beta_SE = Beta(X_SE, mu_SE)
net_acceleration_SE = net_accel(X_SE, mu_SE, n_rf_r1, Beta_SE)
print(f'Net acceleration at S-E L2 with solar sail: {net_acceleration_SE} m/s^2')
Beta_SM = Beta(X_SM, mu_SM)
net_acceleration_SM = net_accel(X_SM, mu_SM, n_rf_r1, Beta_SM)
print(f'Net acceleration at S-M L2 with solar sail: {net_acceleration_SM} m/s^2')

# Verification thanks to "Sunjammer: Preliminary End-to-End Mission Desig" paper

def verify_tot_acceleration(X_shifted, mu, beta, n_hat):
    """
    Verify that the sum of accelerations at the artificial equilibrium point is ~0.
    Uses equations (14) and (15) from Heiligers et al. (2014).
    """
    r1_vec = np.array([X_shifted[0] + mu, X_shifted[1], X_shifted[2]])
    r1_norm = np.linalg.norm(r1_vec)
    r1_hat = r1_vec / r1_norm
    r2_vec = np.array([X_shifted[0] + mu - 1, X_shifted[1], X_shifted[2]])
    r2_norm = np.linalg.norm(r2_vec)
    # 2. gravitational acceleration (Eq. 14, first two terms)
    ag_sun = -(1 - mu) * r1_vec / (r1_norm**3)
    ag_planet = -mu * r2_vec / (r2_norm**3)
    a_grav = ag_sun + ag_planet
    # 3. centrifugal acceleration (in the CR3BP omega = 1, and acts only in the xy-plane)
    a_centrifuga = np.array([X_shifted[0], X_shifted[1], 0])
    # 4. solar sail acceleration (Eq. 15)
    a_sail = beta * ((1 - mu) / r1_norm**2) * (np.dot(r1_hat, n_hat)**2) * n_hat

    a_tot = a_grav + a_centrifuga + a_sail
    a_norm = np.linalg.norm(a_tot)
    return a_norm

a_tot_SE = verify_tot_acceleration(X_SE, mu_SE, Beta_SE, n_SE)
print(f'Norm of total acceleration at shifted S-E L2: {a_tot_SE:.2e} m/s^2 ')
a_tot_SM = verify_tot_acceleration(X_SM, mu_SM, Beta_SM, n_SM)
print(f'Norm of total acceleration at shifted S-M L2: {a_tot_SM:.2e} m/s^2 ')

# Verification by benchmark
X_verif1 = np.array([0.983867, 0, 0]) # True sub-L1 
Beta_verif1 = Beta(X_verif1, mu_SE)
X_verif2 = np.array([0.983908, -0.00144, 0]) # Targeted sub-L1 AEP 
# THE PAPER REPORTED y = -0.0144, 
# but it should be -0.00144 since the spacecraft should stay in the 5 degree exclusion zone
Beta_verif2 = Beta(X_verif2, mu_SE)
print(f'Beta at verification point 1: {Beta_verif1:.4f} (paper value: 0.0363)')
print(f'Beta at verification point 2: {Beta_verif2:.4f} (paper value: 0.0363)')


# TASK 2.3/2.4


# starting from the forumlation for the acceleration with solar pressure (P) and the gravitational mu, 
# we can derive the lightness number (Beta) depending on the solar pressure constant at Eart 
# (P = W_E/c = 4.56e-6 N/m^2), the  Sun gravitational parameter (mu), 1 AU, and the area/mass ratio
#  of the sail. The formula is as follows:
# Beta = (2 * P_Earth * 1AU^2 * A/m / mu_S)
def lenght_sail( Beta, m):
    """ Calculate the area of the solar sail given the lightness number (Beta) and other parameters."""
    mu_S = 132712e15  # m^3/s^2
    P = 1368/299792458  # N/m^2
    AU = 149597870700  # m
    A = Beta * mu_S * m / (2 * P * AU**2)
    length = np.sqrt(A)    
    return length

m_spacecraft = 100 # kg
A_SE = lenght_sail(Beta(X_SE, mu_SE), m_spacecraft)
print(f'Length of the solar sail for Sun-Earth L2: {A_SE:.2f} m')
A_SM = lenght_sail(Beta(X_SM, mu_SM), m_spacecraft)
print(f'Length of the solar sail for Sun-Mars L2: {A_SM:.2f} m')

# Sensitivity analysis
x_shift = np.linspace(-0.002, -0.004, 100)
y_shift = np.linspace(-0.002, 0.002, 100)
Beta_vec_SE = np.array([Beta([L2_SE + x, 0, 0], mu_SE) for x in x_shift])
Beta_vec_SM = np.array([Beta([L2_SM + x, 0, 0], mu_SM) for x in x_shift])
plt.figure(figsize=(10, 6))
plt.plot(L2_SE + x_shift, Beta_vec_SE, label='Sun-Earth L2', color='blue')
plt.plot(L2_SM + x_shift, Beta_vec_SM, label='Sun-Mars L2', color='red')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.title('Lightness number vs Artificial L2 x-coordinate') 
plt.xlabel('x_i coordinates [-]')
plt.ylabel('Lightness number, Beta [-]')
plt.legend()
plt.grid()
plt.show()


# EXTRA REPRESENTATION OF THE SENSITIVITY ANALYSIS IN 3D
X_shift, Y_shift = np.meshgrid(x_shift, y_shift)
Beta_matrix_SE = np.array([[Beta([L2_SE + x, y, 0], mu_SE) for x in x_shift] for y in y_shift])
Beta_matrix_SM = np.array([[Beta([L2_SM + x, y, 0], mu_SM) for x in x_shift] for y in y_shift])
fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')
surf1 = ax1.plot_surface(X_shift, Y_shift, Beta_matrix_SE, cmap='viridis')
surf2 = ax2.plot_surface(X_shift, Y_shift, Beta_matrix_SM, cmap='plasma')
ax1.set_title('Beta around Sun-Earth L2')
ax1.set_xlabel('x Shift (m)')
ax1.set_ylabel('y Shift (m)')
ax1.set_zlabel('Beta')
ax2.set_title('Beta around Sun-Mars L2')
ax2.set_xlabel('x Shift (m)')
ax2.set_ylabel('y Shift (m)')
ax2.set_zlabel('Beta')
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
plt.tight_layout()
plt.show()





