import numpy as np

def dU_dx(x, mu):
    term1 = (1 - mu) / abs(x + mu)**3 * (x + mu)
    term2 = mu / abs(x + mu - 1)**3 * (x + mu - 1)
    return x - term1 - term2
def d2U_dx2(x, mu):
    term1 = -2 * (1 - mu) / abs(x + mu)**3 
    term2 = -2 * mu / abs(x + mu - 1)**3 
    return 1 - term1 - term2
def d2r(x, mu):
    "Total acceleration at x"
    r1 = abs(x + mu)
    r2 = abs(x + mu - 1)
    d2r = -(( 1 - mu )/r1**3 * (x + mu) + mu/r2**3 * (x + mu - 1)) + x 
    return d2r


mu_SM = 3.227154996101724e-7 # Sun-Mars mass ratio
tol = 1e-12
x = 1.01
err_rel = abs(dU_dx(x, mu_SM) / d2U_dx2(x, mu_SM))

# Newton's method
while err_rel > tol and abs(d2U_dx2(x, mu_SM)) > tol:
    x = x - dU_dx(x, mu_SM) / d2U_dx2(x, mu_SM)
    err_rel = abs(dU_dx(x, mu_SM) / d2U_dx2(x, mu_SM))
L2_SM = x
print(f'Sun - Mars L2: {x}')
print(f'Verification S-M d2r/dt2: {d2r(x, mu_SM)}')

mass2 = 5.972e24 # Earth mass
mu_SE = 3.0542e-6 # Sun-Earth mass ratio
err_rel = abs(dU_dx(x, mu_SE) / d2U_dx2(x, mu_SE))

while err_rel > tol and abs(d2U_dx2(x, mu_SE)) > tol:
    x = x - dU_dx(x, mu_SE) / d2U_dx2(x, mu_SE)
    err_rel = abs(dU_dx(x, mu_SE) / d2U_dx2(x, mu_SE))
L2_SE = x
print(f'Sun - Earth L2: {x}')
print(f'Verification S-E d2r/dt2: {d2r(x, mu_SE)}')


# L1, needed for verificationat task 3
x = 0.99
err_rel = abs(dU_dx(x, mu_SE) / d2U_dx2(x, mu_SE))

# Newton's method
while err_rel > tol and abs(d2U_dx2(x, mu_SE)) > tol:
    x = x - dU_dx(x, mu_SE) / d2U_dx2(x, mu_SE)
    err_rel = abs(dU_dx(x, mu_SE) / d2U_dx2(x, mu_SE))
L1_SE = x
print(f'Sun - Earth L1: {x}')