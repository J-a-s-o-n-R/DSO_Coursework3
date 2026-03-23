# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 19:00:18 2026

@author: migue
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

upper = pd.read_csv('RAE2822_upper.csv', header=None, names=['x', 'y_original', 'y_CST'])
lower = pd.read_csv('RAE2822_lower.csv', header=None, names=['x', 'y_original', 'y_CST'])

x_upper     = upper['x'].values[:-1]
y_upper     = upper['y_original'].values[:-1]
y_upper_cst = upper['y_CST'].values[:-1]

x_lower     = lower['x'].values[:-1]
y_lower     = lower['y_original'].values[:-1]
y_lower_cst = lower['y_CST'].values[:-1]

# Residuals
res_upper = y_upper - y_upper_cst
res_lower = y_lower - y_lower_cst
res_all   = np.concatenate([res_upper, res_lower])

# -------------------------------------------------------
# Plot 1: Airfoil comparison
# -------------------------------------------------------
fig1, ax1 = plt.subplots()
ax1.plot(x_upper, y_upper,     'r',  linewidth=2,   label='Original')
ax1.plot(x_lower, y_lower,     'r',  linewidth=2)
ax1.plot(x_upper, y_upper_cst, 'b--',linewidth=1.5, label='CST fit')
ax1.plot(x_lower, y_lower_cst, 'b--',linewidth=1.5)
ax1.set_xlabel('x/c')
ax1.set_ylabel('y/c')
ax1.set_title('RAE2822 — Original vs CST Fit')
ax1.legend(loc='best')
ax1.grid(True)
plt.tight_layout()
plt.savefig('RAE2822_airfoil.png', dpi=200, bbox_inches='tight')
plt.show()

# -------------------------------------------------------
# Plot 2: Residuals
# -------------------------------------------------------
fig2, ax2 = plt.subplots()
ax2.plot(x_upper, res_upper, 'r', linewidth=1.5, label='Upper residual')
ax2.plot(x_lower, res_lower, 'b', linewidth=1.5, label='Lower residual')
ax2.axhline(0, color='k', linestyle='--', linewidth=0.8)
ax2.set_xlabel('x/c')
ax2.set_ylabel('y - y$_{CST}$')
ax2.set_title('Residuals')
ax2.legend(loc='best')
ax2.grid(True)
plt.tight_layout()
plt.savefig('RAE2822_residuals.png', dpi=200, bbox_inches='tight')
plt.show()

# -------------------------------------------------------
# Plot 3: Absolute error (log scale)
# -------------------------------------------------------
fig3, ax3 = plt.subplots()
ax3.semilogy(x_upper, np.abs(res_upper), 'r', linewidth=1.5, label='Upper |error|')
ax3.semilogy(x_lower, np.abs(res_lower), 'b', linewidth=1.5, label='Lower |error|')
ax3.set_xlabel('x/c')
ax3.set_ylabel('|y - y$_{CST}$|')
ax3.set_title('Absolute error (log scale)')
ax3.legend(loc='best')
ax3.grid(True)
plt.tight_layout()
plt.savefig('RAE2822_error.png', dpi=200, bbox_inches='tight')
plt.show()

# -------------------------------------------------------
# Summary statistics
# -------------------------------------------------------
print('--- Upper surface ---')
print(f'  Max absolute error : {np.max(np.abs(res_upper)):.4e}')
print(f'  Mean absolute error: {np.mean(np.abs(res_upper)):.4e}')
print(f'  RMS error          : {np.sqrt(np.mean(res_upper**2)):.4e}')
print(f'  SSE                : {np.sum(res_upper**2):.4e}')

print('--- Lower surface ---')
print(f'  Max absolute error : {np.max(np.abs(res_lower)):.4e}')
print(f'  Mean absolute error: {np.mean(np.abs(res_lower)):.4e}')
print(f'  RMS error          : {np.sqrt(np.mean(res_lower**2)):.4e}')
print(f'  SSE                : {np.sum(res_lower**2):.4e}')

print('--- Combined ---')
print(f'  Max absolute error : {np.max(np.abs(res_all)):.4e}')
print(f'  Mean absolute error: {np.mean(np.abs(res_all)):.4e}')
print(f'  RMS error          : {np.sqrt(np.mean(res_all**2)):.4e}')
print(f'  SSE                : {np.sum(res_all**2):.4e}')


# CST start values all zero before optimization
# CST coefficients
A_upper = np.array([0.1283, 0.1267, 0.1607, 0.1494, 0.1511, 
                    0.2241, 0.1608, 0.2100, 0.1861, 0.2105])

A_lower = np.array([-0.1293, -0.1318, -0.1704, -0.0706, -0.3387,
                     0.0097, -0.2005, -0.0355, -0.0439,  0.0643])

X0 = np.concatenate([A_upper, A_lower])
#print(X0.tolist())

SSE = 7.585345e-08

# Print summary
print(f'Best total squared error: {SSE:.6e}')
print(f'\nNumber of CST modes: {len(A_upper)}')

print('\nUpper surface CST coefficients:')
for i, a in enumerate(A_upper):
    print(f'  A{i+1:02d} = {a:.4f}')

print('\nLower surface CST coefficients:')
for i, a in enumerate(A_lower):
    print(f'  A{i+1:02d} = {a:.4f}')


alpha_initial = 1.4  # degrees

# ============================================================
# PART 2: Optimised CST coefficients
# ============================================================
A_upper_opt = np.array([0.1099, 0.1930, 0.0941, 0.1914, 0.1666,
                         0.1764, 0.1732, 0.2705, 0.2259, 0.2641])

A_lower_opt = np.array([-0.0431, -0.0867, -0.2883, -0.0652, -0.1129,
                          0.0096, -0.3056, -0.0394, -0.0501,  0.0573])

alpha_opt = 1.3707  # degrees

# ============================================================
# Aerodynamic coefficients
# ============================================================
aero_initial = {'Cl': 0.64233,  'Cd': 0.013647, 'Cm': -0.12816}
aero_opt     = {'Cl': 0.750023, 'Cd': 0.014818,  'Cm': -0.15000}

# ============================================================
# Thickness analysis
# ============================================================
xt = [0.20, 0.60]  # chord locations

# Upper and lower surface y-values at xt
upper_initial = np.array([0.0517, 0.0570])
lower_initial = np.array([-0.0515, -0.0377])
upper_opt     = np.array([0.0535, 0.0596])
lower_opt     = np.array([-0.0496, -0.0351])

# Thickness = upper - lower
thickness_initial = upper_initial - lower_initial  # [0.1032, 0.0947]
thickness_opt     = upper_opt     - lower_opt      # [0.1031, 0.0947]

# Camber = (upper + lower) / 2
camber_initial = (upper_initial + lower_initial) / 2
camber_opt     = (upper_opt     + lower_opt)     / 2

# ============================================================
# Optimisation settings
# ============================================================
Mach    = 0.73
Cl_min  = 0.75   # lift constraint
Cm_min  = -0.15  # pitching moment constraint
t_min   = thickness_initial  # thickness must not go below initial

# Bounds on CST coefficients (scaling factors)
scale_ub = 3.0
scale_lb = 1/3

# Alpha bounds
alpha_lb = -2.0  # degrees
alpha_ub =  5.0  # degrees

# ============================================================
# Key findings
# ============================================================
# - Cl increased from 0.642 to 0.750 (initial airfoil was infeasible)
# - Cd increased from 0.01365 to 0.01482 (drag penalty for lift constraint)
# - Cm decreased from -0.128 to -0.150 (moment constraint active)
# - Thickness unchanged at both locations (constraint active)
# - Camber increased — optimiser added camber to meet Cl constraint
#   rather than increasing thickness
# - Alpha slightly decreased from 1.4 to 1.3707 degrees