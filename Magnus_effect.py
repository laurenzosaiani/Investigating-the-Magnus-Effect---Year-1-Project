import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# Physical constants
g = 9.81
rho = 1.225
r = 0.019805
m = 0.0027
C_D = 0.47
A = np.pi * r**2
mu = 1.81e-5
I = (2 / 5) * m * r**2

# Inputs
angle_deg = 72#float(input("Launch angle (vertical)(degrees): "))
angle_hor = 0#float(input("Launch angle (horizontal)(degrees): "))
v0 = 16#float(input("Initial velocity (m/s): "))
h0 = 0.97#float(input("Initial height (m): "))
rev_per_s = 13.1#float(input("Initial spin rate (rev/s): "))
polar_deg = 90#float(input("Spin polar angle (degrees): "))
azimuthal_deg = 90#float(input("Spin azimuthal angle (degrees): "))
requested_points = 1000

# Conversions
angle_rad = np.radians(angle_deg)
angle_horrad = np.radians(angle_hor)
omega_0 = 2 * np.pi * rev_per_s  # rad/s
polar_rad = np.radians(polar_deg)
azimuthal_rad = np.radians(azimuthal_deg)

# Velocity components
v0x = v0 * np.sin(angle_rad) * np.cos(angle_horrad)
v0z = v0 * np.sin(angle_rad) * np.sin(angle_horrad)
v0y = v0 * np.cos(angle_rad)

# Initial state
y0 = [0, h0, 0, v0x, v0y, v0z, omega_0]

# Magnus damping coefficient
k_spin = 16 / 15 * np.pi * r**5 * rho * C_D

def dynamics(t, y):
    x, y_pos, z, vx, vy, vz, omega = y
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    vx_unit, vy_unit, vz_unit = (vx / v, vy / v, vz / v) if v != 0 else (0, 0, 0)

    Sx = omega * np.sin(polar_rad) * np.cos(azimuthal_rad)
    Sy = omega * np.sin(polar_rad) * np.sin(azimuthal_rad)
    Sz = omega * np.cos(polar_rad)

    F_D = 0.5 * rho * v**2 * A * C_D
    ax_drag = -F_D * vx_unit / m
    ay_drag = -F_D * vy_unit / m
    az_drag = -F_D * vz_unit / m

    S_mag = r * omega / v if v != 0 else 0
    C_L = (1.6 * S_mag) / (1 + 0.5 * S_mag) if v != 0 else 0
    C_L = min(C_L, 0.5)

    v_vec = np.array([vx, vy, vz])
    omega_vec = np.array([Sx, Sy, Sz])
    v_vec = np.array([vx, vy, vz])
    omega_vec = np.array([Sx, Sy, Sz])
    lift_dir = np.cross(omega_vec, v_vec)
    lift_mag = np.linalg.norm(lift_dir)

    if v > 1.0 and lift_mag > 1e-8:  # Suppress at low velocity and low lift
        lift_dir_unit = lift_dir / lift_mag
        F_M = 0.5 * rho * v**2 * A * C_L * lift_dir_unit
    else:
        F_M = np.zeros(3)

    ax_magnus, ay_magnus, az_magnus = F_M / m
    ax = ax_drag + ax_magnus
    ay = -g + ay_drag + ay_magnus
    az = az_drag + az_magnus

    domega_dt = - (k_spin / I) * omega**2
    return [vx, vy, vz, ax, ay, az, domega_dt]

def dynamics_no_spin(t, y):
    x, y_pos, z, vx, vy, vz = y
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    vx_unit, vy_unit, vz_unit = (vx / v, vy / v, vz / v) if v != 0 else (0, 0, 0)

    F_D = 0.5 * rho * v**2 * A * C_D
    ax_drag = -F_D * vx_unit / m
    ay_drag = -F_D * vy_unit / m
    az_drag = -F_D * vz_unit / m

    ax = ax_drag
    ay = -g + ay_drag
    az = az_drag

    return [vx, vy, vz, ax, ay, az]

def hit_ground(t, y): return y[1]
hit_ground.terminal = True
hit_ground.direction = -1

def set_axes_equal(ax):
    '''Set 3D plot axes to equal scale, but clamp the Z (vertical) axis at 0.'''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    # Clamp the bottom of the Z axis at 0:
    zmin = max(0, z_middle - plot_radius)
    zmax = z_middle + plot_radius
    ax.set_zlim3d([zmin, zmax])

# Initial state for no-spin trajectory (same position, velocity; no omega)
y0_no_spin = [0, h0, 0, v0x, v0y, v0z]

# Pre-solve to find impact time for no-spin case
pre_sol_no_spin = solve_ivp(dynamics_no_spin, (0, 5), y0_no_spin, events=hit_ground, rtol=1e-6, atol=1e-9)
impact_time_no_spin = pre_sol_no_spin.t_events[0][0] if pre_sol_no_spin.t_events[0].size > 0 else 5

# Solve no-spin dynamics
sol_no_spin = solve_ivp(dynamics_no_spin, (0, impact_time_no_spin), y0_no_spin,
                        t_eval=np.linspace(0, impact_time_no_spin, requested_points),
                        rtol=1e-6, atol=1e-9)

# Extract no-spin trajectory
x_ns, y_pos_ns, z_ns = sol_no_spin.y[0], sol_no_spin.y[1], sol_no_spin.y[2]

# Pre-solve to find impact time
pre_sol = solve_ivp(dynamics, (0, 5), y0, events=hit_ground, rtol=1e-6, atol=1e-9)
impact_time = pre_sol.t_events[0][0] if pre_sol.t_events[0].size > 0 else 5

# Solve with time steps
t_eval = np.linspace(0, impact_time, requested_points)
sol = solve_ivp(dynamics, (0, impact_time), y0, t_eval=t_eval, rtol=1e-6, atol=1e-9)

# Extract solution
x, y_pos, z = sol.y[0], sol.y[1], sol.y[2]

# For trajectory with spin
valid_idx = y_pos >= 0
x = x[valid_idx]
y_pos = y_pos[valid_idx]
z = z[valid_idx]

# For no-spin trajectory
valid_idx_ns = y_pos_ns >= 0
x_ns = x_ns[valid_idx_ns]
y_pos_ns = y_pos_ns[valid_idx_ns]
z_ns = z_ns[valid_idx_ns]

# === Real-Life Ball Simulation (Estimated Spin) ===

# Real-life launch parameters
angle_deg_rl = 70   
angle_hor_rl = 0   # straight forward
v0_rl = 16         # m/s
h0_rl = 0.97       # initial height
rev_per_s_rl = 13.1  # estimated spin
polar_deg_rl = 90
azimuthal_deg_rl = 90

start_pos = [0, 0, 0.97]
end_pos = [5.8, -0.8, 1.54]

# Generate a curved trajectory from start_pos to end_pos (realistic approx)
real_life_points = 300
t_rl = np.linspace(0, 1, real_life_points)  # normalized time parameter

# Start and end coordinates
x0, z0, y0_rl = start_pos
x1, z1, y1_rl = end_pos

# Horizontal path: curved in Z (simulate Magnus effect)
x_rl = np.linspace(x0, x1, real_life_points)
z_rl = np.linspace(z0, z1, real_life_points) + 0.3 * np.sin(np.pi * t_rl)  # lateral curve

# Adjust this control height to flatten or raise the vertical arc:
control_height = max(start_pos[2], end_pos[2]) + 0.5  # smaller -> flatter

def bezier_y(t, y0, yc, y1):
    return (1 - t)**2 * y0 + 2 * (1 - t) * t * yc + t**2 * y1

y_rl = bezier_y(t_rl, start_pos[2], control_height, end_pos[2])


# Compute standard deviations from uncertainties
angle_unc = np.radians(30)
angle_hor_unc = np.radians(5)
v0_unc = 2
h0_unc = 0.01
rev_unc = 3
polar_unc = np.radians(35)
azimuthal_unc = np.radians(35)

# Recompute the real-life trajectory bounds with uncertainties
t_rl = np.linspace(0, 1, real_life_points)

# Main (mean) trajectory
x_rl_mean = np.linspace(x0, x1, real_life_points)
z_rl_mean = np.linspace(z0, z1, real_life_points) + 0.3 * np.sin(np.pi * t_rl)
y_rl_mean = bezier_y(t_rl, start_pos[2], control_height, end_pos[2])

# Estimate uncertainty in y using variation in control height
control_height_unc = control_height + 0.5  # exaggerate for visibility
y_upper = bezier_y(t_rl, start_pos[2], control_height_unc, end_pos[2])
y_rl_std = np.abs(y_upper - y_rl_mean)

# Estimate uncertainty in z using variation in lateral curvature
z_upper = np.linspace(z0, z1, real_life_points) + (0.3 + 0.2) * np.sin(np.pi * t_rl)
z_rl_std = np.abs(z_upper - z_rl_mean)


# --- Static 3D Plot ---
fig = plt.figure(figsize = (10,7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, z, y_pos, color='blue', label='With Spin')
ax.plot(x_ns, z_ns, y_pos_ns, color='red', linestyle='--', label='No Spin')
ax.plot(x_rl_mean, z_rl_mean, y_rl_mean, color='green', label='Real-Life')
ax.errorbar(x_rl_mean[::40], z_rl_mean[::40], y_rl_mean[::40],
            yerr=y_rl_std[::40], zerr=z_rl_std[::40],
            fmt='.', color='green', alpha=0.5, label='Uncertainty', capsize=3)
ax.set_xlabel("X (m)")
ax.set_ylabel("Z (m)")
ax.set_zlabel("Y (m)")
ax.set_title("3D Trajectory Comparison")
ax.legend()
ax.set_zlim(bottom = 0)
set_axes_equal(ax)
plt.show()

# --- Top-down (X-Z) Plot with Error Bars ---
plt.figure()
plt.plot(x, z, label='With Spin', color='blue')
plt.plot(x_ns, z_ns, label='No Spin', color='red', linestyle='--')
plt.plot(x_rl, z_rl, label='Real-Life', color='green')
plt.errorbar(x_rl[::40], z_rl[::40], yerr=z_rl_std[::40],
             fmt='.', color='green', alpha=0.5, label='Uncertainty', capsize=3)
plt.xlabel("X (m)")
plt.ylabel("Z (m)")
plt.title("Top-Down (X-Z) View")
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()

# --- Side view (X-Y) Plot with Error Bars ---
plt.figure()
plt.plot(x, y_pos, label='With Spin', color='blue')
plt.plot(x_ns, y_pos_ns, label='No Spin', color='red', linestyle='--')
plt.plot(x_rl, y_rl, label='Real-Life', color='green')
plt.errorbar(x_rl[::40], y_rl[::40], yerr=y_rl_std[::40],
             fmt='.', color='green', alpha=0.5, label='Uncertainty', capsize=3)
plt.xlabel("Horizontal distance (X) [m]")
plt.ylabel("Vertical position (Y) [m]")
plt.title("Trajectory in X-Y Plane (Side View)")
plt.grid(True)
plt.ylim(bottom=0)
plt.legend()
plt.show()