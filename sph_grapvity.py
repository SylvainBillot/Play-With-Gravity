import numpy as np
from numba import njit, prange
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
    
# helpers
@njit
def asarray(a):
    return np.asarray(a, dtype=np.float64)

@njit
def to_device_scalar(x):
    return np.array(x, dtype=np.float64)

#!/usr/bin/env python3

# SPH + self-gravity 3D particles
G = 1.0                 # Gravitational constant
N = 5000                # Number of particles
h = 0.05                # Smoothing length
rho0 = 0.01             # Rest density
k = 20.0                # Gas stiffness 
mu = 0.01               # Viscosity coefficient
dt = 0.0001             # Time step
steps = 1500            # Number of simulation steps
soft = 0.04             # Softening length for gravity to avoid singularities
mass_minimum = 0.01     # Minimum mass for particles to avoid zero mass issues
mass_maximum = 0.1      # Maximum mass for particles to avoid excessively large masses

dotsize = 10            # Adjust dot size based on number of particles for better visibility

rng = np.random.default_rng(42)
# Generate uniform distribution inside a sphere of radius 1.0
radius = 0.7
r = radius * rng.uniform(0, 1, N)**(1/3)  # uniform in volume
cos_theta = rng.uniform(-1, 1, N)  # uniform in cosine to avoid pole clustering
theta = np.arccos(cos_theta)
phi = rng.uniform(0, 2*np.pi, N)
x = r * np.sin(theta) * np.cos(phi)
y = r * np.sin(theta) * np.sin(phi) 
z = r * cos_theta
pos = np.column_stack([x, y, z])
vel = rng.normal(scale=0.05, size=(N, 3))
mass = rng.uniform(mass_minimum, mass_maximum, N)

pos = asarray(pos)
vel = asarray(vel)
mass = asarray(mass)

# Compute density and pressure for each particle
@njit(fastmath=True, parallel=True)
def compute_density_pressure_optimized(pos, mass, h, rho0, k):
    N = pos.shape[0]
    rho = np.zeros(N)
    h2 = h**2
    # Constant for the Poly6 kernel
    poly6_factor = 315.0 / (64.0 * np.pi * h**9)

    for i in prange(N):
        for j in range(N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            r2 = dx*dx + dy*dy + dz*dz
            
            if r2 < h2:
                rho[i] += mass[j] * poly6_factor * (h2 - r2)**3
    
    # Pressure calculation (vectorized on result is fine)
    P = k * np.maximum(rho - rho0, 0.0)
    return rho, P

# Optimized version of compute_forces with manual loop unrolling and reduced redundant calculations
@njit(fastmath=True, parallel=True)
def compute_forces_optimized(pos, vel, mass, rho, P, h, mu, G, soft):
    N = pos.shape[0]
    f = np.zeros((N, 3), dtype=np.float64)
    
    # Precompute constants
    prefactor = -45.0 / (np.pi * h**6)
    visc_prefactor = 45.0 / (np.pi * h**6)
    soft2 = soft**2

    for i in prange(N):
        f_ix, f_iy, f_iz = 0.0, 0.0, 0.0
        
        # Cache i-particle data
        pi = pos[i]
        vi = vel[i]
        rho_i = rho[i]
        # Original: P_i = P / rho**2
        p_rho_i = P[i] / (rho_i**2)
        
        for j in range(N):
            if i == j: continue
            
            # 1. Replicate: diff = pos[i] - pos[j]
            dx = pi[0] - pos[j, 0]
            dy = pi[1] - pos[j, 1]
            dz = pi[2] - pos[j, 2]
            r2 = dx*dx + dy*dy + dz*dz
            dist = np.sqrt(r2)

            # --- Gravity (Original: f_grav = G * sum(mass_j * (-diff) * invr3)) ---
            r2_grav = r2 + soft2
            inv_r3 = (G * mass[j]) / (r2_grav * np.sqrt(r2_grav))
            # Note the minus sign to match (-diff)
            f_ix -= dx * inv_r3
            f_iy -= dy * inv_r3
            f_iz -= dz * inv_r3

            # --- SPH Pressure & Viscosity ---
            if 1e-12 < dist < h:
                h_r = h - dist
                
                # A. Pressure
                # Original logic: pres_matrix = -(P_i + P_j) * mass_j * gradW
                # Where gradW = prefactor * (h-dist)^2 * (r_for_grad / dist)
                # And r_for_grad = -diff
                
                # Corrected scalar: -(P_i + P_j) * mass_j * prefactor * (h-r)^2 / r
                # Since r_for_grad = -diff, the two negatives cancel out.
                p_term = (p_rho_i + P[j]/(rho[j]**2)) * mass[j]
                # We multiply by dx (the 'diff' component)
                p_scalar = p_term * prefactor * (h_r**2) / dist
                
                f_ix += dx * p_scalar
                f_iy += dy * p_scalar
                f_iz += dz * p_scalar

                # B. Viscosity
                # Original logic: mu * (-vel_diff / rho_j) * visc_lap * mass_j
                # vel_diff = vel[i] - vel[j]
                v_scalar = mu * (mass[j] / rho[j]) * (visc_prefactor * h_r)
                f_ix -= (vi[0] - vel[j, 0]) * v_scalar
                f_iy -= (vi[1] - vel[j, 1]) * v_scalar
                f_iz -= (vi[2] - vel[j, 2]) * v_scalar

        f[i, 0] = f_ix
        f[i, 1] = f_iy
        f[i, 2] = f_iz
        
    return f

# Compute potential energy of the system
@njit(fastmath=True, parallel=True)
def compute_potential_energy_optimized(pos, mass, G, soft):
    N = pos.shape[0]
    total_pe = 0.0
    soft2 = soft**2
    
    # prange distributes the outer loop across CPU cores
    # We use a scalar accumulator to avoid allocating N*N matrices
    for i in prange(N):
        sub_pe = 0.0
        # Symmetry: only calculate for j > i
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            
            r2 = dx*dx + dy*dy + dz*dz + soft2
            
            # Potential Energy = -G * (m1 * m2) / r
            sub_pe += (mass[i] * mass[j]) / np.sqrt(r2)
        
        total_pe += sub_pe
        
    return -G * total_pe

fig = plt.figure(figsize=(14,7))
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_proj_type('persp')  # Enable perspective projection
ax1.view_init(elev=20, azim=45)  # Set a nice viewing angle
ax2 = fig.add_subplot(122)
ax3 = ax2.twinx()  # Create a twin axis for Temperature
pos_plot = pos
scat = ax1.scatter(pos_plot[:,0], pos_plot[:,1], pos_plot[:,2], s=np.sqrt(mass) * dotsize, c='cyan')
ax1.set_xlim(-1,1); ax1.set_ylim(-1,1); ax1.set_zlim(-1,1); ax1.set_facecolor('k')
ax1.set_box_aspect([1, 1, 1])  # Ensure equal aspect ratio for 3D axes
ax1.axis('off')

# Energy plots
ke_list = []
pe_list = []
te_list = []
temp_list = []
line_ke, = ax2.plot([], [], 'r-', label='Kinetic Energy')
line_pe, = ax2.plot([], [], 'b-', label='Potential Energy')
line_te, = ax2.plot([], [], 'g-', label='Total Energy')
line_temp, = ax3.plot([], [], 'm--', label='Avg Temperature') # Magenta dashed line
lines = [line_ke, line_pe, line_te, line_temp]
labels = [l.get_label() for l in lines]
ax2.legend(lines, labels, loc='upper left')

ax2.set_xlabel('Time Step')
ax2.set_ylabel('Energy')
ax3.set_ylabel('Temperature (Diagnostic)', color='m')
ax3.tick_params(axis='y', labelcolor='m')

# Create energy text at figure level
energy_text_obj = fig.text(0.5, 0.02, '', ha='center', va='bottom', fontsize=11, 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Mouse wheel zoom/unzoom
def on_scroll(event):
    base_scale = 1.15
    if event.button == 'up':
        scale = 1.0/base_scale
    elif event.button == 'down':
        scale = base_scale
    else:
        return

    xlim = ax1.get_xlim3d()
    ylim = ax1.get_ylim3d()
    zlim = ax1.get_zlim3d()

    x_mid = (xlim[0] + xlim[1]) * 0.5
    y_mid = (ylim[0] + ylim[1]) * 0.5
    z_mid = (zlim[0] + zlim[1]) * 0.5

    x_half = (xlim[1] - xlim[0]) * 0.5 * scale
    y_half = (ylim[1] - ylim[0]) * 0.5 * scale
    z_half = (zlim[1] - zlim[0]) * 0.5 * scale

    ax1.set_xlim3d([x_mid - x_half, x_mid + x_half])
    ax1.set_ylim3d([y_mid - y_half, y_mid + y_half])
    ax1.set_zlim3d([z_mid - z_half, z_mid + z_half])

    fig.canvas.draw_idle()

fig.canvas.mpl_connect('scroll_event', on_scroll)

# Combined function to update position, velocity, and calculate kinetic energy in one pass
@njit(fastmath=True)
def integrate_and_ke(pos, vel, F, mass, dt):
    # Update velocity and position
    accel = F / mass[:, np.newaxis]
    vel += accel * dt
    pos += vel * dt
    
    # Calculate Kinetic Energy in the same pass
    ke = 0.5 * np.sum(mass * np.sum(vel**2, axis=1))
    return ke

# Animation update function
def update(frame):
    global pos, vel, energy_text_obj
    rho, P = compute_density_pressure_optimized(pos, mass, h, rho0, k)
    F = compute_forces_optimized(pos, vel, mass, rho, P, h, mu, G, soft)
    ke = integrate_and_ke(pos, vel, F, mass, dt)
    pe = float(compute_potential_energy_optimized(pos, mass, G, soft))
    te = ke + pe
  
    # Calculate Average Temperature (P/rho)
    # We add a tiny epsilon to rho to avoid division by zero
    avg_temp = np.mean(P / (rho + 1e-9))
    temp_list.append(avg_temp)


    ke_list.append(ke)
    pe_list.append(pe)
    te_list.append(te)
    temp_list.append(avg_temp)

    pos_plot = pos
    scat._offsets3d = (pos_plot[:,0], pos_plot[:,1], pos_plot[:,2])

    line_ke.set_data(range(len(ke_list)), ke_list)
    line_pe.set_data(range(len(pe_list)), pe_list)
    line_te.set_data(range(len(te_list)), te_list)
    line_temp.set_data(range(len(temp_list)), temp_list)
    
    ax2.set_xlim(0, len(ke_list))
    ax2.relim()
    ax2.autoscale_view()
    # Autoscale Temperature axis separately
    ax3.relim()
    ax3.autoscale_view()
    
    # Update energy text
    energy_text = f'KE: {ke:.4f} | PE: {pe:.4f} | TE: {te:.4f} | Avg Temp: {avg_temp:.4f}'
    energy_text_obj.set_text(energy_text)

    return scat, line_ke, line_pe, line_te

ani = FuncAnimation(fig, update, frames=steps, interval=0, blit=False)
plt.show()