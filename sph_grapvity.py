import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# helpers
def asarray(a):
    return np.asarray(a, dtype=np.float64)

def to_device_scalar(x):
    return np.array(x, dtype=np.float64)

def to_cpu(a):
    return a

#!/usr/bin/env python3

# SPH + self-gravity 3D particles
G = 1.0                 # Gravitational constant
N = 500                 # Number of particles
h = 0.1                 # Smoothing length
rho0 = 0.01              # Rest density
k = 20.0                # Gas stiffness 
mu = 0.01               # Viscosity coefficient
dt = 0.0001             # Time step
steps = 1500            # Number of simulation steps
soft = 0.01             # Softening length for gravity to avoid singularities


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
mass = np.ones(N) * 0.05

pos = asarray(pos)
vel = asarray(vel)
mass = asarray(mass)

# SPH kernel functions
def poly6(r2, h):
    hr2 = h*h - r2
    return np.where(r2 < h*h, 315.0 / (64*np.pi*h**9) * hr2**3, 0.0)

# Gradient of spiky kernel
def spiky_grad(r, rlen, h):
    mask = (rlen > 1e-12) & (rlen < h)
    coef = -45.0 / (np.pi*h**6) * (h - rlen[mask])**2
    out = np.zeros_like(r)
    out[mask] = coef[:, None] * r[mask] / rlen[mask, None]
    return out

# Viscosity Laplacian
def viscosity_lap(rlen, h):
    return np.where((rlen>0) & (rlen<h),
                    45.0 / (np.pi*h**6) * (h - rlen),
                    0.0)

# Compute density and pressure for each particle
def compute_density_pressure(pos, mass):
    diff = pos[:, None, :] - pos[None, :, :]
    r2 = np.sum(diff**2, axis=2)
    W = poly6(r2, h)
    rho = np.sum(mass[None, :] * W, axis=1)
    P = k * np.maximum(rho - rho0, to_device_scalar(0.0))
    return rho, P

# Compute forces on each particle from pressure, viscosity, and gravity
def compute_forces(pos, vel, rho, P):
    diff = pos[:, None, :] - pos[None, :, :]  # (N, N, 3), diff[i,j] = pos[i] - pos[j]
    rlen = np.linalg.norm(diff, axis=2)  # (N, N)
    r2 = np.sum(diff**2, axis=2)  # (N, N)
    
    # SPH pressure
    r_for_grad = -diff  # pos[j] - pos[i]
    mask = (rlen > 1e-12) & (rlen < h)
    coef = -45.0 / (np.pi * h**6) * (h - rlen[mask])**2
    gradW_matrix = np.zeros((N, N, 3))
    gradW_matrix[mask] = coef[:, None] * r_for_grad[mask] / rlen[mask, None]
    
    P_i = P[:, None] / rho[:, None]**2  # (N, 1)
    P_j = P[None, :] / rho[None, :]**2  # (1, N)
    total_P = P_i + P_j  # (N, N)
    mass_j = mass[None, :]  # (1, N)
    pres_matrix = -total_P[:, :, None] * mass_j[:, :, None] * gradW_matrix
    f_pres = np.sum(pres_matrix, axis=1)
    
    # Viscosity
    visc_lap = np.where((rlen > 0) & (rlen < h), 45.0 / (np.pi * h**6) * (h - rlen), 0.0)
    vel_diff = vel[:, None, :] - vel[None, :, :]  # vel[i] - vel[j]
    visc_matrix = mu * ((-vel_diff) / rho[None, :, None]) * visc_lap[:, :, None] * mass_j[:, :, None]
    f_visc = np.sum(visc_matrix, axis=1)
    
    # Gravity
    r2_grav = r2 + soft**2
    invr3 = 1.0 / (np.sqrt(r2_grav) * r2_grav)
    np.fill_diagonal(invr3, 0.0)
    f_grav = G * np.sum(mass_j[:, :, None] * (-diff) * invr3[:, :, None], axis=1)
    
    f = f_pres + f_visc + f_grav
    return f

# Compute potential energy of the system
def compute_potential_energy(pos, mass):
    diff = pos[:, None, :] - pos[None, :, :]
    r2 = np.sum(diff**2, axis=2) + soft**2
    r = np.sqrt(r2)
    mass_matrix = mass[:, None] * mass[None, :]
    pe = -G * np.sum(np.triu(mass_matrix / r, k=1))
    return pe

fig = plt.figure(figsize=(14,7))
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_proj_type('persp')  # Enable perspective projection
ax1.view_init(elev=20, azim=45)  # Set a nice viewing angle
ax2 = fig.add_subplot(122)
pos_plot = to_cpu(pos)
scat = ax1.scatter(pos_plot[:,0], pos_plot[:,1], pos_plot[:,2], s=8, c='cyan')
ax1.set_xlim(-1,1); ax1.set_ylim(-1,1); ax1.set_zlim(-1,1); ax1.set_facecolor('k')
ax1.set_box_aspect([1, 1, 1])  # Ensure equal aspect ratio for 3D axes
ax1.axis('off')

# Energy plots
ke_list = []
pe_list = []
te_list = []
line_ke, = ax2.plot([], [], 'r-', label='Kinetic Energy')
line_pe, = ax2.plot([], [], 'b-', label='Potential Energy')
line_te, = ax2.plot([], [], 'g-', label='Total Energy')
ax2.legend()
ax2.set_xlabel('Time Step')
ax2.set_ylabel('Energy')
# ax2.set_xlim(0, steps)  # Will be set dynamically in update
#ax2.set_ylim(-10000, 10000)  # Remove fixed ylim to allow autoscaling

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

# Animation update function
def update(frame):
    global pos, vel
    rho, P = compute_density_pressure(pos, mass)
    F = compute_forces(pos, vel, rho, P)
    vel += dt * (F / mass[:,None])
    pos += dt * vel

    ke = float(0.5 * np.sum(mass * np.sum(vel**2, axis=1)))
    pe = float(compute_potential_energy(pos, mass))
    te = ke + pe

    ke_list.append(ke)
    pe_list.append(pe)
    te_list.append(te)

    pos_plot = to_cpu(pos)
    scat._offsets3d = (pos_plot[:,0], pos_plot[:,1], pos_plot[:,2])
    line_ke.set_data(range(len(ke_list)), ke_list)
    line_pe.set_data(range(len(pe_list)), pe_list)
    line_te.set_data(range(len(te_list)), te_list)
    ax2.set_xlim(0, len(ke_list))
    ax2.relim()
    ax2.autoscale_view()
    return scat, line_ke, line_pe, line_te

ani = FuncAnimation(fig, update, frames=steps, interval=20, blit=False)
plt.show()