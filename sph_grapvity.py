import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

#!/usr/bin/env python3

# SPH + self-gravity 3D particles
G = 1.0
N = 250
h = 0.2
rho0 = 1.0
k = 20.0
mu = 0.05
dt = 0.002
steps = 1500
soft = 0.05


rng = np.random.default_rng(42)
pos = rng.uniform(-1.0, 1.0, (N, 3)) * 0.7
vel = rng.normal(scale=0.05, size=(N, 3))
mass = np.ones(N) * 0.05

def poly6(r2, h):
    hr2 = h*h - r2
    return np.where(r2 < h*h, 315.0 / (64*np.pi*h**9) * hr2**3, 0.0)

def spiky_grad(r, rlen, h):
    mask = (rlen > 1e-12) & (rlen < h)
    coef = -45.0 / (np.pi*h**6) * (h - rlen[mask])**2
    out = np.zeros_like(r)
    out[mask] = coef[:, None] * r[mask] / rlen[mask, None]
    return out

def viscosity_lap(rlen, h):
    return np.where((rlen>0) & (rlen<h),
                    45.0 / (np.pi*h**6) * (h - rlen),
                    0.0)

def compute_density_pressure(pos, mass):
    rho = np.zeros(N)
    for i in range(N):
        rij = pos - pos[i]
        r2 = np.sum(rij*rij, axis=1)
        rho[i] = np.sum(mass * poly6(r2, h))
    P = k * np.maximum(rho - rho0, 0.0)
    return rho, P

def compute_forces(pos, vel, rho, P):
    f = np.zeros((N,3))
    # SPH pressure + viscosity
    for i in range(N):
        rij = pos - pos[i]
        rlen = np.linalg.norm(rij, axis=1)
        gradW = spiky_grad(rij, rlen, h)
        pres = -np.sum((P[i]/rho[i]**2 + P/rho**2)[:,None] * mass[:,None] * gradW, axis=0)
        visc = mu * np.sum(((vel - vel[i]) / rho[:,None]) * viscosity_lap(rlen, h)[:,None] * mass[:,None], axis=0)
        f[i] += pres + visc
    # gravity
    for i in range(N):
        rij = pos - pos[i]
        r2 = np.sum(rij*rij, axis=1) + soft**2
        invr3 = 1.0 / (np.sqrt(r2)*r2)
        invr3[i] = 0.0
        f[i] += G * np.sum(mass[:,None] * rij * invr3[:,None], axis=0)
    return f

def compute_potential_energy(pos, mass):
    pe = 0.0
    for i in range(N):
        for j in range(i+1, N):
            rij = pos[i] - pos[j]
            r = np.sqrt(np.sum(rij**2) + soft**2)
            pe -= G * mass[i] * mass[j] / r
    return pe

fig = plt.figure(figsize=(14,7))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)
scat = ax1.scatter(pos[:,0], pos[:,1], pos[:,2], s=8, c='cyan')
ax1.set_xlim(-1,1); ax1.set_ylim(-1,1); ax1.set_zlim(-1,1); ax1.set_facecolor('k')
ax1.axis('off')

# Energy plots
ke_list = []
pe_list = []
line_ke, = ax2.plot([], [], 'r-', label='Kinetic Energy')
line_pe, = ax2.plot([], [], 'b-', label='Potential Energy')
ax2.legend()
ax2.set_xlabel('Time Step')
ax2.set_ylabel('Energy')
ax2.set_xlim(0, steps)
# ax2.set_ylim(-50, 50)  # Remove fixed ylim to allow autoscaling

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

def update(frame):
    global pos, vel
    rho, P = compute_density_pressure(pos, mass)
    F = compute_forces(pos, vel, rho, P)
    vel += dt * (F / mass[:,None])
    pos += dt * vel
    ke = 0.5 * np.sum(mass * np.sum(vel**2, axis=1))
    pe = compute_potential_energy(pos, mass)
    ke_list.append(ke)
    pe_list.append(pe)
    scat._offsets3d = (pos[:,0], pos[:,1], pos[:,2])
    line_ke.set_data(range(len(ke_list)), ke_list)
    line_pe.set_data(range(len(pe_list)), pe_list)
    ax2.relim()
    ax2.autoscale_view()
    return scat, line_ke, line_pe

ani = FuncAnimation(fig, update, frames=steps, interval=20, blit=False)
plt.show()