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
dt = 0.0002
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

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')
scat = ax.scatter(pos[:,0], pos[:,1], pos[:,2], s=8, c='cyan')
ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(-1,1); ax.set_facecolor('k')
ax.axis('off')

# Mouse wheel zoom/unzoom
def on_scroll(event):
    base_scale = 1.15
    if event.button == 'up':
        scale = 1.0/base_scale
    elif event.button == 'down':
        scale = base_scale
    else:
        return

    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    x_mid = (xlim[0] + xlim[1]) * 0.5
    y_mid = (ylim[0] + ylim[1]) * 0.5
    z_mid = (zlim[0] + zlim[1]) * 0.5

    x_half = (xlim[1] - xlim[0]) * 0.5 * scale
    y_half = (ylim[1] - ylim[0]) * 0.5 * scale
    z_half = (zlim[1] - zlim[0]) * 0.5 * scale

    ax.set_xlim3d([x_mid - x_half, x_mid + x_half])
    ax.set_ylim3d([y_mid - y_half, y_mid + y_half])
    ax.set_zlim3d([z_mid - z_half, z_mid + z_half])

    fig.canvas.draw_idle()

fig.canvas.mpl_connect('scroll_event', on_scroll)

def update(frame):
    global pos, vel
    rho, P = compute_density_pressure(pos, mass)
    F = compute_forces(pos, vel, rho, P)
    vel += dt * (F / mass[:,None])
    pos += dt * vel
    scat._offsets3d = (pos[:,0], pos[:,1], pos[:,2])
    return scat,

ani = FuncAnimation(fig, update, frames=steps, interval=20, blit=False)
plt.show()