import numpy as np
from numba import njit, prange
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# ── Simulation parameters ────────────────────────────────────────────────────
N             = 5000        # Number of particles
G             = 1.0         # Gravitational constant (scaled for stability)
soft          = 0.05        # Softening parameter for gravity
h             = 0.05        # SPH smoothing length (= grid cell size)²
rho0          = 0.01        # Rest density for pressure calculation
k             = 50.0        # Pressure stiffness constant (higher → more incompressible)
mu            = 0.05        # Viscosity coefficient (higher → more damping)
dt_factor     = 1.0         # Time step factor for adaptive stepping (lower → more stable but slower)
epsilon       = 1e-12       # Small value to avoid division by zero
mass_min      = 0.01        # Minimum particle mass
mass_max      = 0.1         # Maximum particle mass
dotsize       = 10          # Size of dots in the plot

# ── Caching interval for PE only (display-only, doesn't affect physics)
PE_INTERVAL = 10   # recompute potential energy every N frames

# ── Initial conditions ───────────────────────────────────────────────────────
rng      = np.random.default_rng(42)
r_s      = 0.7 * rng.uniform(0, 1, N) ** (1/3)
cos_th   = rng.uniform(-1, 1, N)
theta    = np.arccos(cos_th)
phi      = rng.uniform(0, 2*np.pi, N)

pos  = np.column_stack([r_s * np.sin(theta) * np.cos(phi),
                        r_s * np.sin(theta) * np.sin(phi),
                        r_s * cos_th]).astype(np.float64)
vel  = rng.normal(scale=0.05, size=(N, 3)).astype(np.float64)
mass = rng.uniform(mass_min, mass_max, N).astype(np.float64)


# ── Grid builder ─────────────────────────────────────────────────────────────
@njit(fastmath=True, cache=True)
def build_grid(pos, h):
    """
    Counting-sort particles into a uniform grid with cell size h.
    Returns:
        sorted_ids  – particle indices ordered by cell
        cell_start  – prefix-sum array: particles in cell c are
                       sorted_ids[cell_start[c] : cell_start[c+1]]
        nx, ny, nz  – grid dimensions
        x0, y0, z0  – grid origin
    """
    N     = pos.shape[0]
    inv_h = 1.0 / h

    # Dynamic bounding box with one-cell padding
    xmin = ymin = zmin =  1e30
    xmax = ymax = zmax = -1e30
    for i in range(N):
        if pos[i, 0] < xmin: xmin = pos[i, 0]
        if pos[i, 1] < ymin: ymin = pos[i, 1]
        if pos[i, 2] < zmin: zmin = pos[i, 2]
        if pos[i, 0] > xmax: xmax = pos[i, 0]
        if pos[i, 1] > ymax: ymax = pos[i, 1]
        if pos[i, 2] > zmax: zmax = pos[i, 2]
    xmin -= h;  ymin -= h;  zmin -= h

    nx     = max(1, int((xmax - xmin) * inv_h) + 2)
    ny     = max(1, int((ymax - ymin) * inv_h) + 2)
    nz     = max(1, int((zmax - zmin) * inv_h) + 2)
    ncells = nx * ny * nz

    # Count particles per cell
    counts  = np.zeros(ncells + 1, dtype=np.int32)
    cid_buf = np.empty(N, dtype=np.int32)
    for i in range(N):
        ix  = min(int((pos[i, 0] - xmin) * inv_h), nx - 1)
        iy  = min(int((pos[i, 1] - ymin) * inv_h), ny - 1)
        iz  = min(int((pos[i, 2] - zmin) * inv_h), nz - 1)
        cid = ix * (ny * nz) + iy * nz + iz
        cid_buf[i]      = cid
        counts[cid + 1] += 1

    # Prefix sum → cell start offsets
    for c in range(1, ncells + 1):
        counts[c] += counts[c - 1]

    # Scatter particles into sorted order
    offsets    = counts[:ncells].copy()
    sorted_ids = np.empty(N, dtype=np.int32)
    for i in range(N):
        cid = cid_buf[i]
        sorted_ids[offsets[cid]] = i
        offsets[cid] += 1

    return sorted_ids, counts, nx, ny, nz, xmin, ymin, zmin


# ── SPH: density + pressure ──────────────────────────────────────────────────
# BEFORE: O(N²) — 25M pair checks (N=5000)
# AFTER:  O(N·k) — only 27 neighbour cells checked per particle
@njit(fastmath=True, parallel=True, cache=True)
def compute_density_pressure(pos, mass, h, rho0, k,
                              sids, cstart, nx, ny, nz, x0, y0, z0):
    N     = pos.shape[0]
    rho   = np.zeros(N)
    h2    = h * h
    inv_h = 1.0 / h
    poly6 = 315.0 / (64.0 * np.pi * h**9)
    nynz  = ny * nz           # precomputed to avoid multiplication in inner loop

    for i in prange(N):
        ix = min(max(int((pos[i, 0] - x0) * inv_h), 0), nx - 1)
        iy = min(max(int((pos[i, 1] - y0) * inv_h), 0), ny - 1)
        iz = min(max(int((pos[i, 2] - z0) * inv_h), 0), nz - 1)
        acc = 0.0

        for ox in range(-1, 2):
            jx = ix + ox
            if jx < 0 or jx >= nx: continue
            for oy in range(-1, 2):
                jy = iy + oy
                if jy < 0 or jy >= ny: continue
                for oz in range(-1, 2):
                    jz = iz + oz
                    if jz < 0 or jz >= nz: continue
                    cid = jx * nynz + jy * nz + jz
                    for idx in range(cstart[cid], cstart[cid + 1]):
                        j  = sids[idx]
                        dx = pos[i, 0] - pos[j, 0]
                        dy = pos[i, 1] - pos[j, 1]
                        dz = pos[i, 2] - pos[j, 2]
                        r2 = dx*dx + dy*dy + dz*dz
                        if r2 < h2:
                            q   = h2 - r2
                            acc += mass[j] * q * q * q
        rho[i] = acc * poly6

    P = k * np.maximum(rho - rho0, 0.0)
    return rho, P


# ── SPH: pressure + viscosity forces ────────────────────────────────────────
# Reuses the same grid built above → no second O(N) grid scan
@njit(fastmath=True, parallel=True, cache=True)
def compute_sph_forces(pos, vel, mass, rho, P, h, mu,
                       sids, cstart, nx, ny, nz, x0, y0, z0):
    N     = pos.shape[0]
    f     = np.zeros((N, 3))
    h2    = h * h
    inv_h = 1.0 / h
    pref  = -45.0 / (np.pi * h**6)
    vpref =  45.0 / (np.pi * h**6)
    nynz  = ny * nz

    for i in prange(N):
        ix  = min(max(int((pos[i, 0] - x0) * inv_h), 0), nx - 1)
        iy  = min(max(int((pos[i, 1] - y0) * inv_h), 0), ny - 1)
        iz  = min(max(int((pos[i, 2] - z0) * inv_h), 0), nz - 1)
        fx  = fy = fz = 0.0
        pri = P[i] / (rho[i] * rho[i])

        for ox in range(-1, 2):
            jx = ix + ox
            if jx < 0 or jx >= nx: continue
            for oy in range(-1, 2):
                jy = iy + oy
                if jy < 0 or jy >= ny: continue
                for oz in range(-1, 2):
                    jz = iz + oz
                    if jz < 0 or jz >= nz: continue
                    cid = jx * nynz + jy * nz + jz
                    for idx in range(cstart[cid], cstart[cid + 1]):
                        j = sids[idx]
                        if i == j: continue
                        dx = pos[i, 0] - pos[j, 0]
                        dy = pos[i, 1] - pos[j, 1]
                        dz = pos[i, 2] - pos[j, 2]
                        r2 = dx*dx + dy*dy + dz*dz
                        if r2 >= h2 or r2 < epsilon: continue
                        dist = np.sqrt(r2)
                        h_r  = h - dist

                        # Pressure gradient
                        ps  = (pri + P[j] / (rho[j]*rho[j])) * mass[j] * pref * h_r*h_r / dist
                        fx += dx * ps
                        fy += dy * ps
                        fz += dz * ps

                        # Viscosity
                        vs  = mu * (mass[j] / rho[j]) * vpref * h_r
                        fx -= (vel[i, 0] - vel[j, 0]) * vs
                        fy -= (vel[i, 1] - vel[j, 1]) * vs
                        fz -= (vel[i, 2] - vel[j, 2]) * vs

        f[i, 0] = fx;  f[i, 1] = fy;  f[i, 2] = fz
    return f


# ── Gravity (O(N²), computed every frame — feeds integration, must not be cached) ──
@njit(fastmath=True, parallel=True, cache=True)
def compute_gravity(pos, mass, G, soft):
    N     = pos.shape[0]
    f     = np.zeros((N, 3))
    soft2 = soft * soft
    for i in prange(N):
        fx = fy = fz = 0.0
        for j in range(N):
            if i == j: continue
            dx    = pos[i, 0] - pos[j, 0]
            dy    = pos[i, 1] - pos[j, 1]
            dz    = pos[i, 2] - pos[j, 2]
            r2    = dx*dx + dy*dy + dz*dz + soft2
            coeff = G * mass[j] / (r2 * np.sqrt(r2))
            fx   -= dx * coeff
            fy   -= dy * coeff
            fz   -= dz * coeff
        f[i, 0] = fx;  f[i, 1] = fy;  f[i, 2] = fz
    return f


# ── Potential energy (O(N²), cached every PE_INTERVAL frames) ────────────────
@njit(fastmath=True, parallel=True, cache=True)
def compute_potential_energy(pos, mass, G, soft):
    N     = pos.shape[0]
    soft2 = soft * soft
    total = 0.0
    for i in prange(N):
        sub = 0.0
        for j in range(i + 1, N):
            dx  = pos[i, 0] - pos[j, 0]
            dy  = pos[i, 1] - pos[j, 1]
            dz  = pos[i, 2] - pos[j, 2]
            sub += (mass[i] * mass[j]) / np.sqrt(dx*dx + dy*dy + dz*dz + soft2)
        total += sub
    return -G * total


# ── Symplectic Euler integration + KE in one pass ────────────────────────────
t = 0.0 
@njit(fastmath=True, cache=True)
def integrate(pos, vel, F, mass, t):
    global soft, epsilon, dt_factor
    accel = F / mass[:, np.newaxis]
    max_a2 = np.max(np.sum(accel**2, axis=1))
    dt = dt_factor * np.sqrt(soft / (np.sqrt(max_a2) + epsilon))
    t += dt
    vel  += accel * dt
    pos  += vel   * dt
    ke    = 0.5 * np.sum(mass * (vel[:, 0]**2 + vel[:, 1]**2 + vel[:, 2]**2))
    return t, dt, ke


# ── Matplotlib setup ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 7))
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_proj_type('persp')
ax1.view_init(elev=20, azim=45)
ax2 = fig.add_subplot(122)
ax3 = ax2.twinx()

scat = ax1.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                   s=np.sqrt(mass) * dotsize, c='cyan')
ax1.set_xlim(-1, 1);  ax1.set_ylim(-1, 1);  ax1.set_zlim(-1, 1)
ax1.set_facecolor('k');  ax1.set_box_aspect([1, 1, 1]);  ax1.axis('off')

ke_list, pe_list, te_list, temp_list = [], [], [], []
line_ke,   = ax2.plot([], [], 'r-',  label='Kinetic Energy')
line_pe,   = ax2.plot([], [], 'b-',  label='Potential Energy')
line_te,   = ax2.plot([], [], 'g-',  label='Total Energy')
line_temp, = ax3.plot([], [], 'm--', label='Avg Temperature')

lines = [line_ke, line_pe, line_te, line_temp]
ax2.legend(lines, [l.get_label() for l in lines], loc='upper left')
ax2.set_xlabel('Time');  ax2.set_ylabel('Energy')
ax3.set_ylabel('Temperature (Diagnostic)', color='m')
ax3.tick_params(axis='y', labelcolor='m')

energy_text = fig.text(0.5, 0.02, '', ha='center', va='bottom', fontsize=11,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

def on_scroll(event):
    scale = 1/1.15 if event.button == 'up' else 1.15
    for get, set_ in [(ax1.get_xlim3d, ax1.set_xlim3d),
                      (ax1.get_ylim3d, ax1.set_ylim3d),
                      (ax1.get_zlim3d, ax1.set_zlim3d)]:
        lo, hi  = get()
        mid     = (lo + hi) * 0.5
        half    = (hi - lo) * 0.5 * scale
        set_([mid - half, mid + half])
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('scroll_event', on_scroll)

# ── Simulation state ─────────────────────────────────────────────────────────
frame_count = 0
cached_pe   = 0.0
time_list = []
t_sim = 0.0

def update(frame):
    global pos, vel, frame_count, cached_pe, time_list, t_sim

    # Build spatial grid once per frame — shared by density AND force passes
    sids, cstart, nx, ny, nz, x0, y0, z0 = build_grid(pos, h)

    rho, P = compute_density_pressure(pos, mass, h, rho0, k,
                                      sids, cstart, nx, ny, nz, x0, y0, z0)
    f_sph  = compute_sph_forces(pos, vel, mass, rho, P, h, mu,
                                sids, cstart, nx, ny, nz, x0, y0, z0)

    # Gravity must be computed every frame — it directly drives integration
    f_grav = compute_gravity(pos, mass, G, soft)

    F  = f_sph + f_grav
    t_sim, dt, ke = integrate(pos, vel, F, mass, t_sim)

    # PE is display-only → safe to amortize without affecting physics
    if frame_count % PE_INTERVAL == 0:
        cached_pe = float(compute_potential_energy(pos, mass, G, soft))

    te       = ke + cached_pe
    avg_temp = float(np.mean(P / (rho + epsilon)))  # BUG FIX: was appended twice before

    ke_list.append(ke);       pe_list.append(cached_pe)
    te_list.append(te);       temp_list.append(avg_temp)
    time_list.append(t_sim)
    frame_count += 1

    # ── Update visuals ────────────────────────────────────────────────────────
    scat._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])

    line_ke.set_data(time_list, ke_list);   line_pe.set_data(time_list, pe_list)
    line_te.set_data(time_list, te_list);   line_temp.set_data(time_list, temp_list)

    if time_list.__len__() > 1:
        ax2.set_xlim(time_list[0], time_list[-1])
        
    ax2.relim()
    ax2.autoscale_view()
    ax3.relim()
    ax3.autoscale_view()

    ax3.relim();  ax3.autoscale_view()

    energy_text.set_text(
        f'dt: {dt:.9f} | KE: {ke:.4f} | PE: {cached_pe:.4f} | TE: {te:.4f} | Avg Temp: {avg_temp:.4f}'
    )
    return scat, line_ke, line_pe, line_te, line_temp


ani = FuncAnimation(fig, update, frames=24, interval=1, blit=False)
plt.show()