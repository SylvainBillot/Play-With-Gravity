import numpy as np
from numba import njit, prange
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm   # colormap

# ── Simulation parameters ────────────────────────────────────────────────────
R = 0.7 # base radius or side
L = 20 #If cube num of segment on a side
N = L**3  # Number of particles
G = 1.0  # Gravitational constant (scaled for stability)
soft = 0.05  # Softening parameter for gravity
h = 0.05  # SPH smoothing length (= grid cell size)²
rho0 = 0.01  # Rest density for pressure calculation
k = 20.0  # Pressure stiffness constant (higher → more incompressible)
mu = 0.01  # Viscosity coefficient (higher → more damping)
H0 = 1e2   # Hubble constant en s⁻¹
IE = 0.05 # Initial Speed factor 
Omega_m = 0.3          # densité matière
Omega_L = 0.7          # densité énergie sombre
a0 = 1.0               # facteur d’échelle initial (au temps t=0)
dt_factor = (
    1.0  # Time step factor for adaptive stepping (lower → more stable but slower)
)
epsilon = 1e-24  # Small value to avoid division by zero
mass_min = 0.01  # Minimum particle mass
mass_max = 0.1  # Maximum particle mass
dotsize = 20  # Size of dots in the plot

rho_min, rho_max = np.inf, -np.inf   # initialisés à des valeurs extrêmes
cmap = plt.colormaps.get_cmap('autumn')       # vous pouvez choisir n’importe quel colormap
norm = None                          # sera créé après la première frame

# Random from reponse from univer 
rng = np.random.default_rng(42)

# ── Initial conditions ───────────────────────────────────────────────────────
def initializeSphere():
    r_s = R * rng.uniform(0, 1, N) ** (1 / 3)
    cos_th = rng.uniform(-1, 1, N)
    theta = np.arccos(cos_th)
    phi = rng.uniform(0, 2 * np.pi, N)

    return  np.column_stack(
        [r_s * np.sin(theta) * np.cos(phi), r_s * np.sin(theta) * np.sin(phi), r_s * cos_th]
    ).astype(np.float64)

def initializeCubeRandom():
    return np.column_stack(
        [rng.uniform(-R/2.0,R/2.0, N), rng.uniform(-R/2.0,R/2.0, N), rng.uniform(-R/2.0,R/2.0, N)]
    ).astype(np.float64)

def initializeCube():  # cube centered at origin with side length 2R
    x = np.linspace(-R, R, L)
    y = np.linspace(-R, R, L)
    z = np.linspace(-R, R, L)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    return np.column_stack([X, Y, Z]).astype(np.float64)

def initializeParticles():
    # return initializeSphere()
    # return initializeCubeRandom()
    return initializeCube()

#pos = initializeSphere()
pos = initializeParticles()
vel = rng.normal(scale=IE, size=(N, 3)).astype(np.float64)
mass = rng.uniform(mass_min, mass_max, N).astype(np.float64)


# ── Simulation functions ───────────────────────────────────────────────────────
@njit
def calculate_forces(r, v, m, rho, h, soft, G, k, mu):
    N = len(r)
    F = np.zeros_like(r)
    P = np.zeros_like(r)
    T = np.zeros_like(r)
    for i in prange(N):
        for j in prange(i + 1, N):
            dr = r[i] - r[j]
            dist = np.linalg.norm(dr)
            if dist > 0:
                f = G * m[i] * m[j] / (dist ** 2 + soft ** 2)
                F[i] += f * dr / dist
                F[j] -= f * dr / dist

                # Pressure force
                rho_i = rho[i]
                rho_j

@njit(cache=True)
def cosmology(a, H0, Omega_m, Omega_L):
    """
    Retourne le nouveau facteur d’échelle a(t+dt) et le Hubble H(t)
    en supposant une expansion de Friedmann‑Lemaître‑Robertson‑Walker
    (matière + énergie sombre, k=0).
    """
    # équation de Friedmann simplifiée : H^2 = H0^2 * (Omega_m/a^3 + Omega_L)
    H = H0 * np.sqrt(Omega_m / (a**3) + Omega_L)
    return H

# Builds a uniform grid for neighbor search. O(N) complexity, shared by density and force passes.
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
    N = pos.shape[0]
    inv_h = 1.0 / h

    # Dynamic bounding box with one-cell padding
    xmin = ymin = zmin = 1e30
    xmax = ymax = zmax = -1e30
    for i in range(N):
        if pos[i, 0] < xmin:
            xmin = pos[i, 0]
        if pos[i, 1] < ymin:
            ymin = pos[i, 1]
        if pos[i, 2] < zmin:
            zmin = pos[i, 2]
        if pos[i, 0] > xmax:
            xmax = pos[i, 0]
        if pos[i, 1] > ymax:
            ymax = pos[i, 1]
        if pos[i, 2] > zmax:
            zmax = pos[i, 2]
    xmin -= h
    ymin -= h
    zmin -= h

    nx = max(1, int((xmax - xmin) * inv_h) + 2)
    ny = max(1, int((ymax - ymin) * inv_h) + 2)
    nz = max(1, int((zmax - zmin) * inv_h) + 2)
    ncells = nx * ny * nz

    # Count particles per cell
    counts = np.zeros(ncells + 1, dtype=np.int32)
    cid_buf = np.empty(N, dtype=np.int32)
    for i in range(N):
        ix = min(int((pos[i, 0] - xmin) * inv_h), nx - 1)
        iy = min(int((pos[i, 1] - ymin) * inv_h), ny - 1)
        iz = min(int((pos[i, 2] - zmin) * inv_h), nz - 1)
        cid = ix * (ny * nz) + iy * nz + iz
        cid_buf[i] = cid
        counts[cid + 1] += 1

    # Prefix sum → cell start offsets
    for c in range(1, ncells + 1):
        counts[c] += counts[c - 1]

    # Scatter particles into sorted order
    offsets = counts[:ncells].copy()
    sorted_ids = np.empty(N, dtype=np.int32)
    for i in range(N):
        cid = cid_buf[i]
        sorted_ids[offsets[cid]] = i
        offsets[cid] += 1

    return sorted_ids, counts, nx, ny, nz, xmin, ymin, zmin


# ── SPH density and pressure (O(N) with grid) ───────────────────────────────
@njit(fastmath=True, parallel=True, cache=True)
def compute_density_pressure(
    pos, mass, h, rho0, k, sids, cstart, nx, ny, nz, x0, y0, z0
):
    N = pos.shape[0]
    rho = np.zeros(N)
    h2 = h * h
    inv_h = 1.0 / h
    poly6 = 315.0 / (64.0 * np.pi * h**9)
    nynz = ny * nz  # precomputed to avoid multiplication in inner loop

    for i in prange(N):
        ix = min(max(int((pos[i, 0] - x0) * inv_h), 0), nx - 1)
        iy = min(max(int((pos[i, 1] - y0) * inv_h), 0), ny - 1)
        iz = min(max(int((pos[i, 2] - z0) * inv_h), 0), nz - 1)
        acc = 0.0

        for ox in range(-1, 2):
            jx = ix + ox
            if jx < 0 or jx >= nx:
                continue
            for oy in range(-1, 2):
                jy = iy + oy
                if jy < 0 or jy >= ny:
                    continue
                for oz in range(-1, 2):
                    jz = iz + oz
                    if jz < 0 or jz >= nz:
                        continue
                    cid = jx * nynz + jy * nz + jz
                    for idx in range(cstart[cid], cstart[cid + 1]):
                        j = sids[idx]
                        dx = pos[i, 0] - pos[j, 0]
                        dy = pos[i, 1] - pos[j, 1]
                        dz = pos[i, 2] - pos[j, 2]
                        r2 = dx * dx + dy * dy + dz * dz
                        if r2 < h2:
                            q = h2 - r2
                            acc += mass[j] * q * q * q
        rho[i] = acc * poly6

    P = k * np.maximum(rho - rho0, 0.0)
    return rho, P


# ── SPH forces (O(N) with grid) ─────────────────────────────────────────────
@njit(fastmath=True, parallel=True, cache=True)
def compute_sph_forces(
    pos, vel, mass, rho, P, h, mu, sids, cstart, nx, ny, nz, x0, y0, z0
):
    N = pos.shape[0]
    f = np.zeros((N, 3))
    h2 = h * h
    inv_h = 1.0 / h
    pref = -45.0 / (np.pi * h**6)
    vpref = 45.0 / (np.pi * h**6)
    nynz = ny * nz

    for i in prange(N):
        ix = min(max(int((pos[i, 0] - x0) * inv_h), 0), nx - 1)
        iy = min(max(int((pos[i, 1] - y0) * inv_h), 0), ny - 1)
        iz = min(max(int((pos[i, 2] - z0) * inv_h), 0), nz - 1)
        fx = fy = fz = 0.0
        pri = P[i] / (rho[i] * rho[i])

        for ox in range(-1, 2):
            jx = ix + ox
            if jx < 0 or jx >= nx:
                continue
            for oy in range(-1, 2):
                jy = iy + oy
                if jy < 0 or jy >= ny:
                    continue
                for oz in range(-1, 2):
                    jz = iz + oz
                    if jz < 0 or jz >= nz:
                        continue
                    cid = jx * nynz + jy * nz + jz
                    for idx in range(cstart[cid], cstart[cid + 1]):
                        j = sids[idx]
                        if i == j:
                            continue
                        dx = pos[i, 0] - pos[j, 0]
                        dy = pos[i, 1] - pos[j, 1]
                        dz = pos[i, 2] - pos[j, 2]
                        r2 = dx * dx + dy * dy + dz * dz
                        if r2 >= h2 or r2 < epsilon:
                            continue
                        dist = np.sqrt(r2)
                        h_r = h - dist
                        
                        # Pressure gradient
                        ps = (
                            (pri + P[j] / (rho[j] * rho[j]))
                            * mass[j]
                            * pref
                            * h_r
                            * h_r
                            / dist
                        )
                        fx += dx * ps
                        fy += dy * ps
                        fz += dz * ps
                        
                        # Viscosity
                        vs = mu * (mass[j] / rho[j]) * vpref * h_r
                        fx -= (vel[i, 0] - vel[j, 0]) * vs
                        fy -= (vel[i, 1] - vel[j, 1]) * vs
                        fz -= (vel[i, 2] - vel[j, 2]) * vs

        f[i, 0] = fx
        f[i, 1] = fy
        f[i, 2] = fz
    return f


# Global constant for Barnes-Hut: higher = faster/less accurate, lower = slower/more accurate
THETA = 1.0 

@njit(fastmath=True)
def compute_gravity_barnes_hut(pos, mass, G, soft, theta=THETA):
    N = pos.shape[0]
    forces = np.zeros((N, 3))
    soft2 = soft * soft
    
    # 1. Define the bounding box of the system
    xmin, ymin, zmin = np.min(pos[:,0]), np.min(pos[:,1]), np.min(pos[:,2])
    xmax, ymax, zmax = np.max(pos[:,0]), np.max(pos[:,1]), np.max(pos[:,2])
    size = max(xmax - xmin, ymax - ymin, zmax - zmin) * 1.01
    mid_x, mid_y, mid_z = (xmin + xmax)/2, (ymin + ymax)/2, (zmin + zmax)/2

    # 2. Build the Octree
    # We use arrays to represent the tree for Numba compatibility
    # max_nodes: N particles plus internal nodes (usually < 2N)
    max_nodes = 2 * N 
    node_mass = np.zeros(max_nodes)
    node_com = np.zeros((max_nodes, 3))
    node_children = -np.ones((max_nodes, 8), dtype=np.int32)
    node_size = np.zeros(max_nodes)
    
    next_node = 1
    node_size[0] = size
    node_com[0] = np.array([mid_x, mid_y, mid_z])

    # Insert particles into Octree
    for i in range(N):
        curr = 0 # Start at root
        p_pos = pos[i]
        p_mass = mass[i]
        
        # Traverse down to find a leaf
        s = size
        cx, cy, cz = mid_x, mid_y, mid_z
        
        while True:
            # Update node Center of Mass and total mass as we go down
            new_total_mass = node_mass[curr] + p_mass
            node_com[curr] = (node_com[curr] * node_mass[curr] + p_pos * p_mass) / new_total_mass
            node_mass[curr] = new_total_mass
            
            # Determine which octant
            s /= 2
            octant = 0
            if p_pos[0] > cx: octant += 1; cx += s/2
            else: cx -= s/2
            if p_pos[1] > cy: octant += 2; cy += s/2
            else: cy -= s/2
            if p_pos[2] > cz: octant += 4; cz += s/2
            else: cz -= s/2
            
            child = node_children[curr, octant]
            if child == -1:
                # Found empty spot, insert leaf
                node_children[curr, octant] = next_node
                node_mass[next_node] = p_mass
                node_com[next_node] = p_pos
                node_size[next_node] = s
                next_node += 1
                break
            else:
                # Move to child node
                curr = child

    # 3. Calculate Forces using the tree
    total = 0.0
    for i in prange(N):
        p_pos = pos[i]
        # Stack for non-recursive traversal
        stack = [0]
        fx = fy = fz = 0.0
        p_pe = 0.0
        while len(stack) > 0:
            curr = stack.pop()
            
            # Distance from particle to node center of mass
            dx = node_com[curr, 0] - p_pos[0]
            dy = node_com[curr, 1] - p_pos[1]
            dz = node_com[curr, 2] - p_pos[2]
            r2 = dx*dx + dy*dy + dz*dz + soft2
            r = np.sqrt(r2)
            
            # Barnes-Hut Criterion: Is the node far enough?
            if node_size[curr] / r < theta or np.all(node_children[curr] == -1):
                # Approximation: Treat node as a single point mass
                if r > epsilon:
                    inv_r3 = 1.0 / (r * r * r)
                    p_pe += (p_mass * node_mass[curr]) / r
                    coeff = G * node_mass[curr] * inv_r3
                    fx += dx * coeff
                    fy += dy * coeff
                    fz += dz * coeff
            else:
                # Too close: Open the node and check children
                for octant in range(8):
                    child = node_children[curr, octant]
                    if child != -1:
                        stack.append(child)
        
        forces[i, 0] = fx
        forces[i, 1] = fy
        forces[i, 2] = fz
        total += p_pe
        
    return -0.5 * G * total, forces # PE calculation omitted for tree for simplicity


# ── Intégration avec facteur d’expansion (O(N)) ───────────────────────
t = 0.0
a = a0                     # facteur d’échelle courant
@njit(fastmath=True, cache=True)
def integrate(pos, vel, F, mass, t, a):
    """
    pos, vel : coordonnées comobiles
    F       : forces physiques (gravité + SPH) en coordonnées comobiles
    a       : facteur d’échelle actuel
    Retourne (t, a, dt, ke)
    """
    # 1. Calcul du Hubble actuel
    H = cosmology(a, H0, Omega_m, Omega_L)

    # 2. Accélération comobile (forces / masse) + terme de décélération cosmologique
    accel = F / mass[:, np.newaxis] - H * vel

    # 3. Pas de temps adaptatif (identique à l’ancien code)
    max_a2 = np.max(np.sum(accel**2, axis=1))
    dt = dt_factor * np.sqrt(soft / (np.sqrt(max_a2) + epsilon))

    # 4. Mise à jour des vitesses et positions comobiles
    vel += accel * dt
    pos += vel * dt

    # 5. Mise à jour du facteur d’échelle (Euler simple)
    a += H * a * dt          # da/dt = H·a

    # 6. Avance du temps physique
    t += dt

    # 7. Énergie cinétique (reste en coordonnées comobiles)
    ke = 0.5 * np.sum(mass * (vel[:, 0]**2 + vel[:, 1]**2 + vel[:, 2]**2))
    return t, a, dt, ke


# ── Matplotlib setup ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 7))
ax1 = fig.add_subplot(121, projection="3d")
ax1.set_proj_type("persp")
ax1.view_init(elev=20, azim=45)
ax2 = fig.add_subplot(122)
ax3 = ax2.twinx()

scat = ax1.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=np.sqrt(mass) * dotsize)
ax1.set_xlim(-1, 1)
ax1.set_ylim(-1, 1)
ax1.set_zlim(-1, 1)
ax1.set_facecolor("k")
ax1.set_box_aspect([1, 1, 1])
ax1.axis("off")

ke_list, pe_list, te_list, temp_list = [], [], [], []
(line_ke,) = ax2.plot([], [], "r-", label="Kinetic Energy")
(line_pe,) = ax2.plot([], [], "b-", label="Potential Energy")
(line_te,) = ax2.plot([], [], "g-", label="Total Energy")
(line_temp,) = ax3.plot([], [], "m--", label="Avg Temperature")

lines = [line_ke, line_pe, line_te, line_temp]
ax2.legend(lines, [l.get_label() for l in lines], loc="upper left")
ax2.set_xlabel("Time")
ax2.set_ylabel("Energy")
ax3.set_ylabel("Temperature (Diagnostic)", color="m")
ax3.tick_params(axis="y", labelcolor="m")

energy_text = fig.text(
    0.5,
    0.02,
    "",
    ha="center",
    va="bottom",
    fontsize=11,
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
)

@njit(cache=True)
def normalize_density(rho, rmin, rmax):
    """Renvoie un tableau de valeurs entre 0 et 1."""
    # évite la division par zéro si rmax == rmin
    if rmax <= rmin:
        return np.zeros_like(rho)
    return (rho - rmin) / (rmax - rmin)

def on_scroll(event):
    scale = 1 / 1.15 if event.button == "up" else 1.15
    for get, set_ in [
        (ax1.get_xlim3d, ax1.set_xlim3d),
        (ax1.get_ylim3d, ax1.set_ylim3d),
        (ax1.get_zlim3d, ax1.set_zlim3d),
    ]:
        lo, hi = get()
        mid = (lo + hi) * 0.5
        half = (hi - lo) * 0.5 * scale
        set_([mid - half, mid + half])
    fig.canvas.draw_idle()

fig.canvas.mpl_connect("scroll_event", on_scroll)

# ── Simulation state ─────────────────────────────────────────────────────────
pe = 0.0
time_list = []
t_sim = 0.0
a_factor = a0


def update(frame):
    global pos, vel, pe, time_list, t_sim, a_factor
    global rho_min, rho_max, norm

    # Build spatial grid once per frame — shared by density AND force passes
    sids, cstart, nx, ny, nz, x0, y0, z0 = build_grid(pos, h)

    rho, P = compute_density_pressure(
        pos, mass, h, rho0, k, sids, cstart, nx, ny, nz, x0, y0, z0
    )
    f_sph = compute_sph_forces(
        pos, vel, mass, rho, P, h, mu, sids, cstart, nx, ny, nz, x0, y0, z0
    )

    # ── mise à jour du min / max de densité (dynamique) ─────────────────
    cur_min = np.min(rho)
    cur_max = np.max(rho)
    rho_min = min(rho_min, cur_min)
    rho_max = max(rho_max, cur_max)

    # normalisation (0‑1) pour la frame courante
    rho_norm = normalize_density(rho, rho_min, rho_max)

    # création du ScalarMappable (une seule fois, puis on le met à jour)
    if norm is None:
        norm = plt.Normalize(vmin=0.0, vmax=1.0)   # 0‑1 après normalisation
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    else:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)

    colors = sm.to_rgba(rho_norm)   # tableau (N,4) RGBA

    # Gravity must be computed every frame — it directly drives integration
    pe, f_grav = compute_gravity_barnes_hut(pos, mass, G, soft)

    F = f_sph + f_grav
    t_sim, a_factor, dt, ke = integrate(pos, vel, F, mass, t_sim, a_factor)

    te = ke + pe
    avg_temp = float(np.mean(P / (rho + epsilon)))  # BUG FIX: was appended twice before

    ke_list.append(ke)
    pe_list.append(pe)
    te_list.append(te)
    temp_list.append(avg_temp)
    time_list.append(t_sim)

    # ── mise à jour du nuage de points avec nouvelles couleurs ───────
    scat._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2]) 
    scat.set_facecolor(colors)          # <-- couleur dépendante de rho
    scat.set_edgecolor('none')          # évite le contour noir
    scat.set_sizes(np.sqrt(mass) * dotsize)          # <-- garde la même taille qu’au départ

    line_ke.set_data(time_list, ke_list)
    line_pe.set_data(time_list, pe_list)
    line_te.set_data(time_list, te_list)
    line_temp.set_data(time_list, temp_list)

    if time_list.__len__() > 1:
        ax2.set_xlim(time_list[0], time_list[-1])

    ax2.relim()
    ax2.autoscale_view()
    ax3.relim()
    ax3.autoscale_view()

    ax3.relim()
    ax3.autoscale_view()

    energy_text.set_text(
        f"dt:{dt:.2e} "
        f"KE:{ke:.2e} PE:{pe:.2e} TE:{te:.2e} Temp:{avg_temp:.2e}"
    )
    return scat, line_ke, line_pe, line_te, line_temp


ani = FuncAnimation(fig, update, frames=24, interval=1, blit=False)
plt.show()
