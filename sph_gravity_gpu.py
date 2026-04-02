import cupy as cp
import numpy as np  # Conservé pour Numba et Matplotlib
from numba import njit, prange
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm

# ── Simulation parameters ────────────────────────────────────────────────────
R = 0.7 
L = 20 
N = L**3  
G = 1.0  
soft = 0.05  
h = 0.05  
rho0 = 0.01  
k = 20.0  
mu = 0.01  
H0 = 1e2   
IE = 0.05 
Omega_m = 0.3          
Omega_L = 0.7          
a0 = 1.0               
dt_factor = 1.0  
epsilon = 1e-24  
mass_min = 0.01  
mass_max = 0.1  
dotsize = 20  

rho_min, rho_max = np.inf, -np.inf   
cmap = plt.colormaps.get_cmap('autumn')       
norm = None                          

# ── Random setup avec CuPy ───────────────────────────────────────────────────
# Attention : CuPy n'utilise pas tout à fait la même API 'default_rng' que NumPy
cp.random.seed(42)

# ── Initial conditions (Version CuPy) ────────────────────────────────────────
def initializeSphere():
    r_s = R * cp.random.uniform(0, 1, N) ** (1 / 3)
    cos_th = cp.random.uniform(-1, 1, N)
    theta = cp.arccos(cos_th)
    phi = cp.random.uniform(0, 2 * np.pi, N)

    return cp.column_stack(
        [r_s * cp.sin(theta) * cp.cos(phi), r_s * cp.sin(theta) * cp.sin(phi), r_s * cos_th]
    ).astype(cp.float64)

def initializeCubeRandom():
    return cp.random.uniform(-R/2.0, R/2.0, (N, 3)).astype(cp.float64)

def initializeCube():
    x = cp.linspace(-R, R, L)
    y = cp.linspace(-R, R, L)
    z = cp.linspace(-R, R, L)
    X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')
    return cp.column_stack([X.flatten(), Y.flatten(), Z.flatten()]).astype(cp.float64)

def initializeParticles():
    return initializeCube()

# Initialisation des tableaux sur le GPU
pos = initializeParticles()
# FIX: Utilisation de cp.random.normal au lieu de rng.normal
vel = cp.random.normal(loc=0.0, scale=IE, size=(N, 3)).astype(cp.float64)
mass = cp.random.uniform(mass_min, mass_max, N).astype(cp.float64)

# ── Numba Functions (Restent en NumPy) ───────────────────────────────────────
# Note : Numba ne peut pas lire directement les objets CuPy. 
# On passera des versions .get() dans la boucle update.

@njit(fastmath=True, cache=True)
def build_grid(pos, h):
    """
    Trie les particules dans une grille uniforme avec Numba.
    """
    N = pos.shape[0]
    inv_h = 1.0 / h

    # 1. Calcul des limites de la grille (Bounding Box)
    xmin = ymin = zmin = 1e30
    xmax = ymax = zmax = -1e30
    for i in range(N):
        if pos[i, 0] < xmin: xmin = pos[i, 0]
        if pos[i, 1] < ymin: ymin = pos[i, 1]
        if pos[i, 2] < zmin: zmin = pos[i, 2]
        if pos[i, 0] > xmax: xmax = pos[i, 0]
        if pos[i, 1] > ymax: ymax = pos[i, 1]
        if pos[i, 2] > zmax: zmax = pos[i, 2]
    
    xmin -= h
    ymin -= h
    zmin -= h

    nx = max(1, int((xmax - xmin) * inv_h) + 2)
    ny = max(1, int((ymax - ymin) * inv_h) + 2)
    nz = max(1, int((zmax - zmin) * inv_h) + 2)
    ncells = nx * ny * nz

    # 2. Compter les particules par cellule
    counts = np.zeros(ncells + 1, dtype=np.int32)
    cid_buf = np.empty(N, dtype=np.int32)
    
    for i in range(N):
        ix = min(int((pos[i, 0] - xmin) * inv_h), nx - 1)
        iy = min(int((pos[i, 1] - ymin) * inv_h), ny - 1)
        iz = min(int((pos[i, 2] - zmin) * inv_h), nz - 1)
        cid = ix * (ny * nz) + iy * nz + iz
        cid_buf[i] = cid
        counts[cid + 1] += 1

    # 3. Somme préfixe (Prefix sum) pour transformer les comptes en offsets
    for c in range(1, ncells + 1):
        counts[c] += counts[c - 1]

    # 4. Remplissage du tableau sorted_ids
    # On crée une copie des offsets pour savoir où écrire chaque particule
    offsets = counts[:ncells].copy()
    sorted_ids = np.empty(N, dtype=np.int32)
    
    for i in range(N):
        cid = cid_buf[i]
        sorted_ids[offsets[cid]] = i
        offsets[cid] += 1

    return sorted_ids, counts, nx, ny, nz, xmin, ymin, zmin

# ── Intégration (Version CuPy pour éviter les transferts inutiles) ───────────
def cosmology_cp(a, H0, Omega_m, Omega_L):
    H = H0 * cp.sqrt(Omega_m / (a**3) + Omega_L)
    return H

def integrate_gpu(pos, vel, F, mass, t, a):
    H = cosmology_cp(a, H0, Omega_m, Omega_L)
    accel = F / mass[:, cp.newaxis] - H * vel
    
    max_a2 = cp.max(cp.sum(accel**2, axis=1))
    dt = dt_factor * cp.sqrt(soft / (cp.sqrt(max_a2) + epsilon))

    vel += accel * dt
    pos += vel * dt
    a += H * a * dt
    t += dt
    ke = 0.5 * cp.sum(mass * cp.sum(vel**2, axis=1))
    return t, float(a), float(dt), float(ke)

# ── SPH density and pressure (Numba CPU) ───────────────────────────────
@njit(fastmath=True, parallel=True, cache=True)
def compute_density_pressure(
    pos, mass, h, rho0, k, sids, cstart, nx, ny, nz, x0, y0, z0
):
    N = pos.shape[0]
    rho = np.zeros(N)
    h2 = h * h
    poly6 = 315.0 / (64.0 * np.pi * h**9)
    nynz = ny * nz

    for i in prange(N):
        ix = min(max(int((pos[i, 0] - x0) * (1.0/h)), 0), nx - 1)
        iy = min(max(int((pos[i, 1] - y0) * (1.0/h)), 0), ny - 1)
        iz = min(max(int((pos[i, 2] - z0) * (1.0/h)), 0), nz - 1)
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

# ── SPH forces (Numba CPU) ─────────────────────────────────────────────
@njit(fastmath=True, parallel=True, cache=True)
def compute_sph_forces(
    pos, vel, mass, rho, P, h, mu, sids, cstart, nx, ny, nz, x0, y0, z0
):
    N = pos.shape[0]
    f = np.zeros((N, 3))
    h2 = h * h
    pref = -45.0 / (np.pi * h**6)
    vpref = 45.0 / (np.pi * h**6)
    nynz = ny * nz

    for i in prange(N):
        ix = min(max(int((pos[i, 0] - x0) * (1.0/h)), 0), nx - 1)
        iy = min(max(int((pos[i, 1] - y0) * (1.0/h)), 0), ny - 1)
        iz = min(max(int((pos[i, 2] - z0) * (1.0/h)), 0), nz - 1)
        fx = fy = fz = 0.0
        
        # Sécurité division par zéro
        rho_i = rho[i] if rho[i] > epsilon else epsilon
        pri = P[i] / (rho_i * rho_i)

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
                        r2 = dx * dx + dy * dy + dz * dz
                        
                        if r2 >= h2 or r2 < epsilon: continue
                        
                        dist = np.sqrt(r2)
                        h_r = h - dist
                        rho_j = rho[j] if rho[j] > epsilon else epsilon
                        
                        # Gradient de pression
                        ps = (pri + P[j] / (rho_j * rho_j)) * mass[j] * pref * h_r**2 / dist
                        fx += dx * ps
                        fy += dy * ps
                        fz += dz * ps
                        
                        # Viscosité
                        vs = mu * (mass[j] / rho_j) * vpref * h_r
                        fx -= (vel[i, 0] - vel[j, 0]) * vs
                        fy -= (vel[i, 1] - vel[j, 1]) * vs
                        fz -= (vel[i, 2] - vel[j, 2]) * vs

        f[i, 0], f[i, 1], f[i, 2] = fx, fy, fz
    return f

# ── Gravity forces (Numba CPU) ───────────────────────────────────────────
@njit(fastmath=True, parallel=True, cache=True)
def compute_gravity(pos, mass, G, soft):
    N = pos.shape[0]
    f = np.zeros((N, 3))
    soft2 = soft * soft
    total_pe = 0.0
    
    for i in prange(N):
        fx = fy = fz = 0.0
        sub_pe = 0.0
        for j in range(N):
            if i == j: continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            r2 = dx * dx + dy * dy + dz * dz + soft2
            dist = np.sqrt(r2)
            coeff = G * mass[j] / (r2 * dist)
            fx -= dx * coeff
            fy -= dy * coeff
            fz -= dz * coeff
            sub_pe += (mass[i] * mass[j]) / dist
        
        f[i, 0], f[i, 1], f[i, 2] = fx, fy, fz
        total_pe += sub_pe
        
    return -0.5 * G * total_pe, f


# ── Matplotlib setup ─────────────────────────────────────────────────────────
# ── Matplotlib setup (Nécessite NumPy) ───────────────────────────────────────
fig = plt.figure(figsize=(14, 7))
ax1 = fig.add_subplot(121, projection="3d")
pos_np = pos.get()
ax1.set_proj_type("persp")
ax1.view_init(elev=20, azim=45)
ax2 = fig.add_subplot(122)
ax3 = ax2.twinx()

scat = ax1.scatter(pos_np[:, 0], pos_np[:, 1], pos_np[:, 2], s=np.sqrt(mass.get()) * dotsize)
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
    global pos, vel, t_sim, a_factor, rho_min, rho_max, norm

    # 1. Transfert vers CPU pour les fonctions Numba complexes (Grid Search)
    pos_np = pos.get()
    vel_np = vel.get()
    mass_np = mass.get()

    # 2. Appel des fonctions Numba (CPU)
    sids, cstart, nx, ny, nz, x0, y0, z0 = build_grid(pos_np, h)
    rho_np, P_np = compute_density_pressure(pos_np, mass_np, h, rho0, k, sids, cstart, nx, ny, nz, x0, y0, z0)
    f_sph_np = compute_sph_forces(pos_np, vel_np, mass_np, rho_np, P_np, h, mu, sids, cstart, nx, ny, nz, x0, y0, z0)
    pe, f_grav_np = compute_gravity(pos_np, mass_np, G, soft)

    # 3. Retour sur GPU pour l'intégration et les calculs lourds restants
    f_sph = cp.array(f_sph_np)
    f_grav = cp.array(f_grav_np)
    rho = cp.array(rho_np)
    P = cp.array(P_np)
    
    F = f_sph + f_grav
    t_sim, a_factor, dt, ke = integrate_gpu(pos, vel, F, mass, t_sim, a_factor)

    te = ke + pe
    avg_temp = float(np.mean(P / (rho + epsilon)))  # BUG FIX: was appended twice before

    ke_list.append(float(ke))
    pe_list.append(float(pe))
    te_list.append(float(te))
    temp_list.append(float(avg_temp))
    time_list.append(float(t_sim))

    # 4. Préparation des couleurs (Matplotlib)
    cur_min, cur_max = float(cp.min(rho)), float(cp.max(rho))
    rho_min = min(rho_min, cur_min)
    rho_max = max(rho_max, cur_max)
    
    # Normalisation sur CPU pour Matplotlib
    rho_norm = (rho_np - rho_min) / (rho_max - rho_min + epsilon)
    sm = cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=cmap)
    colors = sm.to_rgba(rho_norm)

    # 5. Mise à jour du plot (CPU)
    scat._offsets3d = (pos_np[:, 0], pos_np[:, 1], pos_np[:, 2]) 
    scat.set_facecolor(colors)

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

    return scat, 

ani = FuncAnimation(fig, update, frames=24, interval=1, blit=False)
plt.show()