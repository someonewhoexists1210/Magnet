import numpy as np
output_f = open('log.txt', 'w')

mu0 = 4 * np.pi * 1e-7   # vacuum permeability
I = 1.0                # current in amps
R = 0.10                # radius in meters
Nseg = 50              # number of wire segments
a = 5 * 10**-3          # cross-sectional radius of wire (for realistic simulation)

Nloops = 30 # number of loops in the coil
loopsPerloop = 10  #overlapping loops to simulate layered coil
coil_length = 0.15

loop_positions_z = np.linspace(-coil_length, coil_length, Nloops // loopsPerloop)
theta = np.linspace(0, 2*np.pi, Nseg, endpoint=False)
dtheta = 2*np.pi / Nseg

def B_at_point(x, y, z, r, dl):
    B = np.array([0.0, 0.0, 0.0])
    rx, ry, rz = r
    dlx, dly, dlz = dl
    
    for loop_z in rz:
        Bloop = np.array([0.0, 0.0, 0.0])
        for k in range(Nseg):
            # vector from segment to field point
            Rx = x - rx[k]
            Ry = y - ry[k]
            Rz = z - loop_z
            
            R_dist = np.sqrt(Rx**2 + Ry**2 + Rz**2)
            
            # avoid singularity if you ever sample on the wire
            if R_dist < 1e-9:
                continue
            
            # cross product dl × R
            rk = np.array([Rx, Ry, Rz])
            dlk = np.array([dlx[k], dly[k], dlz[k]])
            ck = np.cross(dlk, rk)
            ck_total = np.array([0.0, 0.0, 0.0])
            for i in range(loopsPerloop):
                ck_total += ck / (R_dist**2 + (a + 2 * (i-1) * R)**2)**(3/2)            
            Bloop += ck_total
        B += Bloop

    factor = mu0 / (4*np.pi)
    B *= factor
    Bx, By, Bz = B
    return Bx, By, Bz

xmax = 0.5
ymax = 0.5
zmax = 0.5

Nx = 50
Ny = 50    
Nz = 50

x_vals = np.linspace(-xmax, xmax, Nx)
y_vals = np.linspace(-ymax, ymax, Ny)
z_vals = np.linspace(-zmax, zmax, Nz)
Ivals = I * np.sin(np.linspace(0, 2*np.pi, 100))


import matplotlib.pyplot as plt
from matplotlib import animation
# X, Z = np.meshgrid(x_vals, z_vals)
X, Z = np.meshgrid(x_vals, z_vals)

def update_quiver(frame, q, Bx, Bz):
    f = frame % 100 
    Ival = Ivals[f]
    q.set_UVC(Bx * Ival, Bz * Ival)
    return q

def on_click(event):
    if event.xdata is not None and event.ydata is not None:
        x = event.xdata
        z = event.ydata
        print(f"Clicked at: x={x:.4f} m, z={z:.4f} m")
        print("B field at this point: ", end='')
        nearest_x_idx = (np.abs(x_vals - x)).argmin()
        nearest_z_idx = (np.abs(z_vals - z)).argmin()
        bx = Bx[nearest_z_idx, nearest_x_idx]
        by = By[nearest_z_idx, nearest_x_idx]
        bz = Bz[nearest_z_idx, nearest_x_idx]
        print(f"Bx={bx:.3e} T, By={by:.3e} T, Bz={bz:.3e} T")


fig, ax = plt.subplots(1, 1)
Bx = np.zeros((Nz, Nx))
By = np.zeros((Nz, Nx))
Bz = np.zeros((Nz, Nx))

def compute_coil(x, y):
    # Wire positions r_k
    rx = R * np.cos(theta) + x
    ry = R * np.sin(theta) + y
    rz = loop_positions_z

    r = [rx, ry, rz]

    # Segment vectors Δl_k
    dlx = -R * np.sin(theta) * dtheta
    dly =  R * np.cos(theta) * dtheta
    dlz = np.zeros_like(theta)

    dl = [dlx, dly, dlz]

    Bcoilx = np.zeros((Nz, Nx))
    Bcoily = np.zeros((Nz, Nx))
    Bcoilz = np.zeros((Nz, Nx))

    for i, z in enumerate(z_vals):
        for j, x in enumerate(x_vals):
            bx, by, bz = B_at_point(x, 0.0, z, r, dl)
            print(f"Computed B at (i={i}, j={j}): Bx={bx:.3e}, By={by:.3e}, Bz={bz:.3e}", file=output_f)
            # Bmag = np.sqrt(bx**2 + bz**2)
            # Bx[i, j] = 2 * bx / Bmag
            # Bz[i, j] = 2 * bz / Bmag
            Bcoilx[i, j] = bx
            Bcoily[i, j] = by
            Bcoilz[i, j] = bz

    return Bcoilx, Bcoily, Bcoilz


# x, y, current direction
coil_positions = [(-00.10, 0, -1), (0.10, 0, 1)]

for i, pos in enumerate(coil_positions):
    Bx_coil, By_coil, Bz_coil = compute_coil(pos[0], pos[1])
    #contribution at point x = 0, z = 0
    xz0_bx = Bx_coil[Nz//2, Nx//2] * pos[2]
    xz0_by = By_coil[Nz//2, Nx//2] * pos[2]
    xz0_bz = Bz_coil[Nz//2, Nx//2] * pos[2]
    print(f'At (x=0, z=0) coil {i+1} B field: Bx={xz0_bx:.3e}, By={xz0_by:.3e}, Bz={xz0_bz:.3e}')
    Bx += Bx_coil * pos[2]
    By += By_coil * pos[2]
    Bz += Bz_coil * pos[2]
    xz0_bx = Bx[Nz//2, Nx//2]
    xz0_by = By[Nz//2, Nx//2]
    xz0_bz = Bz[Nz//2, Nx//2]
    print(f'New total B field at (x=0, z=0): Bx={xz0_bx:.3e}, By={xz0_by:.3e}, Bz={xz0_bz:.3e}')

Q = plt.quiver(X, Z, Bx, Bz, pivot='mid', scale=1e-4)
# Q = plt.quiver(X, Y, Bx, By)
fig.canvas.mpl_connect('button_press_event', on_click)

anim = animation.FuncAnimation(fig, update_quiver, fargs=(Q, Bx, Bz),
                            interval=5, blit=False)

for x, y, _ in coil_positions:
    for loop_z in loop_positions_z:
        plt.plot(R*np.cos(theta) + x, R*np.sin(theta)*0 + loop_z + y, 'k.')  # projection of coil

plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.title("Magnetic field of circular loop")
plt.axis("equal")
plt.show()

m = Nloops * I * np.pi * R**2  # magnetic moment
print(f"Magnetic moment of coil: m = {m:.4e} A·m²")
