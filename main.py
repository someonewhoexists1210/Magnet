import numpy as np

mu0 = 4 * np.pi * 1e-7   # vacuum permeability
I = 1.0                 # current in amps
R = 0.05                # radius in meters
Nseg = 300              # number of wire segments

theta = np.linspace(0, 2*np.pi, Nseg, endpoint=False)
dtheta = 2*np.pi / Nseg

# Wire positions r_k
rx = R * np.cos(theta)
ry = R * np.sin(theta)
rz = np.zeros_like(theta)

# Segment vectors Δl_k
dlx = -R * np.sin(theta) * dtheta
dly =  R * np.cos(theta) * dtheta
dlz = np.zeros_like(theta)

def B_at_point(x, y, z):
    B = np.array([0.0, 0.0, 0.0])
    
    for k in range(Nseg):
        # vector from segment to field point
        Rx = x - rx[k]
        Ry = y - ry[k]
        Rz = z - rz[k]
        
        R = np.sqrt(Rx**2 + Ry**2 + Rz**2)
        
        # avoid singularity if you ever sample on the wire
        if R < 1e-9:
            continue
        
        # cross product dl × R
        rk = np.array([Rx, Ry, Rz])
        dlk = np.array([dlx[k], dly[k], dlz[k]])
        ck = np.cross(dlk, rk)
        
        B += ck / R**3

    factor = mu0 / (4*np.pi)
    B *= factor
    Bx, By, Bz = B
    return Bx, By, Bz

xmax = 0.10
zmax = 0.10

Nx = 20    
Nz = 20

x_vals = np.linspace(-xmax, xmax, Nx)
z_vals = np.linspace(-zmax, zmax, Nz)
Ivals = np.sin(np.linspace(0, 2*np.pi, 100))

Bx = np.zeros((Nz, Nx))
Bz = np.zeros((Nz, Nx))

for i, z in enumerate(z_vals):
    for j, x in enumerate(x_vals):
        bx, by, bz = B_at_point(x, 0.0, z)
        print(f"Computed B at (i={i}, j={j}): Bx={bx:.3e}, Bz={bz:.3e}")
        # Bmag = np.sqrt(bx**2 + bz**2)
        # Bx[i, j] = 2 * bx / Bmag
        # Bz[i, j] = 2 * bz / Bmag
        Bx[i, j] = bx
        Bz[i, j] = bz

import matplotlib.pyplot as plt
from matplotlib import animation
X, Z = np.meshgrid(x_vals, z_vals)

def update_quiver(frame, q, Bx, Bz):
    f = frame % 100 
    Ival = Ivals[f]
    q.set_UVC(Bx * Ival, Bz * Ival)
    return q

fig, ax = plt.subplots(1, 1)
Q = plt.quiver(X, Z, Bx, Bz)

anim = animation.FuncAnimation(fig, update_quiver, fargs=(Q, Bx, Bz),
                               interval=10, blit=False)
plt.plot(R*np.cos(theta), R*np.sin(theta)*0, 'k.')  # projection of coil
plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.title("Magnetic field of circular loop")
plt.axis("equal")
plt.show()
plt.pause(0.1)

