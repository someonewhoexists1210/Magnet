import numpy as np
output_f = open('log.txt', 'w')

mu0 = 4 * np.pi * 1e-7   # vacuum permeability
I = 1.0                 # current in amps
R = 0.10                # radius in meters
Nseg = 100              # number of wire segments
a = 5 * 10**-3          # cross-sectional radius of wire (for realistic simulation)

Nloops = 400 # number of loops in the coil
loopsPerloop = 20  #overlapping loops to simulate layered coil

coil_positions_z = np.linspace(-0.15, 0.15, Nloops // loopsPerloop)
theta = np.linspace(0, 2*np.pi, Nseg, endpoint=False)
dtheta = 2*np.pi / Nseg

# Wire positions r_k
rx = R * np.cos(theta)
ry = R * np.sin(theta)
rz = coil_positions_z

# Segment vectors Δl_k
dlx = -R * np.sin(theta) * dtheta
dly =  R * np.cos(theta) * dtheta
dlz = np.zeros_like(theta)

def B_at_point(x, y, z):
    B = np.array([0.0, 0.0, 0.0])
    
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
        print(f"Computed B contribution {Bloop} from loop at z={loop_z:.3f} m", file=output_f)
        B += Bloop

    factor = mu0 / (4*np.pi)
    B *= factor
    Bx, By, Bz = B
    return Bx, By, Bz

xmax = 0.5
ymax = 0.5
zmax = 0.5

Nx = 15
Ny = 15    
Nz = 15

x_vals = np.linspace(-xmax, xmax, Nx)
y_vals = np.linspace(-ymax, ymax, Ny)
z_vals = np.linspace(-zmax, zmax, Nz)
Ivals = I * np.sin(np.linspace(0, 2*np.pi, 100))

Bx = np.zeros((Nz, Nx))
By = np.zeros((Nz, Nx))
Bz = np.zeros((Nz, Nx))

for i, z in enumerate(z_vals):
    for j, x in enumerate(x_vals):
        bx, by, bz = B_at_point(x, 0.0, z)
        print(f"Computed B at (i={i}, j={j}): Bx={bx:.3e}, By={by:.3e}, Bz={bz:.3e}", file=output_f)
        # Bmag = np.sqrt(bx**2 + bz**2)
        # Bx[i, j] = 2 * bx / Bmag
        # Bz[i, j] = 2 * bz / Bmag
        Bx[i, j] = bx
        By[i, j] = by
        Bz[i, j] = bz

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
        bx, by, bz = B_at_point(x, 0.0, z)
        print(f"Bx={bx:.3e} T, By={by:.3e} T, Bz={bz:.3e} T")


fig, ax = plt.subplots(1, 1)
Q = plt.quiver(X, Z, Bx, Bz, scale=5e-4)
# Q = plt.quiver(X, Y, Bx, By)
fig.canvas.mpl_connect('button_press_event', on_click)

anim = animation.FuncAnimation(fig, update_quiver, fargs=(Q, Bx, Bz),
                               interval=10, blit=False)
for loop_z in rz:
    plt.plot(R*np.cos(theta), R*np.sin(theta)*0 + loop_z, 'k.')  # projection of coil
plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.title("Magnetic field of circular loop")
plt.axis("equal")
plt.show()

m = Nloops * I * np.pi * R**2  # magnetic moment
print(f"Magnetic moment of coil: m = {m:.4e} A·m²")



