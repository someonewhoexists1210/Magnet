import sys
import numpy as np
from matplotlib import animation, pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.interpolate import RegularGridInterpolator

# GLOBAL VARIABLES
muθ = 4 * np.pi * 1e-7   # vacuum permeability
Nseg = 20               # number of wire segments
frames = 50              # number of time frames

x_limits = (-0.5, 0.5)   
z_limits = (-0.5, 0.5)
Ngrid = (25, 25)

force_arrow_scale = 2e-4  # scaling factor for force arrows

# Generating a grid in the XZ plane (Y=0)
x_values = np.linspace(x_limits[0], x_limits[1], 25)
z_values = np.linspace(z_limits[0], z_limits[1], 25)

X, Z = np.meshgrid(x_values, z_values, indexing='ij')
Y = np.zeros_like(X)
COORDS = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

# Helper functions
length2d = lambda x, z: np.sqrt(x**2 + z**2) 

# Coil class definition
class Coil:
    def __init__(self, radius: float, Nloops: int, coordinate=(0, 0, 0), I=1.00, frequency=1.0, phase=0.0,
                 layers=1, coil_length=0.1, angle=0.00, area=0.0, **kwargs):
        
        self.radius = radius
        self.loops = Nloops
        self.x, self.y, self.z = coordinate
        self.max_current = I
        self.frequency = frequency
        self.current = self.max_current * np.sin(2 * np.pi * frequency * np.linspace(0, 1, frames) + phase)
        self.layers = layers
        self.coil_length = coil_length
        self.angle = angle
        self.cross_sectional_area = area
        self.rotation_matrix = np.array([[np.cos(angle), 0, np.sin(angle)],
                                          [0, 1, 0],
                                          [-np.sin(angle), 0, np.cos(angle)]]) if angle != 0 else np.eye(3)
        self.to_fixed = lambda array: self.rotation_matrix @ array

        self.theta = np.linspace(0, 2*np.pi, Nseg, endpoint=False)
        self.dtheta = 2*np.pi / Nseg

        self.loop_positions = np.linspace(-self.coil_length/2 + self.z, self.coil_length/2 + self.z, self.loops)
        self.r = self.compute_wire_segments()
        self.dl = self.compute_tangent_vectors()

    # Compute wire segment positions
    def compute_wire_segments(self):
        r = np.column_stack((np.cos(self.theta), np.sin(self.theta)))
        r *= self.radius
        r = np.column_stack((
            np.repeat(r, self.loops, axis=0),
            np.tile(self.loop_positions - self.z, len(self.theta))
        ))

        r = r @ self.rotation_matrix.T
        r += np.array([self.x, self.y, self.z])
        return r

    # Compute tangent vectors for wire segments
    def compute_tangent_vectors(self):
        dl = np.column_stack((-1 * np.sin(self.theta), np.cos(self.theta), np.zeros_like(self.theta)))
        dl *= self.radius * self.dtheta
        dl = dl @ self.rotation_matrix.T
        dl = np.repeat(dl, self.loops, axis=0)

        return dl

    # Calculate magnetic field at given points across time frames
    def B_calculator(self, points_array: np.ndarray) -> np.ndarray:
        distance_vector = points_array[:, None, :] - self.r[None, :, :] 
        ln = np.linalg.norm(distance_vector, axis=2)
        distance_vector[ln < 1e-5] = 0

        B = np.cross(self.dl[None, :, :], distance_vector)
        denom = (ln**2 + self.cross_sectional_area**2)**(3/2)
        B /= denom[:, :, None]
        B *= self.layers * muθ / (4 * np.pi)
        B = np.sum(B, axis=1)

        B_time = self.current[:, None, None] * B[None, :, :]
        B_coil_grid = B_time.reshape(frames, Ngrid[0], Ngrid[1], 3).astype(np.float32)

        return B_coil_grid
    
    def force_on_coil(self, external_B: np.ndarray) -> np.ndarray:
        # external_B shape: (frames, Nx, Nz, 3)
        # self.dl shape: (Nseg * Nloops, 3)
        # self.current shape: (frames,)
        B_on_wire = np.zeros((frames, Nseg * self.loops, 3), dtype=np.float32)
        point = self.r[:, [0, 2]]  # Use x and z coordinates for interpolation

        for t in range(frames):
            interp = RegularGridInterpolator(
                (x_values, z_values),
                external_B[t],  # (Nx, Nz, 3)
                bounds_error=False,
                fill_value=0.0
            )
            B_on_wire[t] = interp(point)

        # df.shape (frames, Nseg * Nloops, 3)
        df = self.current[:, None, None] * np.cross(self.dl[None, :, :], B_on_wire)
        force = np.sum(df, axis=1)
        
        print(force.mean(axis=0))
        return force
    
# Event functions for plot interaction
def on_click(event):
    x = event.xdata
    z = event.ydata
    print(f"Clicked at x={x}, z={z}")
    # print(magnitude_over_time)
       
def update_quiver(frame, Q, force_arrows):
    f = frame % frames
    Q.set_UVC(B[f, :, :, 0], B[f, :, :, 2])
    
    # Update force arrows for each coil
    for i, arrow in enumerate(force_arrows):
        force_x = Force[i, f, 0] / force_arrow_scale
        force_z = Force[i, f, 2] / force_arrow_scale
        arrow.set_positions((coils[i].x, coils[i].z), 
                           (coils[i].x + force_x, coils[i].z + force_z))
    
    return [Q] + force_arrows
     
# Define coils
coil1 = Coil(
    radius=0.1,
    Nloops=10,
    coordinate=(0.3, 0.0, -0.2),
    I=1.0,
    frequency=2.0,
    phase=np.radians(0),
    layers=10,
    coil_length=0.2,
    angle=np.radians(0),
    area=1e-2
)

coil2 = Coil(
    radius=0.1,
    Nloops=10,
    coordinate=(0.0, 0.0, 0.1),
    I=2.0,
    frequency=2.0,
    phase=0.0,
    layers=10,
    coil_length=0.2,
    angle=0.0,
    area=1e-2
)

coil3 = Coil(
    radius=0.1,
    Nloops=10,
    coordinate=(-0.3, 0.0, -0.2),
    I=1.0,
    frequency=2.0,
    phase=np.radians(0),
    layers=10,
    coil_length=0.2,
    angle=np.radians(0),
    area=1e-2
)

coil4 = Coil(
    radius=0.1,
    Nloops=10,
    coordinate=(-0.3, 0.0, 0.2),
    I=2.0,
    frequency=5.0,
    phase=0.0,
    layers=10,
    coil_length=0.2,
    angle=0.0,
    area=1e-2
)

coils = [coil1, coil2]
if len(coils) == 0:
    sys.exit("No coils defined.")

# Calculate total magnetic field from all coils
B_times = np.array([coil.B_calculator(COORDS) for coil in coils])
B = np.sum(B_times, axis=0)
Bx, By, Bz = B[0, :, :, 0], B[0, :, :, 1], B[0, :, :, 2]

Force = np.array([coil.force_on_coil(B - B_times[i]) for i, coil in enumerate(coils)])


# Initial analysis at center line (x=0, z=0)
# nearest_x = np.argmin(np.abs(x_values))
# nearest_z = np.argmin(np.abs(z_values))
# B_val = B[:, nearest_z, nearest_x, :]

# max_z_time = B_val[:,2].argmax()
# max_z = B_val[max_z_time]

# min_z_time = B_val[:,2].argmin()
# min_z = B_val[min_z_time]

# with open('log.txt', 'a') as output_f:
#     output_f.write(f"Coil 1: {coil1.frequency}\t")
#     output_f.write(f"Coil 2: {coil2.frequency}\n")
#     output_f.write(f"Time index: {max_z_time}\n")
#     output_f.write(f"Max vector: {max_z}\n\n")

# Plot management
fig, ax = plt.subplots()
Q = plt.quiver(
    X, Z, 
    Bx, Bz, 
    scale=5e-3, 
    pivot='mid', 
    color='b', 
    width=0.003, 
    headwidth=3,
)

# Plot coil projections
for coil in coils:
    plt.plot(coil.r[:,0], coil.r[:,2], 'k.')  

# Create force arrows for each coil using FancyArrowPatch
force_arrows = []
for i, force_coil in enumerate(Force):
    print(f"Coil X: {coils[i].x}, Z: {coils[i].z} -> Force: {force_coil[25]}")
    arrow = FancyArrowPatch(
        (coils[i].x, coils[i].z),
        (coils[i].x + force_coil[0, 0] / force_arrow_scale, coils[i].z + force_coil[0, 2] / force_arrow_scale),
        arrowstyle='->', 
        mutation_scale=20, 
        linewidth=2,
        color='red'
    )
    ax.add_patch(arrow)
    force_arrows.append(arrow)

fig.canvas.mpl_connect('button_press_event', on_click)

# Update animation to include force arrows
anim = animation.FuncAnimation(
    fig, 
    update_quiver, 
    fargs=(Q, force_arrows),
    interval=20, 
    blit=False
)

plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.xlim(x_limits)
plt.ylim(z_limits)
plt.title("Magnetic field of circular loop")
plt.axis("equal")
plt.show()