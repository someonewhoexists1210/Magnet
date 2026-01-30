import numpy as np
from matplotlib import animation, pyplot as plt

# GLOBAL VARIABLES
output_f = open('log.txt', 'w')
muθ = 4 * np.pi * 1e-7   # vacuum permeability
Nseg = 100               # number of wire segments
frames = 100
I_multipliers = np.sin(np.linspace(0, 2 * np.pi, frames))

x_values = np.linspace(-0.5, 0.5, 26)
# y_values = np.linspace(-0.5, 0.5, 50)
z_values = np.linspace(-0.5, 0.5, 26)

X, Z = np.meshgrid(x_values, z_values)
Y = np.zeros_like(X)

COORDS = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

length = lambda x, y, z: np.sqrt(x**2 + y**2 + z**2)


class Coil:
    def __init__(self, radius: float, Nloops: int, coordinate=(0, 0, 0), I=1.00, frequency=1.0, phase=0.0,
                 layers=1, coil_length=0.1, angle=0.00, area=0.0, **kwargs):
        
        self.radius = radius
        self.loops = Nloops
        self.x, self.y, self.z = coordinate
        self.max_current = I
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


    def compute_tangent_vectors(self):
        dl = np.column_stack((-1 * np.sin(self.theta), np.cos(self.theta), np.zeros_like(self.theta)))
        dl *= self.radius * self.dtheta
        dl = dl @ self.rotation_matrix.T
        dl = np.repeat(dl, self.loops, axis=0)

        return dl

    def B_calculator(self, points_array: np.ndarray) -> np.ndarray:
        distance_vector = self.r[None, :, :] - points_array[:, None, :]
        ln = np.linalg.norm(distance_vector, axis=2)
        distance_vector[ln < 1e-5] = 0

        B = np.cross(self.dl[None, :, :], distance_vector)
        denom = (ln**2 + self.cross_sectional_area**2)**(3/2)
        B /= denom[:, :, None]
        B *= self.layers * muθ / (4 * np.pi)
        B = np.sum(B, axis=1)

        B_time = self.current[:, None, None] * B[None, :, :]
        B_coil_grid = B_time.reshape(frames, len(z_values), len(x_values), 3).astype(np.float32)

        return B_coil_grid
    

def on_click(event):
    x = event.xdata
    z = event.ydata
    print(f"Clicked at x={x}, z={z}")
    nearest_x = np.argmin(np.abs(x_values - x))
    nearest_z = np.argmin(np.abs(z_values - z))
    print("Nearest indices: ", nearest_x, nearest_z)
    B_val = B[:, nearest_z, nearest_x, :]
    magnitude_over_time = length(B_val[:,0], B_val[:,1], B_val[:,2])
    print("Time index: ", magnitude_over_time.argmax())
    print("Max vector: ", B_val[magnitude_over_time.argmax()])
    # print(magnitude_over_time)
    
    
def update_quiver(frame, Q):
    f = frame % 100
    Q.set_UVC(B[f, :, :, 0], B[f, :, :, 2])
    return Q

        
n = Coil(radius=0.1, Nloops=100, coordinate=(0.15, 0, 0), layers=2, coil_length=0.2, phase=np.radians(180),
         frequency=4.0, I=1.0, angle=np.radians(90), area=(1/len(x_values)))
n1 = Coil(radius=0.1, Nloops=100, coordinate=(-0.1, 0, 0.0), layers=1, coil_length=0.1,
         frequency=4.0, I=1.0, angle=np.radians(90), area=(1/len(x_values)))


coils = [n, n1]

fig, ax = plt.subplots()



B_times = np.array([coil.B_calculator(COORDS) for coil in coils])
B = np.sum(B_times, axis=0)
Bx, By, Bz = B[0, :, :, 0], B[0, :, :, 1], B[0, :, :, 2]
# B_magnitude = length(Bx, By, Bz)
Q = plt.quiver(X, Z, Bx, Bz, scale=5e-3, pivot='mid', color='b', width=0.003, headwidth=3)

fig.canvas.mpl_connect('button_press_event', on_click)

anim = animation.FuncAnimation(fig, update_quiver, fargs=(Q,),
                            interval=20, blit=False)

for coil in coils:
    plt.plot(coil.r[:,0], coil.r[:,2], 'k.')  # coil wire segments

plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.xlim(-0.5, 0.5)
plt.ylim(-0.4, 0.4)
plt.title("Magnetic field of circular loop")
plt.axis("equal")
plt.show()