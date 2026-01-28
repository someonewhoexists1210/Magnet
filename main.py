import numpy as np
from matplotlib import animation, pyplot as plt

# GLOBAL VARIABLES
output_f = open('log.txt', 'w')
mu0 = 4 * np.pi * 1e-7   # vacuum permeability
Nseg = 100               # number of wire segments

length = lambda x, y, z: np.sqrt(x**2 + y**2 + z**2)


class Coil:
    def __init__(self, radius: float, Nloops: int, coordinate=(0, 0, 0), I=1.00, 
                 layers=1, coil_length=0.1, angle=0.00, area=0.0, **kwargs):
        
        self.radius = radius
        self.loops = Nloops
        self.x, self.y, self.z = coordinate
        self.current = I
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
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.r[:, 0], self.r[:, 1], self.r[:, 2], 'b-', linewidth=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()


        self.dl = self.compute_tangent_vectors()

    def compute_wire_segments(self):
        r = np.column_stack((np.cos(self.theta), np.sin(self.theta)))
        r *= self.radius
        r += np.array([self.x, self.y])

         
        return np.column_stack((
            np.repeat(r, self.loops, axis=0),
            np.tile(self.loop_positions, len(self.theta))
        ))

        

    def compute_tangent_vectors(self):
        dl = np.column_stack((-1 * np.sin(self.theta), np.cos(self.theta)))
        dl *= self.radius * self.dtheta

        return np.column_stack((
            np.repeat(dl, self.loops, axis=0),
            np.tile(self.loop_positions, len(self.theta))
        ))


    def B_at_point(self, point: np.ndarray):
        pass
        


        
n = Coil(radius=0.1, Nloops=5, coordinate=(0, 0, 0), I=1.0, layers=1, coil_length=0.2, )