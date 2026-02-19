from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import random

app = Flask(__name__)
CORS(app, resources={"*": {"origins": "*"}})


# GLOBAL VARIABLES
muθ = 4 * np.pi * 1e-7   # vacuum permeability
Nseg = 20               # number of wire segments
frames = 30              # number of time frames


# Coil class definition
class Coil:
    def __init__(self, radius: float, loops: int, position=(0, 0, 0), current=1.00, frequency=1.0, phase=0.0,
                 layers=1, angle=0.00, **kwargs):
        
        # all values in degrees, not radians
        self.radius = radius
        self.loops = loops
        self.x, self.y, self.z = position
        self.max_current = current
        self.frequency = frequency
        if self.frequency == 0:
            self.current = np.full(frames, self.max_current)
        else:
            self.current = self.max_current * np.sin(2 * np.pi * frequency * np.linspace(0, 1, frames) + np.radians(phase))
        self.layers = layers
        self.coil_length = loops * 0.01
        self.angle = np.radians(angle)

        c = np.cos(self.angle)
        s = np.sin(self.angle)
        self.rotation_matrix = np.array([[c, 0, s],
                                          [0, 1, 0],
                                          [-s, 0, c]]) if self.angle != 0 else np.eye(3)
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
        return r # (Nseg * Nloops, 3)

    # Compute tangent vectors for wire segments
    def compute_tangent_vectors(self):
        dl = np.column_stack((-1 * np.sin(self.theta), np.cos(self.theta), np.zeros_like(self.theta)))
        dl *= self.radius * self.dtheta
        dl = dl @ self.rotation_matrix.T
        dl = np.repeat(dl, self.loops, axis=0)

        return dl

    # Calculate magnetic field at given points across time frames
    def B_calculator(self, points_array: np.ndarray) -> np.ndarray:
        distance_vector = points_array[:, None, :] - self.r[None, :, :] # (Nx*Ny*Nz, Nseg*Nloops, 3)
        ln = np.linalg.norm(distance_vector, axis=2)
        print("Min distance:", np.min(ln))
        distance_vector[ln < 1e-5] = 0

        B = np.cross(self.dl[None, :, :], distance_vector) # (Nx*Ny*Nz, Nseg*Nloops, 3)
        denom = (ln**2 + 1e-6)**(3/2)
        B /= denom[:, :, None]
        B *= self.layers * muθ / (4 * np.pi)
        B = np.sum(B, axis=1) # (Nx*Ny*Nz, 3)

        B_time = self.current[:, None, None] * B[None, :, :] # (frames, Nx*Ny*Nz, 3)

        return B_time
    
    def force_on_coil(self, external_B: np.ndarray, vals: tuple) -> np.ndarray:
        # external_B shape: (frames, Nx, Ny, Nz, 3)
        # self.dl shape: (Nseg * Nloops, 3)
        # self.current shape: (frames,)
        B_on_wire = np.zeros((frames, Nseg * self.loops, 3), dtype=np.float32)
        
        x_values, y_values, z_values = vals
        for t in range(frames):
            interp = RegularGridInterpolator(
                (x_values, y_values, z_values),
                external_B[t],  # (Nx, Ny, Nz, 3)
                bounds_error=False,
                fill_value=0.0
            )
            B_on_wire[t] = interp(self.r)

        # df.shape (frames, Nseg * Nloops, 3)
        df = self.current[:, None, None] * np.cross(self.dl[None, :, :], B_on_wire)
        force = np.sum(df, axis=1)
        force = force.tolist()
        return force # (frames, 3)

class Simulation:
    def __init__(self, id, limits=1.0, points=9, coils: dict[int, Coil]={}):
        self.id = id
        self.coils = coils
        self.Ngrid = (points, points, points)


        # Generating a grid in the XZ plane (Y=0)
        x_values = np.linspace(-limits, limits, self.Ngrid[0])
        y_values = np.linspace(-limits, limits, self.Ngrid[1])
        z_values = np.linspace(-limits, limits, self.Ngrid[2])

        X, Y, Z = np.meshgrid(x_values, y_values, z_values, indexing='ij')
        self.COORDS = np.column_stack((X.ravel(), Y.ravel(), Z.ravel())) # (Nx*Ny*Nz, 3)
        self.vals = (x_values, y_values, z_values)

    def run(self):
        self.external_B = np.zeros((frames, self.COORDS.shape[0], 6))
        toFextB = np.zeros((frames, self.Ngrid[0], self.Ngrid[1], self.Ngrid[2], 3))

        # set coordinates once
        exp_coords = np.broadcast_to(
            self.COORDS,
            (frames, *self.COORDS.shape)
        )
        self.external_B[..., :3] = exp_coords

        per_c = {}
        for id, coil in self.coils.items():
            calresB = coil.B_calculator(self.COORDS)  # (frames, Npts, 3)

            # add only B field
            self.external_B[..., 3:] += calresB
            calresF = calresB.reshape(
                frames,
                self.Ngrid[0],
                self.Ngrid[1],
                self.Ngrid[2],
                3
            )
            per_c[id] = calresF

            toFextB += calresF

        forces = {
            id: coil.force_on_coil(toFextB - per_c[id], self.vals)
            for id, coil in self.coils.items()
        }

        return self.external_B, forces

    
active_sims = {1: Simulation(1)}
del active_sims[1]

@app.route('/', methods=['GET'])
def index():
    return render_template('/index.html')


@app.route("/api/simulate/", methods=['GET'])
def get_simulation(sim_id):
    sim_id = int(sim_id)
    if sim_id not in active_sims:
        return jsonify({"error": "Simulation not found"}), 404
    
    sim = active_sims[sim_id]
    return jsonify({
        "sim_id": sim.id,
        "coils": [{
            "coil_id": coil_id,
            "radius": coil.radius,
            "loops": coil.loops,
            "coordinate": (coil.x, coil.y, coil.z),
            "current": coil.max_current,
            "frequency": coil.frequency,
            "phase": 0.0,
            "layers": coil.layers,
            "coil_length": coil.coil_length,
            "angle": coil.angle
        } for coil_id, coil in sim.coils.items()]
        }), 200

@app.route("/api/simulate/", methods=['POST'])
def simulate():
    data = request.get_json()
    glimit = data.get('limits', 1.0)
    gpoints = data.get('points', 10)
    coils = data.get('coils', [])

    sim = Simulation(random.randint(1000, 9999), limits=glimit, points=gpoints, coils={c['id']: Coil(**c) for c in coils})
    active_sims[sim.id] = sim
    
    B, forces = sim.run()
    B[..., 3:] *= 1e5
    forces = {id: [[float(f) * 1e5 for f in frame] for frame in frames] for id, frames in forces.items()}
    r = jsonify({
        "sim_id": sim.id,
        "B": B.tolist(),
        "forces": forces
    })
    
    return r

@app.route('/api/simulate/<sim_id>/delete', methods=['POST'])
def delete_sim(sim_id):
    sim_id = int(sim_id)
    if sim_id not in active_sims:
        return jsonify({"error": "Simulation not found"}), 404
    
    del active_sims[sim_id]
    return jsonify({"message": "Simulation deleted successfully"}), 200

HOST = '0.0.0.0'
PORT = 5500
if __name__ == '__main__':
    app.run(host=HOST, port=PORT, debug=True)
    