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
frames = 100              # number of time frames

x_limits = (-0.5, 0.5)   
y_limits = (-0.5, 0.5)
z_limits = (-0.5, 0.5)
Ngrid = (5, 5, 5)

# Generating a grid in the XZ plane (Y=0)
x_values = np.linspace(x_limits[0], x_limits[1], Ngrid[0])
y_values = np.linspace(y_limits[0], y_limits[1], Ngrid[1])
z_values = np.linspace(z_limits[0], z_limits[1], Ngrid[2])

X, Y, Z = np.meshgrid(x_values, y_values, z_values, indexing='ij')
COORDS = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

# Coil class definition
class Coil:
    def __init__(self, radius: float, Nloops: int, coordinate=(0, 0, 0), I=1.00, frequency=1.0, phase=0.0,
                 layers=1, coil_length=0.1, angle=0.00, **kwargs):
        
        self.radius = radius
        self.loops = Nloops
        self.x, self.y, self.z = coordinate
        self.max_current = I
        self.frequency = frequency
        self.current = self.max_current * np.sin(2 * np.pi * frequency * np.linspace(0, 1, frames) + phase)
        self.layers = layers
        self.coil_length = coil_length
        self.angle = angle
        self.cross_sectional_area = 2 * np.pi * radius**2
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
        print(distance_vector.shape)
        ln = np.linalg.norm(distance_vector, axis=2)
        distance_vector[ln < 1e-5] = 0

        B = np.cross(self.dl[None, :, :], distance_vector)
        denom = (ln**2 + self.cross_sectional_area**2)**(3/2)
        B /= denom[:, :, None]
        B *= self.layers * muθ / (4 * np.pi)
        B = np.sum(B, axis=1)

        B_time = self.current[:, None, None] * B[None, :, :]
        B_coil_grid = B_time.reshape(frames, Ngrid[0], Ngrid[1], Ngrid[2], 3).astype(np.float32)

        return B_coil_grid # (frames, Nx, Ny, Nz, 3)
    
    def force_on_coil(self, external_B: np.ndarray) -> np.ndarray:
        # external_B shape: (frames, Nx, Nz, 3)
        # self.dl shape: (Nseg * Nloops, 3)
        # self.current shape: (frames,)
        B_on_wire = np.zeros((frames, Nseg * self.loops, 3), dtype=np.float32)
        
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
        return force # (frames, 3)

class Simulation:
    def __init__(self, id):
        self.id = id
        self.coils = {}

    def add_coil(self, coil: Coil):
        coil_id = random.randint(1000, 9999)
        while coil_id in self.coils.keys():
            coil_id = random.randint(1000, 9999)
        
        self.coils[coil_id] = coil
        return coil_id
    
    def remove_coil(self, coil_id):
        if coil_id in self.coils:
            del self.coils[coil_id]
            return True
        return False
    
    def edit_coil(self, coil_id, new_coil):
        if coil_id in self.coils:
            self.coils[coil_id] = new_coil
            return True
        return False

    def run(self):
        self.external_B = np.zeros((frames, Ngrid[0], Ngrid[1], Ngrid[2], 3), dtype=np.float32)
        for coil in self.coils:
            self.external_B += coil.B_calculator(COORDS)
        
        forces = [coil.force_on_coil(self.external_B) for coil in self.coils]
        return self.external_B, forces
    
active_sims = {}
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/api/simulate', methods=['POST'])
def create_simulation():
    id = random.randint(1000, 9999)
    while id in active_sims:
        id = random.randint(1000, 9999)
    
    active_sims[id] = Simulation(id)
    return jsonify({'sim_id': id}), 200

@app.route("/api/simulate/<sim_id>", methods=['POST'])
def simulate(sim_id):
    sim_id = int(sim_id)
    if sim_id not in active_sims:
        return jsonify({"error": "Simulation not found"}), 404
    
    sim = active_sims[sim_id]
    B, forces = sim.run()
    return jsonify({
        "B": B.tolist(),
        "forces": [force.tolist() for force in forces]
    })

@app.route("/api/simulate/<sim_id>/add", methods=['POST'])
def add_coil(sim_id):
    sim_id = int(sim_id)
    if sim_id not in active_sims:
        return jsonify({"error": "Simulation not found"}), 404
    
    sim = active_sims[sim_id]
    coil_data = request.get_json()
    coil = Coil(**coil_data)
    coil_id = sim.add_coil(coil)

    return jsonify({'coil_id': coil_id}), 200

@app.route("/api/simulate/<sim_id>/<coil_id>/remove", methods=['POST'])
def remove_coil(sim_id, coil_id):
    sim_id = int(sim_id)
    coil_id = int(coil_id)
    if sim_id not in active_sims:
        return jsonify({"error": "Simulation not found"}), 404
    
    sim = active_sims[sim_id]
    if coil_id not in sim.coils:
        return jsonify({"error": "Coil not found"}), 404
    
    successful = sim.remove_coil(coil_id)
    if successful:
        return jsonify({"message": "Coil removed successfully"}), 200
    return jsonify({"error": "Failed to remove coil"}), 500

@app.route("/api/simulate/<sim_id>/<coil_id>/edit", methods=['POST'])
def edit_coil(sim_id, coil_id):
    sim_id = int(sim_id)
    coil_id = int(coil_id)
    if sim_id not in active_sims:
        return jsonify({"error": "Simulation not found"}), 404
    
    sim = active_sims[sim_id]
    if coil_id not in sim.coils:
        return jsonify({"error": "Coil not found"}), 404
    
    coil_data = request.get_json()
    new_coil = Coil(**coil_data)
    successful = sim.edit_coil(coil_id, new_coil)
    if successful:
        return jsonify({"message": "Coil edited successfully"}), 200
    return jsonify({"error": "Failed to edit coil"}), 500

HOST = '0.0.0.0'
PORT = 5500
if __name__ == '__main__':
    app.run(host=HOST, port=PORT)
    