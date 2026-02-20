![Magnet](/static/logo.png)

# Magneto Sim
A browser based 3D open-source magnetic field and force simulator for multiple coils

## Features
* Add and edit multiple coils
* Adjustable coil parameters, including:
   * Position
   * Radius
   * Number of loops
   * Number of layers
   * Current
   * Frequency and Phase
   * Angle
* Force calculations on the coils
* Interactive
  * Timeline playback
  * Arrows for vector fields and forces

## Tech Stack
### Backend
- Numpy
- Flask
- Scipy

### Frontend
- THREE.js
- OrbitControls

## Physics
The coil is first split up into "Nseg" many segments, which is then used to calculate the magnetic field using the Biot Savart law

The current over timeframes is made using a sin wave with an amplitude of the max current

The forces are calculated using
```
dF = I dl × B_external
```
excluding the coils own magnetic field (the coil aint gonna accelerate itself)

## Installation
### 1. Clone the Repo
### 2. Install dependencies
``` pip install -r requirements.txt```
### 3. Run the app
First add ```app.run()``` inside the main.py file, then run it.

## Controls
* **Add coil**: creates a new editable coil
* **Edit panel**: adjust parameters
* **Calculate**: compute field and forces
* **Timeline**: scrub animation frames
* **Play/Pause**: animate AC behavior
* **Toggles**:

  * Grid
  * Field vectors
  * Force arrows
  * Vector scale