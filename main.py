import numpy as np
import matplotlib.pyplot as plt
from ball_simulation import BallSimulation
from particle_filter import ParticleFilter  
from utils import plot_results, animate_simulation_pygame

# --- Simulation Parameters ---
NUM_BALLS = 2
DT_SIMULATION = 0.01  
TOTAL_TIME = 10.0      

# Define true ball parameters (these are the 'ground truth' and unknown to the PF)
# Adjust these to test different scenarios.
TRUE_BALL_PARAMS = [
    {'x0': 5.0, 'y0': 0.0, 'speed': 15.0, 'angle_deg': 40.0},
    {'x0': 0.0, 'y0': 10.0, 'speed': 20.0, 'angle_deg': 70.0},
    # {'x0': 7.0, 'y0': 3.0, 'speed': 15.0, 'angle_deg': 45.0},
    # {'x0': 3.5, 'y0': 0.0, 'speed': 15.2, 'angle_deg': 44.0},
    # {'x0': -5.0, 'y0': 0.0, 'speed': 10.0, 'angle_deg': 30.0},
    # {'x0': 10.0, 'y0': 0.0, 'speed': 20.0, 'angle_deg': 80.0},
]

# --- Observation Parameters ---
OBSERVATION_INTERVAL = 0.05 # Time between sensor observations (s)
OBSERVATION_NOISE_STD = 0.1 # Standard deviation of Gaussian noise of (x,y) m
# Periods where observations will drop out completely. The filter must estimate during these times.
DROPOUT_PERIODS =[(1.5, 2.5), (3.5, 4.0)] 

# --- Particle Filter Parameters ---
NUM_PARTICLES =2000
PROCESS_NOISE_STD_POS = 0.1 
PROCESS_NOISE_STD_VEL = 0.1 
# Initial ranges for particles. This defines the broad prior for the filter,
# as the balls' starting parameters are unknown.
INITIAL_STATE_RANGES = {
    'x': (0.0, 10.0),  # Balls can start anywhere in this X range
    'y': (0.0, 10.0),   # Balls start on or above ground
    'vx': (0.0, 20.0), # Initial X velocity can be in this range
    'vy': (0.0, 20.0)    # Initial Y velocity (typically positive for launch)
}

# --- Initialize Simulation ---
simulator = BallSimulation(TRUE_BALL_PARAMS, DT_SIMULATION)

# --- Initialize Particle Filter ---
pf = ParticleFilter(NUM_PARTICLES, NUM_BALLS, INITIAL_STATE_RANGES,
                    PROCESS_NOISE_STD_POS, PROCESS_NOISE_STD_VEL,
                    OBSERVATION_NOISE_STD)

# --- Main Simulation and Estimation Loop ---
true_trajectories = [] # Stores ground truth states (for plotting/evaluation)
observations_log = []  # Stores noisy observations (or None if no observation)
estimated_trajectories = [] # Stores estimated states from PF
particle_states = []  # Initialize list to store particle states

current_time = 0.0
last_observation_time = -np.inf # Initialize to ensure an observation is attempted at t=0

print(f"Starting simulation with {NUM_BALLS} balls and {NUM_PARTICLES} particles...")

# Loop through simulation time
while current_time <= TOTAL_TIME + DT_SIMULATION/2: 
    # 1. Simulate true ball positions (ground truth)
    simulator.step() # Advance the true physical simulation
    true_ball_states = simulator.get_current_states()
    true_trajectories.append(true_ball_states)

    # 2. Check for observation dropout periods
    is_dropout = False
    for start, end in DROPOUT_PERIODS:
        if start <= current_time <= end:
            is_dropout = True
            break

    # 3. Generate and log observations based on interval and dropout
    current_observation = None
    if not is_dropout and (current_time - last_observation_time) >= OBSERVATION_INTERVAL - 1e-6:
        # Get noisy observations from the simulator at this time step
        observed_positions = simulator.get_observed_positions(OBSERVATION_NOISE_STD)
        observations_log.append(observed_positions)
        current_observation = observed_positions
        last_observation_time = current_time
    else:
        observations_log.append(None) 

    # 4. Particle Filter Steps
    pf.predict(DT_SIMULATION) # Predict next state of particles based on dynamics
    if current_observation is not None:
        pf.update(current_observation) # If an observation is available, update particle weights and resample

    # 5. Estimate ball positions from the current particle set
    # This will give the best estimate of the ball states based on the particles
    estimated_ball_states = pf.estimate_ball_positions()
    estimated_trajectories.append(estimated_ball_states)

    # Store particle states
    particle_states.append(pf.get_particle_states())

    current_time += DT_SIMULATION
    # Loading bar visualization
    progress = current_time / TOTAL_TIME
    bar_length = 40
    filled_length = int(bar_length * progress)
    bar = '=' * filled_length + '-' * (bar_length - filled_length)
    print(f"\r[{bar}] {progress*100:5.1f}%  Time: {current_time:.2f}s", end='', flush=True)

print("Simulation complete. Generating plots...")

# --- Plot Results ---
plot_results(true_trajectories, observations_log, estimated_trajectories, DT_SIMULATION)
animate_simulation_pygame(true_trajectories, particle_states, estimated_trajectories, DT_SIMULATION)