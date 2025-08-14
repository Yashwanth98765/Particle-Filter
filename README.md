# Particle-Filter

# Portfolio 2 – Particle Filter for Ball Trajectory Estimation

**A simulation and estimation system for tracking the trajectories of multiple flying balls using a Particle Filter.**

---

## Task

Realize an implementation of the Particle Filter in a programming language of your choice for a simulation of the ball-throwing example from the lecture slides. The task of your Particle Filter is to estimate the positions and velocity vectors of n ≥ 1 balls flying simultaneously only from the observed erroneous positions over time.

 - Simulate the trajectory of n balls with the parameters launch position (x, y) (the imaginary ground is at y = 0), launch speed and launch angle/launch direction of each ball.

 - The inital positions and flying directions of the balls are unknown to your estimation. You can only assume that the ball starts within a rather large range of, for example, 50 × 50 meters.

 - Simulate the observations of the ball positions (x, y). The estimated ball positions shall be subject to high uncertainty and it shall be possible to parameterize this uncertainty. In addition, the time span between two observations shall be variable and the observations shall be able to drop out completely over a certain period of time. It is necessary that the positions of the ball is also estimated during the time of the sensor failure.
  
 - How to deal with more than one ball flying at the same time? How do you define your state? Think intensively about what the transition model and the evaluation model should look like in the case of n balls. How do you estimate n positions from the sensor fusion density? If the density of the ball positions is multimodal, select a suitable method that can determine the positions of the balls.
  
 - You are not able to distinguish between the balls. They are indistinguishable from an observational point of view.
  
 - The starting position and other starting parameters are just as unknown and cannot be specified more precisely. You should handle both the case where the starting positions and directions of the balls are very similar and the case where the positions and directions are clearly different in one common approach.




---

## File Structure

```
portfolio2/
├── main.py               # Entry point: runs the simulation and estimation
├── ball_simulation.py    # Simulates physical trajectories and observations
├── particle_filter.py    # Implements the Particle Filter
├── utils.py              # Handles visualization and Pygame animation
├── README.md             # This file
```

---

## Run the Program

### Dependencies

Tested on Python 3.13.4. install dependencies:

```bash
pip install numpy matplotlib pygame scikit-learn scipy
```

### Simulation

```bash
python main.py
```

This will:
- Simulate trajectories of multiple balls
- Add noisy/missing observations
- Estimate states using Particle Filtering
- Visualize results via both static plots and animated simulation

---

## Configuration Parameters

Below are the key parameters you can tune in main.py:

### Simulation Parameters

| Parameter | Description | Example |
|----------|-------------|-------|
| `NUM_BALLS` | Number of balls to simulate | `2` |
| `DT_SIMULATION` | Time step for simulation (seconds) | `0.01` |
| `TOTAL_TIME` | Total duration of simulation (seconds) | `3.0` |
| `TRUE_BALL_PARAMS` | Launch positions, speeds, and angles of balls | `{'x0': 5.0, 'y0': 0.0, 'speed': 15.0, 'angle_deg': 40.0}` |

---

### Observation Parameters

| Parameter | Description | Example |
|----------|-------------|-------|
| `OBSERVATION_INTERVAL` | Time interval between observations | `0.1 s` |
| `OBSERVATION_NOISE_STD` | Std. dev of observation noise | `0.1 m` |
| `DROPOUT_PERIODS` | Time ranges with no observations | `[(1.5, 2.5)]` |

---

### Particle Filter Parameters

| Parameter | Description | Example |
|----------|-------------|-------|
| `NUM_PARTICLES` | Number of particles | `5000` |
| `PROCESS_NOISE_STD_POS` | Std. dev of process noise (position) | `0.1 m` |
| `PROCESS_NOISE_STD_VEL` | Std. dev of process noise (velocity) | `0.1 m/s` |

---

### Initial State Ranges

| State | Range |
|-------|-------|
| `x` | `(0.0, 10.0)` |
| `y` | `(0.0, 10.0)` |
| `vx` | `(0.0, 10.0)` |
| `vy` | `(0.0, 10.0)` |

These ranges define the initial distribution of particles at the start.

---

## Visual Output

Your program will generate:

- **Matplotlib plot of true vs estimated trajectories**  
- **Matplotlib plot of estimation errors over time**  
- **Live animation using Pygame**, showing real-time motion and particle behavior

---

## Clustering Method

The filter uses clustering to handle the **multimodal particle distribution** created by multiple balls. Two algorithms are supported:

- **k-means** – best when the number of balls is known
- **DBSCAN** – for adaptive clustering when ball count is uncertain

You can select the method via a config flag in the code.

---

## Missing Data Handling

During the simulation, **observation dropouts** are introduced by skipping measurements within predefined time ranges (`DROPOUT_PERIODS`). When no observations are available:
- The Particle Filter continues propagating particles using the motion model
- Weights are **not updated** until new observations arrive
- The system still estimates positions during the dropout phase, simulating real-world sensor failures

---

## Extra Features

- Real-time **Pygame animation**
- Toggle between **k-means** and **DBSCAN** clustering
- Dynamic error visualization
- Multiple test configurations for different launch scenarios

---

## References

- Course lecture slides
- [Particle Filter and Monte Carlo Localization (Cyrill Stachniss)](https://www.youtube.com/watch?v=MsYlueVDLI0&list=WL&index=1&t=2104s&ab_channel=CyrillStachniss)
- [scikit-learn clustering documentation](https://scikit-learn.org/stable/modules/clustering.html)

---

## Tested Environment

- Python 3.13.4
- Windows 11
- Libraries: `numpy`, `matplotlib`, `pygame`, `scikit-learn`, `scipy`, `sys`

---

## Authors
- Sai Karthik Shankar  
- Yashwanth Goud Chithaloori
---
