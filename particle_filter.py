import numpy as np
from scipy.optimize import linear_sum_assignment # For Hungarian algorithm
from sklearn.cluster import KMeans # For clustering
from sklearn.cluster import DBSCAN
g = 9.81

class ParticleFilter:
    #particle Filter for Tracking Multiple Indistinguishable Balls.
    def __init__(self, num_particles, num_balls, initial_state_ranges,
                 process_noise_std_pos, process_noise_std_vel,
                 observation_noise_std):
        """
        Initializes the Particle Filter.
            num_particles (int): Number of particles in the filter.
            num_balls (int): Number of balls to track.
            initial_state_ranges (dict): Dictionary with keys 'x', 'y', 'vx', 'vy'.
            Each key maps to a tuple (min, max) defining the range for that state variable.
            process_noise_std_pos (float): Standard deviation of process noise for position.
            process_noise_std_vel (float): Standard deviation of process noise for velocity.
            observation_noise_std (float): Standard deviation of observation noise for (x, y) positions.
        """
        self.num_particles = num_particles
        self.num_balls = num_balls
        self.process_noise_std_pos = process_noise_std_pos
        self.process_noise_std_vel = process_noise_std_vel
        self.observation_noise_std = observation_noise_std

        # Initialize particles: each particle holds the state of all balls
        # e.g., for 2 balls: [x1, y1, vx1, vy1, x2, y2, vx2, vy2]
        self.particles = np.zeros((num_particles, num_balls * 4)) # 4 state variables per ball
        self.weights = np.ones(num_particles) / num_particles # Initialize weights uniformly

        self._initialize_particles(initial_state_ranges)

    def _initialize_particles(self, initial_state_ranges):
        """
        Initializes particles with random states within the specified ranges.
        """
        for i in range(self.num_particles):
            for j in range(self.num_balls):
                # Sample x, y, vx, vy for each ball within the particle
                self.particles[i, j*4 + 0] = np.random.uniform(*initial_state_ranges['x'])
                self.particles[i, j*4 + 1] = np.random.uniform(*initial_state_ranges['y'])
                self.particles[i, j*4 + 2] = np.random.uniform(*initial_state_ranges['vx'])
                self.particles[i, j*4 + 3] = np.random.uniform(*initial_state_ranges['vy'])
                # if y is negative
                if self.particles[i, j*4 + 1] < 0:
                    self.particles[i, j*4 + 1] = 0.0

    def _predict_single_ball(self, state, dt):
        """
        Predicts the next state of a single ball based on physics and adds process noise.
        """
        x, y, vx, vy = state

        # Predict next state using physics 
        x_pred = x + vx * dt
        y_pred = y + vy * dt
        vx_pred = vx
        vy_pred = vy - g * dt

        # Apply process noise (random walk for each state component)
        x_pred += np.random.normal(0, self.process_noise_std_pos)
        y_pred += np.random.normal(0, self.process_noise_std_pos)
        vx_pred += np.random.normal(0, self.process_noise_std_vel)
        vy_pred += np.random.normal(0, self.process_noise_std_vel)

        # Simple ground bounce for particles
        if y_pred < 0:
            y_pred = abs(y_pred) # Reflect y position
            vy_pred = -vy_pred * 0.8 # Reflect and lose some energy (e.g., 20% damping)
            # Check if vy is very small
            if abs(vy_pred) < 0.1:
                vy_pred = 0.0
                vx_pred = vx_pred * 0.9 # horizontal friction 
                if abs(vx_pred) < 0.1: # If horizontal velocity is also very low, stop completely
                    vx_pred = 0.0

        return np.array([x_pred, y_pred, vx_pred, vy_pred])

    def predict(self, dt):
        # Based on motion model,predict next states
        
        for i in range(self.num_particles):
            for j in range(self.num_balls):
                ball_state_idx = slice(j * 4, (j + 1) * 4) # Slice to get [x, y, vx, vy] for ball j
                self.particles[i, ball_state_idx] = self._predict_single_ball(
                    self.particles[i, ball_state_idx], dt
                )

    def update(self, observations):
        """
        Updates particle weights based on observations.
        Handles indistinguishable balls using data association (Hungarian algorithm).

        Args:
            observations (list of np.array): A list of observed [x, y] positions for the balls.
                                             Assumes len(observations) == self.num_balls.
        """
        if observations is None or len(observations) == 0:
            # If no observations, weights remain unchanged. 
            return

        # Check observations are a NumPy array for easier indexing
        observations_np = np.array(observations)

        # Variance for likelihood calculation (assuming independent x and y noise)
        obs_var = self.observation_noise_std**2

        new_weights = np.zeros(self.num_particles)

        for i in range(self.num_particles):
            particle_ball_positions = []
            for j in range(self.num_balls):
                # Extract predicted [x, y] position for each ball in this particle
                particle_ball_positions.append(self.particles[i, j*4 : j*4 + 2])

            particle_ball_positions_np = np.array(particle_ball_positions)

            # --- Data Association for Indistinguishable Balls ---
            #  Create a cost matrix where cost[i, j] is the squared distance
            cost_matrix = np.zeros((self.num_balls, self.num_balls))
            for p_idx in range(self.num_balls):
                for o_idx in range(self.num_balls):
                    diff = particle_ball_positions_np[p_idx] - observations_np[o_idx]
                    cost_matrix[p_idx, o_idx] = np.sum(diff**2) # Squared Euclidean distance

            # Hungarian algorithm to minimize total cost 
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Calculate likelihood for this particle based on the optimal assignment
            particle_likelihood = 1.0
            # Iterate through the matched pairs of predicted and observed positions
            for r, c in zip(row_ind, col_ind):
                predicted_pos = particle_ball_positions_np[r]
                observed_pos = observations_np[c]

                # Calculate the likelihood contribution for this matched pair using Gaussian PDF
                diff = predicted_pos - observed_pos
                # For 2D independent Gaussian noise, likelihood is proportional to exp(-0.5 * (dx^2 + dy^2) / sigma^2)
                exponent = -0.5 * np.sum(diff**2) / obs_var
                likelihood_term = (1 / (2 * np.pi * obs_var)) * np.exp(exponent)
                particle_likelihood *= likelihood_term

            new_weights[i] = particle_likelihood

        # Normalize weights
        sum_weights = np.sum(new_weights)
        if sum_weights == 0:
            # This indicates that all particles have extremely low likelihoods
            # To prevent filter collapse, reset weights to uniform otherwise it misconfigures noise 
            print("Warning: All particle weights are zero. Resetting to uniform weights.")
            self.weights = np.ones(self.num_particles) / self.num_particles
        else:
            self.weights = new_weights / sum_weights

        # Resample particles
        self._resample()

    def _resample(self):
        
        #Resamples particles based on their weights using systematic resampling.
    
        if np.sum(self.weights) == 0:
            # If for some reason weights are all zero before resampling (should be caught by update),
            # just re-initialize to uniform to prevent errors.
            self.weights = np.ones(self.num_particles) / self.num_particles

        # Generate new particle indices based on weights
        # np.random.choice with p=self.weights handles the probability distribution
        # replace=True means we can pick the same particle multiple times (which is the point of resampling)
        indices = np.random.choice(
            a=np.arange(self.num_particles),
            size=self.num_particles,
            replace=True,
            p=self.weights
        )

        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles # Reset weights to uniform after resampling

    def estimate_ball_positions(self):
        """
        Estimates the position and velocity of each ball by clustering the particles.
        Returns the mean of each cluster as the estimated state.

        """
        if self.num_particles == 0 or self.num_balls == 0:
            return []

        all_ball_states_flat = self.particles.reshape(-1, 4)

        # If there are fewer effective data points (individual ball states) than
        # the number of balls we want to cluster, clustering won't work well.
        if all_ball_states_flat.shape[0] < self.num_balls:
             # Fallback: if we don't have enough data for distinct clusters,
             # just return the overall mean (less accurate for distinct balls).
             mean_overall_state = np.mean(all_ball_states_flat, axis=0)
             return [mean_overall_state] * self.num_balls

        return self._cluster_ball_states_kmeans(all_ball_states_flat)

    def _cluster_ball_states_kmeans(self, all_ball_states_flat):
        """
        Cluster the individual ball states using KMeans and return the estimated states.

        """
        positions_flat = all_ball_states_flat[:, :2]
        try:
            kmeans = KMeans(n_clusters=self.num_balls, random_state=0, n_init='auto')
            kmeans.fit(positions_flat)
            cluster_labels = kmeans.labels_

            estimated_states = []
            for k in range(self.num_balls):
                indices_in_cluster = np.where(cluster_labels == k)[0]
                if len(indices_in_cluster) > 0:
                    mean_state_in_cluster = np.mean(all_ball_states_flat[indices_in_cluster], axis=0)
                    estimated_states.append(mean_state_in_cluster)
                else:
                    print(f"Warning: KMeans created an empty cluster {k}. Using center position with zero velocity.")
                    estimated_states.append(np.array([
                        kmeans.cluster_centers_[k, 0], kmeans.cluster_centers_[k, 1], 0.0, 0.0
                    ]))
            estimated_states.sort(key=lambda s: s[0])
            return estimated_states
        except Exception as e:
            print(f"Error during KMeans clustering: {e}. Returning mean of all particles as fallback.")
            mean_overall_state = np.mean(all_ball_states_flat, axis=0)
            return [mean_overall_state] * self.num_balls

    def _cluster_ball_states_dbscan(self, all_ball_states_flat, eps=0.5, min_samples=50):
        """
        Cluster the individual ball states using DBSCAN and return the estimated states.

        (NOT USED DUE TO HIGH COMPUTATION COST)

        """
        positions_flat = all_ball_states_flat[:, :2]
        try:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(positions_flat)

            estimated_states = []
            unique_labels = set(cluster_labels)
            unique_labels.discard(-1)  # Remove noise label

            for label in unique_labels:
                indices_in_cluster = np.where(cluster_labels == label)[0]
                if len(indices_in_cluster) > 0:
                    mean_state_in_cluster = np.mean(all_ball_states_flat[indices_in_cluster], axis=0)
                    estimated_states.append(mean_state_in_cluster)
            # Optionally sort by x for consistency
            estimated_states.sort(key=lambda s: s[0])
            return estimated_states
        except Exception as e:
            print(f"Error during DBSCAN clustering: {e}. Returning mean of all particles as fallback.")
            mean_overall_state = np.mean(all_ball_states_flat, axis=0)
            return [mean_overall_state]

    def get_particle_states(self):
        """
        Returns a copy of the current particle states for animation.
        """
        return self.particles.copy()