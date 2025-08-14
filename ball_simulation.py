import numpy as np
g = 9.81

class BallSimulation:
    # Simulating the motion of multiple balls under projectile motion physics.
 
    def __init__(self, ball_params, dt_simulation):
        """
        Initializing the ball simulation.
        ball_params(dict list) for each ball.
        dt_simlation -time step.
        """
        self.dt = dt_simulation
        self.num_balls = len(ball_params)
        self.balls = [] # Stores the true state [x, y, vx, vy] of each ball
        self.current_time = 0.0

        for params in ball_params:
            angle_rad = np.deg2rad(params['angle_deg'])
            vx0 = params['speed'] * np.cos(angle_rad)
            vy0 = params['speed'] * np.sin(angle_rad)
            # Initialize each ball's state with position  and  (vx0, vy0)
            self.balls.append(np.array([params['x0'], params['y0'], vx0, vy0], dtype=float))

    def _update_ball_state(self, state):
        """
        Updates the state of a single ball based on its current state and physics.
        """
        x, y, vx, vy = state

        # update postion with current velocity
        x_new = x + vx * self.dt
        y_new = y + vy * self.dt

        vy_new = vy - g * self.dt  #(effected by gravity)
        vx_new = vx                     

        # check y position(above ground)
        if y_new < 0:# If the ball has fallen below ground level
            y_new = 0.0 
            vy_new = -vy_new * 0.8 # Bounce back with some energy loss (80% rebound)
            #  check if vy is very small
            if abs(vy_new) < 0.5: # If vertical velocity is very low, stop bouncing
                vy_new = 0.0
                vx_new = vx_new * 0.9 # Apply some horizontal friction as well
                if abs(vx_new) < 0.1: # If horizontal velocity very low stop bouncing
                    vx_new = 0.0

        return np.array([x_new, y_new, vx_new, vy_new])

    def step(self):
        
        #one time step of the simulation.
        
        for i in range(self.num_balls):
            self.balls[i] = self._update_ball_state(self.balls[i])
        self.current_time += self.dt

    def get_current_states(self):
        
         #Current states of each ball [x,y,vx,vy].
       
        return [ball.copy() for ball in self.balls]

    def get_observed_positions(self, noise_std=0.5):
        """
        noisy observations of current ball positions.

        Args:
             Standard deviation(gaussian noise)
        """
        observed_positions = []
        for ball_state in self.balls:
            x_true, y_true, _, _ = ball_state
            # Add Gaussian noise
            x_observed = x_true + np.random.normal(0, noise_std)
            y_observed = y_true + np.random.normal(0, noise_std)
            observed_positions.append(np.array([x_observed, y_observed]))
        return observed_positions