import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import sys

def plot_results(true_trajectories, observations, estimated_trajectories, dt_simulation):
    """
        true_trajectories : True states of all balls over time.
                                                     list of [x, y, vx, vy] arrays
                                                    for each ball at that time step.
        observations : Observed positions over time.
                                                        Every element is a list of [x, y] arrays
                                                        for each observed ball, or None if no observation.
        estimated_trajectories : Estimated states of all balls over time.
                                                            every element is a list of [x, y, vx, vy] arrays
                                                            for each estimated ball at that time step.
        dt_simulation (float): The time step  simulation
    """
    if not true_trajectories:
        print("No true trajectories to plot.")
        return
    if not estimated_trajectories:
        print("No estimated trajectories to plot.")
        return

    # Assuming all elements in true_trajectories have the same number of balls
    num_balls = len(true_trajectories[0]) if true_trajectories else 0

    # Generate time axis
    time_steps = np.arange(len(true_trajectories)) * dt_simulation

    # --- Plot Trajectories ---
    plt.figure(figsize=(12, 8))
    plt.title('Ball Trajectory Estimation with Particle Filter')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.grid(True)
    plt.axis('equal') # scaling for x and y axes

    # Sort true and estimated positions by x at each time step for consistent plotting
    for t in range(len(true_trajectories)):
        true_trajectories[t] = sorted(true_trajectories[t], key=lambda s: s[0])
        if t < len(estimated_trajectories) and len(estimated_trajectories[t]) == num_balls:
            estimated_trajectories[t] = sorted(estimated_trajectories[t], key=lambda s: s[0])


    # True Trajectories
    true_ball_data = [[] for _ in range(num_balls)]
    for time_step_data in true_trajectories:
        for i, ball_state in enumerate(time_step_data):
            true_ball_data[i].append(ball_state[:2]) # Just x, y

    for i in range(num_balls):
        if true_ball_data[i]: # Check if data exists for this ball
            true_x = [pos[0] for pos in true_ball_data[i]]
            true_y = [pos[1] for pos in true_ball_data[i]]
            plt.plot(true_x, true_y, linestyle='--', color=f'C{i}', label=f'True Ball {i+1}')

    #Estimated Trajectories
    estimated_ball_data = [[] for _ in range(num_balls)]
    for time_step_data in estimated_trajectories:
        # check estimated_trajectories has enough estimates for all balls.
        if len(time_step_data) == num_balls:
            for i, ball_state in enumerate(time_step_data):
                estimated_ball_data[i].append(ball_state[:2])
        else:
            # we fill with NaNs or simply stop processing for clarity in this plot.
            # For simplicity here, we'll just break this inner loop.
            # A more robust solution might try to match based on proximity.
            print(f"Warning: Estimated {len(time_step_data)} balls at time "
                  f"{time_steps[len(estimated_ball_data[0])] if estimated_ball_data[0] else 'N/A'}. "
                  f"Expected {num_balls}. Plotting partial estimates.")
            # Fill remaining estimated_ball_data for this time step with NaNs to keep lists same length
            for i in range(len(time_step_data), num_balls):
                estimated_ball_data[i].append(np.array([np.nan, np.nan]))


    for i in range(num_balls):
        if estimated_ball_data[i]: # Ensure there's data to plot
            est_x = [pos[0] for pos in estimated_ball_data[i]]
            est_y = [pos[1] for pos in estimated_ball_data[i]]
            # Filter out NaNs for plotting to avoid lines connecting gaps
            valid_indices = ~np.isnan(est_x)
            plt.plot(np.array(est_x)[valid_indices], np.array(est_y)[valid_indices],
                     linestyle='-', color=f'C{i}', label=f'Estimated Ball {i+1}', linewidth=2, alpha=0.8)

    # Plot Observations
    obs_x_data = []
    obs_y_data = []
    for obs_t in observations:
        if obs_t is not None:
            for obs_ball_pos in obs_t:
                obs_x_data.append(obs_ball_pos[0])
                obs_y_data.append(obs_ball_pos[1])
    plt.scatter(obs_x_data, obs_y_data, color='gray', marker='x', s=15, label='Observations', alpha=0.5, zorder=5)


    # Create a single legend entry for all true, estimated, and observed points
    # This ensures that each ball's true and estimated trajectories are labeled correctly
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles)) # Use a dict to remove duplicate labels if any
    plt.legend(unique_labels.values(), unique_labels.keys(), loc='upper right')
    plt.show()

    #  Position Errors Over Time ---
    plt.figure(figsize=(12, 6))
    plt.title('Estimated Position Error Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Position Error (m)')
    plt.grid(True)

    for i in range(num_balls):
        errors = []
        # Iterate up to the minimum length of true_trajectories and estimated_trajectories
        for t_idx in range(min(len(true_trajectories), len(estimated_trajectories))):
            true_pos = true_trajectories[t_idx][i][:2] # Get (x, y)
            # Check if estimated_trajectories has enough data for this ball at this time step
            if len(estimated_trajectories[t_idx]) > i and not np.any(np.isnan(estimated_trajectories[t_idx][i])):
                est_pos = estimated_trajectories[t_idx][i][:2] # Get (x, y)
                error = np.linalg.norm(true_pos - est_pos) # Euclidean distance
                errors.append(error)
            else:
                errors.append(np.nan) # Append NaN if estimate is missing or invalid for this ball at this time

        # actual erros
        if errors and not np.all(np.isnan(errors)):
            plt.plot(time_steps[:len(errors)], errors, label=f'Position Error Ball {i+1}')
    plt.legend()
    plt.show()


def animate_simulation_pygame(true_trajectories, particle_states, estimated_trajectories, dt_simulation, width=800, height=600):
    """
    Animates true motion of the balls and  state of the particles by pygame,
    with play/pause and timeline controls.

    Controls:
        - Spacebar: Play/Pause
        - Left/Right Arrow: Step frame backward/forward (when paused)
        - Mouse drag on timeline: Jump to frame
        - Close window: Exit
    """
    if not true_trajectories or not particle_states:
        print("No data to animate.")
        return

    # Find bounds for scaling
    all_x = []
    all_y = []
    for frame in true_trajectories:
        for ball in frame:
            all_x.append(ball[0])
            all_y.append(ball[1])
    for frame in particle_states:
        all_x.extend(frame[:, 0])
        all_y.extend(frame[:, 1])
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    def to_screen(x, y):
        # Scale simulation coordinates to screen coordinates
        sx = int((x - min_x) / (max_x - min_x + 1e-6) * (width - 40) + 20)
        sy = int(height - ((y - min_y) / (max_y - min_y + 1e-6) * (height - 80) + 40))  # leave space for timeline
        return sx, sy

    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Ball Motion and Particle States (pygame)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    num_balls = len(true_trajectories[0])
    ball_colors = [(255, 0, 0), (0, 128, 255), (0, 200, 0), (255, 128, 0), (128, 0, 255)]

    running = True
    playing = True
    frame = 0
    total_frames = min(len(true_trajectories), len(particle_states))

    # Timeline bar settings
    timeline_height = 30
    timeline_y = height - timeline_height
    timeline_margin = 40
    timeline_width = width - 2 * timeline_margin
    handle_radius = 8
    dragging = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    playing = not playing
                elif event.key == pygame.K_RIGHT:
                    if not playing:
                        frame = min(frame + 1, total_frames - 1)
                elif event.key == pygame.K_LEFT:
                    if not playing:
                        frame = max(frame - 1, 0)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                # Check if mouse is on the timeline
                if timeline_y <= my <= timeline_y + timeline_height:
                    dragging = True
                    # Set frame based on mouse x
                    rel_x = min(max(mx - timeline_margin, 0), timeline_width)
                    frame = int(rel_x / timeline_width * (total_frames - 1))
            elif event.type == pygame.MOUSEBUTTONUP:
                dragging = False
            elif event.type == pygame.MOUSEMOTION and dragging:
                mx, my = event.pos
                rel_x = min(max(mx - timeline_margin, 0), timeline_width)
                frame = int(rel_x / timeline_width * (total_frames - 1))

        if playing and not dragging:
            frame += 1
            if frame >= total_frames:
                frame = total_frames - 1
                playing = False  # Pause at the end

        screen.fill((255, 255, 255))

        # Draw particles for all balls in each particle
        particles = particle_states[frame]
        for p in particles:
            for b in range(num_balls):
                px = p[4 * b + 0]
                py = p[4 * b + 1]
                sx, sy = to_screen(px, py)
                pygame.draw.circle(screen, (180, 180, 180), (sx, sy), 2)

        # Draw balls
        for i, ball in enumerate(true_trajectories[frame]):
            bx, by = to_screen(ball[0], ball[1])
            color = ball_colors[i % len(ball_colors)]
            pygame.draw.circle(screen, color, (bx, by), 10)

        # Draw estimated ball positions (if available for this frame) as translucent circles over true positions
        if frame < len(estimated_trajectories):
            for i, est in enumerate(estimated_trajectories[frame]):
                ex, ey = to_screen(est[0], est[1])
                color = ball_colors[i % len(ball_colors)] + (120,)  # Add alpha for translucency
                s = pygame.Surface((width, height), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (ex, ey), 10)
                screen.blit(s, (0, 0))

        # Draw translucent estimated trajectories for each ball (thin line)
        if frame > 1:
            for i in range(num_balls):
                est_points = []
                true_points = []
                for f in range(frame):
                    # Estimated trajectory points
                    if f < len(estimated_trajectories) and len(estimated_trajectories[f]) > i:
                        est = estimated_trajectories[f][i]
                        ex, ey = to_screen(est[0], est[1])
                        est_points.append((ex, ey))
                    # True trajectory points
                    if f < len(true_trajectories) and len(true_trajectories[f]) > i:
                        true_ball = true_trajectories[f][i]
                        tx, ty = to_screen(true_ball[0], true_ball[1])
                        true_points.append((tx, ty))
                # Estimated trajectory 
                if len(est_points) > 1:
                    color = ball_colors[i % len(ball_colors)] + (80,)  # Add alpha for translucency
                    s = pygame.Surface((width, height), pygame.SRCALPHA)
                    pygame.draw.lines(s, color, False, est_points, 1)
                    screen.blit(s, (0, 0))
                # True trajectory 
                if len(true_points) > 1:
                    color = (0, 0, 0, 120)  # Black with some alpha
                    s = pygame.Surface((width, height), pygame.SRCALPHA)
                    # Draw dotted line by drawing small circles along the path
                    for idx in range(0, len(true_points), 4):
                        pygame.draw.circle(s, color, true_points[idx], 2)
                    screen.blit(s, (0, 0))

        # Draw timeline bar
        pygame.draw.rect(screen, (220, 220, 220), (timeline_margin, timeline_y + timeline_height//2 - 4, timeline_width, 8))
        # Draw handle
        handle_x = int(timeline_margin + (frame / (total_frames - 1)) * timeline_width)
        handle_y = timeline_y + timeline_height // 2
        pygame.draw.circle(screen, (50, 50, 200), (handle_x, handle_y), handle_radius)
        # Draw frame/time text
        time_text = f"Frame: {frame+1}/{total_frames}   Time: {frame*dt_simulation:.2f}s"
        text_surf = font.render(time_text, True, (0, 0, 0))
        screen.blit(text_surf, (timeline_margin, timeline_y - 25))

        # Draw play/pause indicator
        if playing:
            pygame.draw.polygon(screen, (0, 200, 0), [(width-35, timeline_y+8), (width-15, timeline_y+15), (width-35, timeline_y+22)])
        else:
            pygame.draw.rect(screen, (200, 0, 0), (width-35, timeline_y+8, 6, 14))
            pygame.draw.rect(screen, (200, 0, 0), (width-25, timeline_y+8, 6, 14))

        pygame.display.flip()
        clock.tick(60) 

    pygame.quit()
    sys.exit()