import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrow
from matplotlib.collections import LineCollection
from cyberrunner_env_vision import (
    CyberRunnerEnv,
    BOARD_WIDTH,
    BOARD_HEIGHT,
    HOLE_RADIUS,
    WALL_RADIUS
)

#matplotlib.use('Qt6Agg')


def draw_board(ax, env):
    """Draw the static board elements: walls, holes, path."""
    ax.clear()
    
    # Set board boundaries
    ax.set_xlim(-0.01, BOARD_WIDTH + 0.01)
    ax.set_ylim(-0.01, BOARD_HEIGHT + 0.01)
    ax.set_aspect('equal')
    ax.set_facecolor('#e0e0e0')
    
    # Draw board outline
    board_rect = Rectangle((0, 0), BOARD_WIDTH, BOARD_HEIGHT, 
                            fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(board_rect)
    
    # Draw horizontal walls
    for wall in env.walls_h:
        x_start, x_end, y = wall
        ax.plot([x_start, x_end], [y, y], color='#e64a19', linewidth=3, solid_capstyle='round')
    
    # Draw vertical walls
    for wall in env.walls_v:
        y_start, y_end, x = wall
        ax.plot([x, x], [y_start, y_end], color='#e64a19', linewidth=3, solid_capstyle='round')
    
    # Draw holes
    for hole in env.holes:
        circle = Circle(hole, HOLE_RADIUS, color='#1a1a1a', zorder=2)
        ax.add_patch(circle)
    
    # Draw path (waypoints)
    path_x = env.waypoints[:, 0]
    path_y = env.waypoints[:, 1]
    ax.plot(path_x, path_y, 'g-', linewidth=1.5, alpha=0.6, label='Path')
    
    # Mark start and goal
    ax.plot(env.waypoints[0, 0], env.waypoints[0, 1], 'go', markersize=10, label='Start')
    ax.plot(env.waypoints[-1, 0], env.waypoints[-1, 1], 'r*', markersize=15, label='Goal')


def draw_observation(ax, obs, env):
    """Draw the marble and observation vectors."""
    # Extract from observation
    states = obs["states"]
    joint_angles = states[0:2]
    ball_pos = states[2:4]
    vec_to_closest = states[4:6]
    vec_to_next_wp = states[6:8]
    vec_to_next_next_wp = states[8:10]
    
    # Draw marble (outline only so path is visible)
    marble = Circle(ball_pos, 0.0063, facecolor='none', edgecolor='#1976d2', linewidth=2, zorder=5)
    ax.add_patch(marble)
    
    # Draw vector to closest path point
    closest_point = ball_pos + vec_to_closest
    ax.annotate('', xy=closest_point, xytext=ball_pos,
                arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                zorder=4)
    ax.plot(closest_point[0], closest_point[1], 'b.', markersize=8, zorder=4)
    
    # Draw vector to next waypoint
    next_wp = ball_pos + vec_to_next_wp
    ax.annotate('', xy=next_wp, xytext=ball_pos,
                arrowprops=dict(arrowstyle='->', color='orange', lw=2),
                zorder=4)
    ax.plot(next_wp[0], next_wp[1], 'o', color='orange', markersize=8, zorder=4)
    
    # Draw vector to waypoint after next
    next_next_wp = ball_pos + vec_to_next_next_wp
    ax.annotate('', xy=next_next_wp, xytext=ball_pos,
                arrowprops=dict(arrowstyle='->', color='purple', lw=1.5, linestyle='--'),
                zorder=4)
    ax.plot(next_next_wp[0], next_next_wp[1], 'o', color='purple', markersize=6, zorder=4)
    
    return joint_angles, ball_pos


def main():
    # Initialize environment with vision
    env = CyberRunnerEnv(render_mode=None, randomize_init_pos=True, include_vision=True)

    # Set up the figure with two panels
    fig, (ax, ax_img) = plt.subplots(1, 2, figsize=(14, 8),
                                      gridspec_kw={'width_ratios': [3, 1]})
    ax_img.set_title("Vision (64x64)")
    ax_img.axis('off')
    img_handle = None
    plt.ion()  # Interactive mode

    # Reset environment
    obs, info = env.reset()
    step = 0
    total_reward = 0
    
    print("Click on the figure to advance to the next step.")
    print("Close the window to exit.")
    print("-" * 50)
    
    while True:
        # Draw everything
        draw_board(ax, env)
        joint_angles, ball_pos = draw_observation(ax, obs, env)

        # Update vision image
        if "image" in obs:
            if img_handle is None:
                img_handle = ax_img.imshow(obs["image"])
            else:
                img_handle.set_data(obs["image"])
            ax_img.set_title("Vision (64x64)")
        
        # Add info text
        ax.set_title(f"Step: {step} | Progress: {info['path_progress']:.2f} | Reward: {total_reward:.3f}\n"
                     f"Joints: α={np.degrees(joint_angles[0]):.1f}° β={np.degrees(joint_angles[1]):.1f}° | "
                     f"Ball: ({ball_pos[0]:.3f}, {ball_pos[1]:.3f})")
        
        # Legend
        ax.legend(loc='upper right', fontsize=8)
        
        # Add arrow legend
        ax.text(0.02, 0.02, 
                "Arrows: Blue=closest path | Orange=next WP | Purple=WP+1",
                transform=ax.transAxes, fontsize=8, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.draw()
        plt.pause(0.01)
        
        # Wait for click
        if not plt.waitforbuttonpress():
            # Mouse click - advance
            pass
        
        # Check if window was closed
        if not plt.fignum_exists(fig.number):
            break
        
        # Take random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        
        if terminated or truncated:
            print(f"\nEpisode ended at step {step}")
            print(f"Reason: {info.get('termination_reason', 'unknown')}")
            print(f"Total reward: {total_reward:.4f}")
            print("\nClick to reset...")
            
            plt.waitforbuttonpress()
            
            if not plt.fignum_exists(fig.number):
                break
                
            # Reset
            obs, info = env.reset()
            step = 0
            total_reward = 0
            print("-" * 50)
            print("New episode started. Click to advance.")
    
    env.close()
    plt.close()


if __name__ == "__main__":
    main()