import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from heapq import heappush, heappop
import time

# Global grid variable
grid = None
initial_box_positions = []

# Directions: Up, Right, Down, Left
directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
dir_names = ["up", "right", "down", "left"]

def is_obstacle(r, c, current_grid):
    """Check if a cell is an obstacle or out of bounds."""
    height, width = current_grid.shape
    if r < 0 or r >= height or c < 0 or c >= width or current_grid[r, c] == 1:
        return True
    return False

def calculate_stop_position(start_r, start_c, dr, dc, current_grid, current_box_positions, ignore_box_idx=None):
    """Helper function to find where an object (robot or box) stops sliding."""
    curr_r, curr_c = start_r, start_c
    while True:
        next_r, next_c = curr_r + dr, curr_c + dc

        # Check for grid boundaries or static walls
        if is_obstacle(next_r, next_c, current_grid):
            return (curr_r, curr_c)  # Stop before the obstacle

        # Check for collision with ANY box (unless we are ignoring one, e.g., the box being pushed)
        hit_another_box = False
        for idx, (br, bc) in enumerate(current_box_positions):
            if idx == ignore_box_idx:
                continue
            if (next_r, next_c) == (br, bc):
                hit_another_box = True
                break
        
        if hit_another_box:
            return (curr_r, curr_c)  # Stop before the other box

        # If no obstacle, continue sliding
        curr_r, curr_c = next_r, next_c

def next_position(robot_r, robot_c, direction_idx, current_box_positions_list, current_grid):
    """
    Calculates the final position of the robot and any moved boxes after ONE action.
    Args:
        robot_r, robot_c: Robot's starting position for the step.
        direction_idx: Index into the directions list (0: up, 1: right, etc.).
        current_box_positions_list: A list of tuples [(r1, c1), (r2, c2), ...] for current box positions.
        current_grid: The static grid map.
    Returns:
        tuple: ((final_robot_r, final_robot_c), list_of_moved_boxes)
               where list_of_moved_boxes is [(box_idx, (final_box_r, final_box_c)), ...]
    """
    dr, dc = directions[direction_idx]

    # 1. Calculate robot's immediate next step
    robot_next_r, robot_next_c = robot_r + dr, robot_c + dc

    # 2. Check for immediate wall hit
    if is_obstacle(robot_next_r, robot_next_c, current_grid):
        return (robot_r, robot_c), []  # Robot doesn't move

    # 3. Check for immediate box hit
    hit_box_idx = -1
    for idx, (br, bc) in enumerate(current_box_positions_list):
        if (robot_next_r, robot_next_c) == (br, bc):
            hit_box_idx = idx
            break

    # 4. Handle Box Hit
    if hit_box_idx != -1:
        box_start_r, box_start_c = robot_next_r, robot_next_c  # Box starts where robot tried to move

        # Check if the space BEYOND the box is blocked
        box_next_r, box_next_c = box_start_r + dr, box_start_c + dc

        blocked = False
        if is_obstacle(box_next_r, box_next_c, current_grid):
            blocked = True
        else:
            # Check collision with OTHER boxes
            for idx, (br, bc) in enumerate(current_box_positions_list):
                if idx != hit_box_idx and (box_next_r, box_next_c) == (br, bc):
                    blocked = True
                    break
        
        if blocked:
            # Box cannot move, so robot cannot move either
            return (robot_r, robot_c), []
        else:
            # Box can move. Robot stops where the box was. Box slides.
            final_robot_pos = (box_start_r, box_start_c)
            # Find where the pushed box stops sliding
            final_box_pos = calculate_stop_position(box_next_r, box_next_c, dr, dc, current_grid, 
                                                  current_box_positions_list, ignore_box_idx=hit_box_idx)
            
            return final_robot_pos, [(hit_box_idx, final_box_pos)]

    # 5. Handle Sliding into Empty Space
    else:
        # Robot slides until it hits something (wall or any box)
        final_robot_pos = calculate_stop_position(robot_next_r, robot_next_c, dr, dc, current_grid, 
                                                current_box_positions_list)
        return final_robot_pos, []

def solve_ice_with_boxes_astar():
    """Solve the ice and box problem using A* search."""
    global grid, initial_box_positions
    
    print("Initializing grid and state...")
    # Grid definition: 0=empty, 1=wall, 2=robot, 3=goal, 4=box
    grid = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 2, 0, 1, 0, 0, 1],
        [1, 0, 4, 0, 0, 4, 0, 1],
        [1, 0, 0, 1, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 4, 0, 1],
        [1, 0, 1, 0, 3, 0, 0, 1],
        [1, 0, 0, 1, 0, 4, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ])
    
    height, width = grid.shape
    
    # Find robot, goal, and box positions
    robot_pos = tuple(map(int, np.argwhere(grid == 2)[0]))
    goal_pos = tuple(map(int, np.argwhere(grid == 3)[0]))
    initial_box_positions = [tuple(map(int, pos)) for pos in np.argwhere(grid == 4)]
    num_boxes = len(initial_box_positions)
    
    print(f"Robot starts at: {robot_pos}")
    print(f"Goal position: {goal_pos}")
    print(f"Boxes at: {initial_box_positions}")
    
    # Initial state: (robot_r, robot_c, ((box1_r, box1_c), (box2_r, box2_c), ...))
    initial_state = (robot_pos[0], robot_pos[1], tuple(initial_box_positions))
    
    # Goal check function
    def is_goal(state):
        return (state[0], state[1]) == goal_pos
    
    # Heuristic function (Manhattan distance to goal)
    def heuristic(state):
        return abs(state[0] - goal_pos[0]) + abs(state[1] - goal_pos[1])
    
    # Get successors function using next_position logic
    def get_successors(state):
        robot_r, robot_c, box_positions = state
        successors = []
        
        for d in range(4):  # Try all four directions
            # Use next_position to calculate the outcome
            (new_r, new_c), moved_boxes = next_position(
                robot_r, robot_c, d, list(box_positions), grid)
            
            # If the robot position didn't change, this action didn't work
            if (new_r, new_c) == (robot_r, robot_c) and not moved_boxes:
                continue
                
            # Update box positions based on moved boxes
            new_box_positions = list(box_positions)
            for box_idx, new_pos in moved_boxes:
                new_box_positions[box_idx] = new_pos
                
            # Create new state
            new_state = (new_r, new_c, tuple(sorted(new_box_positions)))  # Sort for consistent hashing
            
            # Add to successors with action and cost
            successors.append((new_state, d, 1))  # state, action, cost
            
        return successors
    
    print("Starting A* search...")
    start_time = time.time()
    
    # Run A* search
    open_set = [(heuristic(initial_state), 0, initial_state, [])]  # (f, g, state, path)
    closed_set = set()
    
    # For tracking progress
    nodes_expanded = 0
    max_queue_size = 1
    
    while open_set:
        max_queue_size = max(max_queue_size, len(open_set))
        
        # Get node with lowest f-value
        f, g, current, path = heappop(open_set)
        
        # Convert state to hashable form for closed set
        state_hash = (current[0], current[1], current[2])
        
        # Skip if already visited
        if state_hash in closed_set:
            continue
            
        # Check if goal reached
        if is_goal(current):
            elapsed_time = time.time() - start_time
            print(f"Solution found! Path length: {len(path)}")
            print(f"Time taken: {elapsed_time:.2f} seconds")
            print(f"Nodes expanded: {nodes_expanded}")
            print(f"Max queue size: {max_queue_size}")
            
            # Convert path to required format for animation
            robot_path, box_paths = extract_paths(path, initial_state)
            return grid, robot_path, box_paths, goal_pos
            
        # Mark as visited
        closed_set.add(state_hash)
        nodes_expanded += 1
        
        # Print progress every 1000 nodes
        if nodes_expanded % 1000 == 0:
            print(f"Nodes expanded: {nodes_expanded}, Queue size: {len(open_set)}")
        
        # Explore successors
        for successor, action, cost in get_successors(current):
            succ_hash = (successor[0], successor[1], successor[2])
            if succ_hash not in closed_set:
                new_g = g + cost
                new_f = new_g + heuristic(successor)
                new_path = path + [action]
                
                heappush(open_set, (new_f, new_g, successor, new_path))
    
    print("No solution found!")
    return None, None, None, None  # No solution found

def extract_paths(action_path, initial_state):
    """
    Extract robot and box paths from the action path.
    Returns:
        - robot_path: List of (r, c) for robot positions
        - box_paths: List of lists of (r, c) for box positions
    """
    robot_r, robot_c, box_positions = initial_state
    robot_path = [(robot_r, robot_c)]
    num_boxes = len(box_positions)
    box_paths = [[(r, c)] for r, c in box_positions]
    
    current_boxes = list(box_positions)
    
    for action in action_path:
        (new_r, new_c), moved_boxes = next_position(robot_r, robot_c, action, current_boxes, grid)
        
        # Update robot position
        robot_r, robot_c = new_r, new_c
        robot_path.append((robot_r, robot_c))
        
        # Update box positions
        for box_idx, new_pos in moved_boxes:
            current_boxes[box_idx] = new_pos
        
        # Add current box positions to paths
        for i in range(num_boxes):
            box_paths[i].append(current_boxes[i])
    
    return robot_path, box_paths

def animate_solution(grid, robot_path, box_paths, goal_pos):
    """Create an animation of the solution."""
    if not robot_path:
        print("No path found to animate.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    h, w = grid.shape
    box_colors = ['blue', 'blue', 'blue', 'blue', 'blue']  # Added more colors

    max_frames = len(robot_path)

    def draw_frame(t):
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
        # Ensure limits cover the whole grid
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)  # Inverted Y for matrix display
        ax.set_aspect('equal')
        ax.set_title(f"Step {t}/{max_frames - 1}")

        # Draw grid cells (walls, goal, empty)
        for r in range(h):
            for c in range(w):
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor='gray', linewidth=0.5))
                if grid[r, c] == 1:  # Wall
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='black'))
                elif (r, c) == goal_pos:  # Goal
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='lime', alpha=0.5))

        # Draw boxes at current frame 't'
        for i, box_path in enumerate(box_paths):
            if t < len(box_path):  # Check if time step exists for this box
                br, bc = box_path[t]
                color = box_colors[i % len(box_colors)]
                # Draw box slightly smaller
                ax.add_patch(plt.Rectangle((bc - 0.4, br - 0.4), 0.8, 0.8, color=color, label=f'Box {i+1}' if t==0 else ""))

        # Draw robot at current frame 't'
        if t < len(robot_path):  # Check if time step exists
            rr, rc = robot_path[t]
            ax.add_patch(plt.Circle((rc, rr), 0.35, color='red', label='Robot' if t==0 else ""))

        if t == 0: 
            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), ncol=1)

    # Create animation
    ani = animation.FuncAnimation(fig, draw_frame, frames=max_frames, interval=500, repeat=True, repeat_delay=2000)
    
    try:
        ani.save("scene_2_astar.gif", writer='pillow', dpi=100)
        print("✅ Animation saved as 'scene_2_astar.gif'")
    except Exception as e:
        print(f"❌ Failed to save animation: {e}")
        print("Ensure you have 'pillow' installed (`pip install pillow`)")
    
    plt.close(fig)

if __name__ == "__main__":
    print("Starting Ice Sliding Puzzle solver...")
    grid_data, robot_path_data, box_paths_data, goal_pos_data = solve_ice_with_boxes_astar()
    
    if grid_data is not None:
        print("Animating solution...")
        animate_solution(grid_data, robot_path_data, box_paths_data, goal_pos_data)
    else:
        print("No solution found or error occurred.")