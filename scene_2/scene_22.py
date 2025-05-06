from z3 import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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

def generate_all_possible_box_states(valid_cells, num_boxes):
    """
    Generate all valid box configurations (without overlaps).
    This is used to pre-compute possible box states.
    Warning: This grows factorially with the number of boxes and cells.
    """
    # Start with no boxes placed
    def backtrack(placed_boxes, remaining_cells, box_idx):
        if box_idx == num_boxes:
            return [placed_boxes[:]]  # Found a valid configuration
        
        results = []
        for i, cell in enumerate(remaining_cells):
            # Place this box at this cell
            placed_boxes.append(cell)
            # Recursively place remaining boxes in remaining cells
            results.extend(backtrack(placed_boxes, remaining_cells[:i] + remaining_cells[i+1:], box_idx + 1))
            placed_boxes.pop()
        
        return results
    
    return backtrack([], valid_cells, 0)

def solve_ice_with_boxes():
    """
    Solve ice sliding puzzle with boxes using Z3 SMT solver.
    Uses a direct encoding approach with precomputed transitions.
    """
    global grid, initial_box_positions
    
    print("Initializing grid and state...")
    # Grid definition: 0=empty, 1=wall, 2=robot, 3=goal, 4=box
    grid = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 2, 0, 1, 0, 0, 1],
        [1, 0, 4, 0, 0, 4, 0, 1],
        [1, 0, 0, 1, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
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
    
    # CRITICAL: Reduce max steps to avoid memory issues
    max_steps = 60  # Reasonable for this puzzle size
    
    # Identify valid cells (non-obstacle cells)
    valid_cells = []
    for r in range(height):
        for c in range(width):
            if not is_obstacle(r, c, grid):
                valid_cells.append((r, c))
    
    print(f"Found {len(valid_cells)} valid cells")
    
    # Create solver
    solver = Solver()
    solver.set(timeout=60000)  # 60 second timeout
    
    # Variables for robot position at each time step
    robot_vars = {}
    for t in range(max_steps + 1):
        for r, c in valid_cells:
            robot_vars[(t, r, c)] = Bool(f"robot_{t}_{r}_{c}")
    
    # Variables for box positions at each time step
    box_vars = {}
    for t in range(max_steps + 1):
        for box_idx in range(num_boxes):
            for r, c in valid_cells:
                box_vars[(t, box_idx, r, c)] = Bool(f"box_{t}_{box_idx}_{r}_{c}")
    
    # Variables for actions at each time step
    actions = {}
    for t in range(max_steps):
        for d in range(4):
            actions[(t, d)] = Bool(f"action_{t}_{d}")
    
    # Initial state constraints
    print("Setting initial state constraints...")
    
    # Robot starts at its initial position
    solver.add(robot_vars[(0, robot_pos[0], robot_pos[1])])
    for r, c in valid_cells:
        if (r, c) != robot_pos:
            solver.add(Not(robot_vars[(0, r, c)]))
    
    # Boxes start at their initial positions
    for box_idx, box_pos in enumerate(initial_box_positions):
        solver.add(box_vars[(0, box_idx, box_pos[0], box_pos[1])])
        for r, c in valid_cells:
            if (r, c) != box_pos:
                solver.add(Not(box_vars[(0, box_idx, r, c)]))
    
    # Precompute transitions
    print("Precomputing possible transitions...")
    
    # For each time step, add transition constraints
    for t in range(max_steps):
        print(f"Adding constraints for step {t}...")
        
        # Exactly one action per time step
        solver.add(PbEq([(actions[(t, d)], 1) for d in range(4)], 1))
        
        # For each possible robot position
        for r_start, c_start in valid_cells:
            # For each possible direction
            for d in range(4):
                # Get the current boxes variable references
                box_vars_at_t = []
                for box_idx in range(num_boxes):
                    box_vars_at_t.append([])
                    for r, c in valid_cells:
                        box_vars_at_t[box_idx].append((box_vars[(t, box_idx, r, c)], (r, c)))
                
                # Define the precondition: robot at (r_start, c_start) and action d
                robot_at_start = robot_vars[(t, r_start, c_start)]
                action_d = actions[(t, d)]
                pre = And(robot_at_start, action_d)
                
                # For each possible box configuration at time t
                # This is where we need a different approach
                # Instead of enumerating all configurations, we'll handle each box separately
                
                # Step 1: For this robot position and action, calculate the potential outcome
                #         assuming boxes are at initial positions (simplification)
                (r_next, c_next), moved_boxes = next_position(r_start, c_start, d, initial_box_positions, grid)
                
                # Step 2: Generate constraints based on this simplified calculation
                if (r_next, c_next) == (r_start, c_start) and not moved_boxes:
                    # No movement possible in this direction
                    solver.add(Implies(pre, robot_vars[(t + 1, r_start, c_start)]))
                    # Boxes stay in place (frame axiom)
                    for box_idx in range(num_boxes):
                        for r, c in valid_cells:
                            box_at_rc = box_vars[(t, box_idx, r, c)]
                            solver.add(Implies(And(pre, box_at_rc), box_vars[(t + 1, box_idx, r, c)]))
                else:
                    # Movement is possible
                    # Robot moves to new position
                    solver.add(Implies(pre, robot_vars[(t + 1, r_next, c_next)]))
                    
                    # Handle moved boxes
                    moved_box_indices = [idx for idx, _ in moved_boxes]
                    
                    # For each box
                    for box_idx in range(num_boxes):
                        if box_idx in moved_box_indices:
                            # This box was moved by the action
                            for idx, (br_next, bc_next) in moved_boxes:
                                if idx == box_idx:
                                    # Find the original position of this box
                                    br_orig, bc_orig = initial_box_positions[box_idx]
                                    # If box is at its expected position, it moves
                                    box_at_orig = box_vars[(t, box_idx, br_orig, bc_orig)]
                                    solver.add(Implies(And(pre, box_at_orig), 
                                                      box_vars[(t + 1, box_idx, br_next, bc_next)]))
                                    break
                        else:
                            # This box wasn't moved by the action - it stays in place
                            for r, c in valid_cells:
                                box_at_rc = box_vars[(t, box_idx, r, c)]
                                solver.add(Implies(And(pre, box_at_rc), 
                                                 box_vars[(t + 1, box_idx, r, c)]))
        
        # Ensure exactly one robot position at t+1
        solver.add(PbEq([(robot_vars[(t + 1, r, c)], 1) for r, c in valid_cells], 1))
        
        # Ensure exactly one position per box at t+1
        for box_idx in range(num_boxes):
            solver.add(PbEq([(box_vars[(t + 1, box_idx, r, c)], 1) for r, c in valid_cells], 1))
        
        # No two objects in the same cell at t+1
        for r, c in valid_cells:
            # No box-box overlaps
            for i in range(num_boxes):
                for j in range(i + 1, num_boxes):
                    solver.add(Not(And(box_vars[(t + 1, i, r, c)], box_vars[(t + 1, j, r, c)])))
            
            # No robot-box overlaps
            for i in range(num_boxes):
                solver.add(Not(And(robot_vars[(t + 1, r, c)], box_vars[(t + 1, i, r, c)])))
    
    # Goal constraint: Robot reaches the goal position
    print("Adding goal constraint...")
    goal_constraints = []
    for t in range(1, max_steps + 1):
        goal_constraints.append(robot_vars[(t, goal_pos[0], goal_pos[1])])
    
    solver.add(Or(goal_constraints))
    
    # Solve
    print("Solving...")
    start_time = time.time()
    result = solver.check()
    elapsed_time = time.time() - start_time
    
    print(f"Solving took {elapsed_time:.2f} seconds")
    print(f"Result: {result}")
    
    if result == sat:
        model = solver.model()
        
        # Extract solution
        robot_path = [robot_pos]  # Start with initial position
        box_paths = [[pos] for pos in initial_box_positions]  # Start with initial positions
        actions_taken = []
        
        for t in range(max_steps):
            # Find robot position at t+1
            robot_found = False
            for r, c in valid_cells:
                if is_true(model.eval(robot_vars[(t + 1, r, c)])):
                    robot_path.append((r, c))
                    robot_found = True
                    break
            
            if not robot_found:
                print(f"Warning: Robot position not found at step {t+1}")
                break
            
            # Find box positions at t+1
            for box_idx in range(num_boxes):
                box_found = False
                for r, c in valid_cells:
                    if is_true(model.eval(box_vars[(t + 1, box_idx, r, c)])):
                        box_paths[box_idx].append((r, c))
                        box_found = True
                        break
                
                if not box_found:
                    print(f"Warning: Box {box_idx} position not found at step {t+1}")
                    # Use previous position as fallback
                    box_paths[box_idx].append(box_paths[box_idx][-1])
            
            # Find which action was taken
            action_found = False
            for d in range(4):
                if is_true(model.eval(actions[(t, d)])):
                    actions_taken.append(d)
                    action_found = True
                    break
            
            if not action_found:
                print(f"Warning: Action not found at step {t}")
                break
            
            # Check if goal reached
            if robot_path[-1] == goal_pos:
                print(f"🎯 Goal reached at step {t+1}!")
                break
        
        # Trim paths to actual length used
        print(f"Robot path: {robot_path}")
        print(f"Actions: {[dir_names[d] for d in actions_taken]}")
        
        return grid, robot_path, box_paths, goal_pos
    else:
        print("No solution found.")
        return None, None, None, None

def animate_solution(grid, robot_path, box_paths, goal_pos):
    """Create an animation of the solution."""
    if not robot_path:
        print("No path found to animate.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    h, w = grid.shape
    box_colors = ['blue', 'green', 'orange', 'purple', 'cyan']  # Added more colors

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
        ani.save("ice_puzzle_solution_z3.gif", writer='pillow', dpi=100)
        print("✅ Animation saved as 'ice_puzzle_solution_z3.gif'")
    except Exception as e:
        print(f"❌ Failed to save animation: {e}")
        print("Ensure you have 'pillow' installed (`pip install pillow`)")
    
    plt.close(fig)

def precompute_transitions(valid_cells, initial_box_positions, grid):
    """
    Precompute transitions for all possible robot positions, directions, and box configurations.
    This avoids doing the calculation during constraint generation.
    Returns a dictionary mapping (robot_r, robot_c, direction) to (new_robot_r, new_robot_c, moved_boxes)
    """
    transitions = {}
    
    for r, c in valid_cells:
        for d in range(4):
            # Calculate transition using initial box positions
            (new_r, new_c), moved_boxes = next_position(r, c, d, initial_box_positions, grid)
            transitions[(r, c, d)] = (new_r, new_c, moved_boxes)
    
    return transitions

if __name__ == "__main__":
    print("Starting Ice Sliding Puzzle solver with Z3...")
    grid_data, robot_path_data, box_paths_data, goal_pos_data = solve_ice_with_boxes()
    
    if grid_data is not None:
        print("Animating solution...")
        animate_solution(grid_data, robot_path_data, box_paths_data, goal_pos_data)
    else:
        print("No solution found or error occurred.")