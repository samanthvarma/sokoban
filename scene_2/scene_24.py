from z3 import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

def solve_sokoban_on_ice():
    """
    Solve a Sokoban on Ice puzzle:
    - Robot slides until hitting an obstacle or box
    - If robot hits a box, the box slides until hitting an obstacle
    - Goal is for the robot to reach a target cell
    """
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
    robot_pos = tuple(map(int, np.argwhere(grid == 2)[0]))
    goal_pos = tuple(map(int, np.argwhere(grid == 3)[0]))
    box_positions = [tuple(map(int, pos)) for pos in np.argwhere(grid == 4)]
    num_boxes = len(box_positions)

    print("Robot:", robot_pos)
    print("Goal:", goal_pos)
    print("Boxes:", box_positions)
    print(f"Number of boxes: {num_boxes}")

    # Define movement directions: up, right, down, left
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    dir_names = ["up", "right", "down", "left"]
    max_steps = 30  # Increased for more complex puzzle

    def is_obstacle(r, c):
        """Check if a cell is a wall or out of bounds."""
        return r < 0 or r >= height or c < 0 or c >= width or grid[r, c] == 1

    def calculate_slide(r, c, dr, dc, box_positions, pushed_box_idx=None):
        """
        Calculate the final position after sliding in a direction.
        
        Args:
            r, c: Starting position
            dr, dc: Direction vector
            box_positions: Current positions of all boxes
            pushed_box_idx: Index of a box being pushed (to ignore in collision checks)
            
        Returns:
            (final_r, final_c): Final position after sliding
        """
        curr_r, curr_c = r, c
        
        while True:
            next_r, next_c = curr_r + dr, curr_c + dc
            
            # Check if we hit a wall
            if is_obstacle(next_r, next_c):
                return (curr_r, curr_c)
                
            # Check if we hit a box (except the one being pushed)
            for idx, (box_r, box_c) in enumerate(box_positions):
                if idx == pushed_box_idx:
                    continue
                if (next_r, next_c) == (box_r, box_c):
                    return (curr_r, curr_c)
                    
            # Continue sliding
            curr_r, curr_c = next_r, next_c

    def next_state(robot_r, robot_c, box_positions, direction_idx):
        """
        Calculate the next state after taking an action.
        
        Args:
            robot_r, robot_c: Robot position
            box_positions: List of box positions
            direction_idx: Direction index (0=up, 1=right, 2=down, 3=left)
            
        Returns:
            (new_robot_pos, new_box_positions): The new state
        """
        dr, dc = directions[direction_idx]
        
        # Calculate robot's next position
        robot_next_r, robot_next_c = robot_r + dr, robot_c + dc
        
        # Check if robot hits a wall immediately
        if is_obstacle(robot_next_r, robot_next_c):
            return (robot_r, robot_c), box_positions
            
        # Check if robot hits a box
        pushed_box_idx = -1
        for idx, (box_r, box_c) in enumerate(box_positions):
            if (robot_next_r, robot_next_c) == (box_r, box_c):
                pushed_box_idx = idx
                break
                
        # If robot hits a box, check if box can move
        if pushed_box_idx >= 0:
            box_r, box_c = box_positions[pushed_box_idx]
            box_next_r, box_next_c = box_r + dr, box_c + dc
            
            # Check if box hits a wall immediately
            if is_obstacle(box_next_r, box_next_c):
                return (robot_r, robot_c), box_positions
                
            # Check if box hits another box immediately
            for idx, (other_r, other_c) in enumerate(box_positions):
                if idx != pushed_box_idx and (box_next_r, box_next_c) == (other_r, other_c):
                    return (robot_r, robot_c), box_positions
                    
            # Box can move, so robot pushes it and both slide
            # Robot stops at box's original position
            new_robot_pos = (box_r, box_c)
            
            # Box slides from its new position
            box_final_pos = calculate_slide(box_next_r, box_next_c, dr, dc, box_positions, pushed_box_idx)
            
            # Create new box positions
            new_box_positions = list(box_positions)
            new_box_positions[pushed_box_idx] = box_final_pos
            
            return new_robot_pos, new_box_positions
            
        # If no box hit, robot slides until hitting a wall or box
        robot_final_pos = calculate_slide(robot_next_r, robot_next_c, dr, dc, box_positions)
        
        return robot_final_pos, box_positions

    # Precompute the state space
    print("Precomputing state space...")
    state_to_id = {}
    id_to_state = {}
    transitions = {}  # (state_id, direction) -> next_state_id
    
    # Initial state gets ID 0
    initial_state = (robot_pos, tuple(box_positions))
    state_to_id[initial_state] = 0
    id_to_state[0] = initial_state
    next_state_id = 1
    
    # Queue for BFS
    queue = [initial_state]
    visited = {initial_state}
    
    # BFS to build the state space
    while queue:
        current_state = queue.pop(0)
        current_id = state_to_id[current_state]
        
        # Try each direction
        for d in range(4):
            robot_pos, box_positions = current_state
            
            # Get next state
            new_robot_pos, new_box_positions = next_state(
                robot_pos[0], robot_pos[1], list(box_positions), d
            )
            
            # Create state
            next_state_tuple = (new_robot_pos, tuple(new_box_positions))
            
            # Only add if state changed
            if next_state_tuple != current_state:
                # Add to state mapping if new
                if next_state_tuple not in state_to_id:
                    state_to_id[next_state_tuple] = next_state_id
                    id_to_state[next_state_id] = next_state_tuple
                    next_state_id += 1
                    
                    # Add to queue if not visited
                    if next_state_tuple not in visited:
                        visited.add(next_state_tuple)
                        queue.append(next_state_tuple)
                        
                # Record transition
                transitions[(current_id, d)] = state_to_id[next_state_tuple]
    
    print(f"State space has {len(state_to_id)} unique states")
    
    # Create Z3 solver
    solver = Solver()
    solver.set(timeout=60000)  # 60 second timeout
    
    # Create variables for states at each time step
    states = {}
    for t in range(max_steps + 1):
        for state_id in range(len(state_to_id)):
            states[(t, state_id)] = Bool(f"state_{t}_{state_id}")
    
    # Create variables for actions at each time step
    actions = {}
    for t in range(max_steps):
        for d in range(4):
            actions[(t, d)] = Bool(f"action_{t}_{d}")
    
    # Initial state constraint: We start in state 0
    solver.add(states[(0, 0)])
    for state_id in range(1, len(state_to_id)):
        solver.add(Not(states[(0, state_id)]))
    
    # Add constraints for each time step
    for t in range(max_steps):
        # Exactly one action per time step
        solver.add(PbEq([(actions[(t, d)], 1) for d in range(4)], 1))
        
        # Exactly one state at each time step
        solver.add(PbEq([(states[(t+1, sid)], 1) for sid in range(len(state_to_id))], 1))
        
        # Transition constraints
        for current_id in range(len(state_to_id)):
            for d in range(4):
                # If we're in state current_id at time t and take action d...
                condition = And(states[(t, current_id)], actions[(t, d)])
                
                # ...then we end up in the next state at t+1
                if (current_id, d) in transitions:
                    next_id = transitions[(current_id, d)]
                    solver.add(Implies(condition, states[(t+1, next_id)]))
                else:
                    # If no transition exists, we stay in the same state
                    solver.add(Implies(condition, states[(t+1, current_id)]))
    
    # Goal constraint: We want to reach a state where robot is at goal
    goal_states = []
    for state_id, state in id_to_state.items():
        robot_pos, _ = state
        if robot_pos == goal_pos:
            goal_states.append(state_id)
    
    # Must reach one of the goal states at some time step
    goal_constraints = []
    for t in range(1, max_steps + 1):
        for goal_id in goal_states:
            goal_constraints.append(states[(t, goal_id)])
    
    solver.add(Or(goal_constraints))
    
    # Solve
    print("Solving...")
    start_time = time.time()
    result = solver.check()
    end_time = time.time()
    
    print(f"Solving took {end_time - start_time:.2f} seconds")
    print(f"Result: {result}")
    
    if result != sat:
        print("No solution found.")
        return None, None, None, None
    
    # Extract solution
    model = solver.model()
    
    robot_path = [robot_pos]
    box_paths = []
    for i in range(num_boxes):
        box_paths.append([box_positions[i]])
    
    plan = []
    current_state_id = 0  # Start with initial state
    
    # Extract the solution path
    for t in range(max_steps):
        # Find action at time t
        action_found = False
        for d in range(4):
            if is_true(model.eval(actions[(t, d)])):
                plan.append(d)
                action_found = True
                break
        
        if not action_found:
            # No more actions, we're done
            break
            
        # Find next state
        next_state_found = False
        for sid in range(len(state_to_id)):
            if is_true(model.eval(states[(t+1, sid)])):
                current_state_id = sid
                next_state_found = True
                break
                
        if not next_state_found:
            # No more states, we're done
            break
            
        # Get state info
        state = id_to_state[current_state_id]
        robot_pos, box_positions = state
        
        # Record positions
        robot_path.append(robot_pos)
        for i in range(num_boxes):
            box_paths[i].append(box_positions[i])
            
        # Check if goal reached
        if robot_pos == goal_pos:
            print(f"Goal reached at step {t+1}!")
            break
    
    print(f"Found solution in {len(plan)} steps")
    print("Plan:", [dir_names[d] for d in plan])
    
    return grid, robot_path, box_paths, goal_pos

def animate_solution(grid, robot_path, box_paths, goal_pos):
    """Create an animation of the solution."""
    fig, ax = plt.subplots(figsize=(10, 10))
    h, w = grid.shape
    num_boxes = len(box_paths)
    box_colors = ['blue', 'green', 'orange', 'purple', 'cyan']

    max_frames = len(robot_path)

    def draw_frame(t):
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)  # Inverted Y for matrix display
        ax.set_aspect('equal')
        ax.set_title(f"Step {t}/{max_frames - 1}")

        # Draw grid cells
        for r in range(h):
            for c in range(w):
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor='gray', linewidth=0.5))
                if grid[r, c] == 1:  # Wall
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='black'))
                elif (r, c) == goal_pos:  # Goal
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='lime', alpha=0.5))

        # Draw boxes
        for i in range(num_boxes):
            if t < len(box_paths[i]):
                br, bc = box_paths[i][t]
                color = box_colors[i % len(box_colors)]
                ax.add_patch(plt.Rectangle((bc - 0.4, br - 0.4), 0.8, 0.8, color=color, label=f'Box {i+1}' if t==0 else ""))

        # Draw robot
        if t < len(robot_path):
            rr, rc = robot_path[t]
            ax.add_patch(plt.Circle((rc, rr), 0.35, color='red', label='Robot' if t==0 else ""))

        # Show legend only in the first frame
        if t == 0: 
            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), ncol=1)

    # Create animation
    ani = animation.FuncAnimation(fig, draw_frame, frames=max_frames, interval=500, repeat=True, repeat_delay=2000)
    
    try:
        ani.save("scene_24.gif", writer='pillow', dpi=100)
        print("✅ Animation saved as 'scene_24.gif'")
    except Exception as e:
        print(f"❌ Failed to save animation: {e}")
        print("Ensure you have 'pillow' installed (`pip install pillow`)")
    
    plt.close(fig)

def verify_sliding_mechanics():
    """Test function to verify sliding mechanics work correctly."""
    # Simple test grid
    grid = np.array([
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 2, 0, 4, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 3, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ])
    
    height, width = grid.shape
    robot_pos = tuple(map(int, np.argwhere(grid == 2)[0]))
    box_positions = [tuple(map(int, pos)) for pos in np.argwhere(grid == 4)]
    
    # Define movement directions
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    dir_names = ["up", "right", "down", "left"]
    
    def is_obstacle(r, c):
        return r < 0 or r >= height or c < 0 or c >= width or grid[r, c] == 1
    
    def calculate_slide(r, c, dr, dc, box_positions, pushed_box_idx=None):
        curr_r, curr_c = r, c
        
        # Print the slide path for debugging
        path = [(curr_r, curr_c)]
        
        while True:
            next_r, next_c = curr_r + dr, curr_c + dc
            
            # Check if we hit a wall
            if is_obstacle(next_r, next_c):
                print(f"  Hit wall at {(next_r, next_c)}")
                break
                
            # Check if we hit a box
            hit_box = False
            for idx, (box_r, box_c) in enumerate(box_positions):
                if idx == pushed_box_idx:
                    continue
                if (next_r, next_c) == (box_r, box_c):
                    print(f"  Hit box {idx} at {(box_r, box_c)}")
                    hit_box = True
                    break
            
            if hit_box:
                break
                
            # Continue sliding
            curr_r, curr_c = next_r, next_c
            path.append((curr_r, curr_c))
        
        print(f"  Slide path: {path}")
        return (curr_r, curr_c)
    
    def next_state(robot_r, robot_c, box_positions, direction_idx):
        dr, dc = directions[direction_idx]
        
        print(f"\nTesting direction: {dir_names[direction_idx]}")
        print(f"  Robot starts at: {(robot_r, robot_c)}")
        print(f"  Boxes start at: {box_positions}")
        
        # Calculate robot's next position
        robot_next_r, robot_next_c = robot_r + dr, robot_c + dc
        
        # Check if robot hits a wall immediately
        if is_obstacle(robot_next_r, robot_next_c):
            print(f"  Robot hits wall immediately at {(robot_next_r, robot_next_c)}")
            return (robot_r, robot_c), box_positions
            
        # Check if robot hits a box
        pushed_box_idx = -1
        for idx, (box_r, box_c) in enumerate(box_positions):
            if (robot_next_r, robot_next_c) == (box_r, box_c):
                pushed_box_idx = idx
                break
                
        # If robot hits a box, check if box can move
        if pushed_box_idx >= 0:
            print(f"  Robot hits box {pushed_box_idx}")
            box_r, box_c = box_positions[pushed_box_idx]
            box_next_r, box_next_c = box_r + dr, box_c + dc
            
            # Check if box hits a wall immediately
            if is_obstacle(box_next_r, box_next_c):
                print(f"  Box would hit wall at {(box_next_r, box_next_c)}")
                return (robot_r, robot_c), box_positions
                
            # Check if box hits another box immediately
            for idx, (other_r, other_c) in enumerate(box_positions):
                if idx != pushed_box_idx and (box_next_r, box_next_c) == (other_r, other_c):
                    print(f"  Box would hit another box at {(box_next_r, box_next_c)}")
                    return (robot_r, robot_c), box_positions
                    
            # Box can move, robot pushes it
            print("  Box can move, robot pushes it")
            
            # Robot stops at box's original position
            new_robot_pos = (box_r, box_c)
            
            # Box slides from its new position
            print("  Calculating box slide:")
            box_final_pos = calculate_slide(box_next_r, box_next_c, dr, dc, box_positions, pushed_box_idx)
            
            # Create new box positions
            new_box_positions = list(box_positions)
            new_box_positions[pushed_box_idx] = box_final_pos
            
            print(f"  Final robot position: {new_robot_pos}")
            print(f"  Final box positions: {new_box_positions}")
            
            return new_robot_pos, new_box_positions
            
        # If no box hit, robot slides until hitting a wall or box
        print("  Robot slides freely")
        print("  Calculating robot slide:")
        robot_final_pos = calculate_slide(robot_next_r, robot_next_c, dr, dc, box_positions)
        
        print(f"  Final robot position: {robot_final_pos}")
        print(f"  Final box positions: {box_positions}")
        
        return robot_final_pos, box_positions
    
    # Test all directions
    for d in range(4):
        new_robot_pos, new_box_positions = next_state(robot_pos[0], robot_pos[1], box_positions, d)
        print(f"Result for {dir_names[d]}: Robot: {new_robot_pos}, Boxes: {new_box_positions}")
        print("=" * 40)

if __name__ == "__main__":
    # Uncomment to verify sliding mechanics
    # verify_sliding_mechanics()
    
    # Solve the puzzle
    grid, robot_path, box_paths, goal_pos = solve_sokoban_on_ice()
    if grid is not None:
        animate_solution(grid, robot_path, box_paths, goal_pos)