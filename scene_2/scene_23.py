from z3 import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import itertools

# Directions: Up, Right, Down, Left
directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
dir_names = ["up", "right", "down", "left"]

class IcePuzzleSolver:
    def __init__(self, grid=None):
        # Default grid if none provided
        if grid is None:
            # Grid definition: 0=empty, 1=wall, 2=robot, 3=goal, 4=box
            self.grid = np.array([
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 2, 0, 1, 0, 0, 1],
                [1, 0, 4, 0, 0, 4, 0, 1],
                [1, 0, 0, 1, 0, 0, 1, 1],
                [1, 0, 0, 0, 0, 4, 0, 1],
                [1, 0, 1, 0, 3, 0, 0, 1],
                [1, 0, 0, 1, 0, 4, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ])
        else:
            self.grid = grid
            
        # Extract dimensions and positions
        self.height, self.width = self.grid.shape
        self.robot_pos = tuple(map(int, np.argwhere(self.grid == 2)[0]))
        self.goal_pos = tuple(map(int, np.argwhere(self.grid == 3)[0]))
        self.box_positions = [tuple(map(int, pos)) for pos in np.argwhere(self.grid == 4)]
        self.num_boxes = len(self.box_positions)
        
        # Calculate valid cells
        self.valid_cells = []
        for r in range(self.height):
            for c in range(self.width):
                if not self.is_obstacle(r, c, self.grid):
                    self.valid_cells.append((r, c))
        
        print(f"Puzzle initialized with {self.num_boxes} boxes")
        print(f"Robot starts at {self.robot_pos}")
        print(f"Goal at {self.goal_pos}")
        print(f"Boxes at {self.box_positions}")
        
    def is_obstacle(self, r, c, current_grid):
        """Check if a cell is an obstacle or out of bounds."""
        if r < 0 or r >= self.height or c < 0 or c >= self.width or current_grid[r, c] == 1:
            return True
        return False
        
    def calculate_stop_position(self, start_r, start_c, dr, dc, current_grid, current_box_positions, ignore_box_idx=None):
        """Find where an object (robot or box) stops sliding."""
        curr_r, curr_c = start_r, start_c
        while True:
            next_r, next_c = curr_r + dr, curr_c + dc

            # Check for grid boundaries or static walls
            if self.is_obstacle(next_r, next_c, current_grid):
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
    
    def next_position(self, robot_r, robot_c, direction_idx, current_box_positions, current_grid):
        """Calculate the final position after one action."""
        dr, dc = directions[direction_idx]

        # Calculate robot's immediate next step
        robot_next_r, robot_next_c = robot_r + dr, robot_c + dc

        # Check for immediate wall hit
        if self.is_obstacle(robot_next_r, robot_next_c, current_grid):
            return (robot_r, robot_c), []  # Robot doesn't move

        # Check for immediate box hit
        hit_box_idx = -1
        for idx, (br, bc) in enumerate(current_box_positions):
            if (robot_next_r, robot_next_c) == (br, bc):
                hit_box_idx = idx
                break

        # Handle Box Hit
        if hit_box_idx != -1:
            box_start_r, box_start_c = robot_next_r, robot_next_c  # Box starts where robot tried to move

            # Check if the space BEYOND the box is blocked
            box_next_r, box_next_c = box_start_r + dr, box_start_c + dc

            blocked = False
            if self.is_obstacle(box_next_r, box_next_c, current_grid):
                blocked = True
            else:
                # Check collision with OTHER boxes
                for idx, (br, bc) in enumerate(current_box_positions):
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
                final_box_pos = self.calculate_stop_position(
                    box_next_r, box_next_c, dr, dc, current_grid, 
                    current_box_positions, ignore_box_idx=hit_box_idx
                )
                
                return final_robot_pos, [(hit_box_idx, final_box_pos)]

        # Handle Sliding into Empty Space
        else:
            # Robot slides until it hits something (wall or any box)
            final_robot_pos = self.calculate_stop_position(
                robot_next_r, robot_next_c, dr, dc, current_grid, 
                current_box_positions
            )
            return final_robot_pos, []
            
    def create_state_space_graph(self, max_states=50000):
        """
        Create a state space graph for more efficient planning.
        Each node is a state (robot_pos, box_positions).
        Each edge is an action that transitions between states.
        """
        print("Building state space graph...")
        
        # Dictionary to store the graph: state -> {action: next_state}
        graph = {}
        # Dictionary to track backward paths: state -> (prev_state, action)
        backtrack = {}
        
        # Queue for BFS
        queue = [(self.robot_pos, tuple(self.box_positions))]
        visited = set(queue)
        goal_state = None
        
        state_count = 0
        while queue and state_count < max_states:
            current_state = queue.pop(0)
            state_count += 1
            
            if state_count % 1000 == 0:
                print(f"Processed {state_count} states, queue size: {len(queue)}")
            
            robot_pos, box_positions = current_state
            box_positions_list = list(box_positions)
            
            # Check if goal reached
            if robot_pos == self.goal_pos:
                goal_state = current_state
                print(f"Goal reached after exploring {state_count} states!")
                break
            
            # Try each direction
            graph[current_state] = {}
            for d in range(4):
                (new_r, new_c), moved_boxes = self.next_position(
                    robot_pos[0], robot_pos[1], d, box_positions_list, self.grid
                )
                
                # Create new box positions after move
                new_box_positions = list(box_positions_list)
                for box_idx, new_pos in moved_boxes:
                    new_box_positions[box_idx] = new_pos
                
                next_state = ((new_r, new_c), tuple(new_box_positions))
                
                # Only add if state actually changed
                if next_state != current_state:
                    graph[current_state][d] = next_state
                    
                    # If new state, add to queue
                    if next_state not in visited:
                        visited.add(next_state)
                        queue.append(next_state)
                        backtrack[next_state] = (current_state, d)
                        
        print(f"State space exploration complete. Explored {state_count} states.")
        print(f"Goal found: {goal_state is not None}")
        
        return graph, backtrack, goal_state

    def extract_path(self, backtrack, goal_state):
        """Extract path from goal to start using backtrack dictionary."""
        if goal_state is None:
            return None, None, None
            
        current = goal_state
        robot_path = [current[0]]
        box_paths = []
        for box_idx in range(self.num_boxes):
            box_paths.append([current[1][box_idx]])
        actions = []
        
        while current in backtrack:
            prev_state, action = backtrack[current]
            robot_path.append(prev_state[0])
            for box_idx in range(self.num_boxes):
                box_paths[box_idx].append(prev_state[1][box_idx])
            actions.append(action)
            current = prev_state
            
        # Reverse paths to go from start to goal
        robot_path.reverse()
        for path in box_paths:
            path.reverse()
        actions.reverse()
        
        return robot_path, box_paths, actions

    def solve_with_state_space_search(self, max_states=50000):
        """Solve the puzzle using state space search."""
        graph, backtrack, goal_state = self.create_state_space_graph(max_states)
        
        if goal_state is None:
            print("No solution found within the state limit.")
            return None, None, None
            
        robot_path, box_paths, actions = self.extract_path(backtrack, goal_state)
        
        print(f"Found solution with {len(actions)} actions:")
        action_names = [dir_names[a] for a in actions]
        print(", ".join(action_names))
        
        return robot_path, box_paths, actions

    def solve_with_z3(self, max_steps=60):
        """
        Solve using Z3 SMT solver with a simplified and more efficient approach.
        """
        print(f"Initializing Z3 solver with {max_steps} steps...")
        
        # Create solver with a timeout
        solver = Solver()
        solver.set(timeout=60000)  # 60 second timeout
        
        # Variables for state representation
        states = {}  # (t, state_id) -> Bool
        actions = {}  # (t, action) -> Bool
        
        # Build a compact state space
        state_to_id = {}
        id_to_state = {}
        transitions = {}  # (state_id, action) -> next_state_id
        
        # Initial state gets ID 0
        initial_state = (self.robot_pos, tuple(self.box_positions))
        state_to_id[initial_state] = 0
        id_to_state[0] = initial_state
        next_state_id = 1
        
        # Create state variables for initial state
        states[(0, 0)] = Bool(f"state_0_0")  # At time 0, we're in state 0
        
        # Ensure we start in the initial state
        solver.add(states[(0, 0)])
        
        # Queue for BFS state space exploration
        queue = [initial_state]
        visited = {initial_state}
        
        print("Precomputing state space...")
        while queue:
            current_state = queue.pop(0)
            current_id = state_to_id[current_state]
            
            robot_pos, box_positions = current_state
            box_positions_list = list(box_positions)
            
            # Try each direction
            for d in range(4):
                (new_r, new_c), moved_boxes = self.next_position(
                    robot_pos[0], robot_pos[1], d, box_positions_list, self.grid
                )
                
                # Create new box positions after move
                new_box_positions = list(box_positions_list)
                for box_idx, new_pos in moved_boxes:
                    new_box_positions[box_idx] = new_pos
                
                next_state = ((new_r, new_c), tuple(new_box_positions))
                
                # Add to state mapping if new
                if next_state not in state_to_id:
                    state_to_id[next_state] = next_state_id
                    id_to_state[next_state_id] = next_state
                    next_state_id += 1
                    
                    # Only add to queue if not visited
                    if next_state not in visited:
                        visited.add(next_state)
                        queue.append(next_state)
                
                # Record transition
                next_id = state_to_id[next_state]
                transitions[(current_id, d)] = next_id
        
        print(f"Precomputed {len(state_to_id)} unique states")
        
        # Create variables for remaining steps
        for t in range(max_steps):
            # Action variables
            for d in range(4):
                actions[(t, d)] = Bool(f"action_{t}_{d}")
            
            # State variables for all discovered states
            for state_id in range(len(state_to_id)):
                states[(t+1, state_id)] = Bool(f"state_{t+1}_{state_id}")
        
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
        
        # Goal constraint: We want to reach a state where the robot is at the goal
        goal_states = []
        for state_id, state in id_to_state.items():
            robot_pos, _ = state
            if robot_pos == self.goal_pos:
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
        
        if result == sat:
            model = solver.model()
            
            # Extract solution
            robot_path = [self.robot_pos]
            box_paths = [[pos] for pos in self.box_positions]
            actions_taken = []
            
            # Find the sequence of states and actions
            current_state_id = 0  # Start with initial state
            for t in range(max_steps):
                # Find action at time t
                action_taken = -1
                for d in range(4):
                    if is_true(model.eval(actions[(t, d)])):
                        action_taken = d
                        actions_taken.append(d)
                        break
                
                if action_taken == -1:
                    break  # No action found, we're done
                
                # Find next state
                next_state_id = -1
                for sid in range(len(state_to_id)):
                    if (t+1, sid) in states and is_true(model.eval(states[(t+1, sid)])):
                        next_state_id = sid
                        break
                
                if next_state_id == -1:
                    break  # No next state found, we're done
                
                # Get the actual state info
                next_state = id_to_state[next_state_id]
                next_robot_pos, next_box_positions = next_state
                
                # Record positions
                robot_path.append(next_robot_pos)
                for box_idx, box_pos in enumerate(next_box_positions):
                    box_paths[box_idx].append(box_pos)
                
                # Check if goal reached
                if next_robot_pos == self.goal_pos:
                    print(f"Goal reached at step {t+1}!")
                    break
                
                current_state_id = next_state_id
            
            print(f"Solution length: {len(actions_taken)} steps")
            print(f"Actions: {[dir_names[a] for a in actions_taken]}")
            
            return robot_path, box_paths, actions_taken
        else:
            print("No solution found.")
            return None, None, None

    def solve(self, method="state_space", max_steps=50):
        """Main solving entry point."""
        start_time = time.time()
        
        if method == "state_space":
            robot_path, box_paths, actions = self.solve_with_state_space_search(max_steps)
        elif method == "z3":
            robot_path, box_paths, actions = self.solve_with_z3(max_steps)
        else:
            raise ValueError(f"Unknown solving method: {method}")
            
        end_time = time.time()
        print(f"Total solving time: {end_time - start_time:.2f} seconds")
        
        if robot_path:
            self.animate_solution(robot_path, box_paths)
            return True
        return False
        
    def animate_solution(self, robot_path, box_paths, filename="scene_23.gif"):
        """Create an animation of the solution."""
        if not robot_path:
            print("No path found to animate.")
            return False

        fig, ax = plt.subplots(figsize=(10, 10))
        box_colors = ['blue', 'green', 'orange', 'purple', 'cyan']

        max_frames = len(robot_path)

        def draw_frame(t):
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(-0.5, self.width - 0.5)
            ax.set_ylim(self.height - 0.5, -0.5)  # Inverted Y for matrix display
            ax.set_aspect('equal')
            ax.set_title(f"Step {t}/{max_frames - 1}")

            # Draw grid cells
            for r in range(self.height):
                for c in range(self.width):
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor='gray', linewidth=0.5))
                    if self.grid[r, c] == 1:  # Wall
                        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='black'))
                    elif (r, c) == self.goal_pos:  # Goal
                        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='lime', alpha=0.5))

            # Draw boxes
            for i, box_path in enumerate(box_paths):
                if t < len(box_path):
                    br, bc = box_path[t]
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
            ani.save(filename, writer='pillow', dpi=100)
            print(f"✅ Animation saved as '{filename}'")
        except Exception as e:
            print(f"❌ Failed to save animation: {e}")
            print("Ensure you have 'pillow' installed (`pip install pillow`)")
        
        plt.close(fig)
        return True


def create_complex_puzzle():
    """Create a more complex puzzle for testing."""
    # Grid definition: 0=empty, 1=wall, 2=robot, 3=goal, 4=box
    grid = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 2, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 4, 0, 0, 4, 0, 1],
        [1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 4, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 3, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ])
    return grid


if __name__ == "__main__":
    # Uncomment for more complex puzzle
    # grid = create_complex_puzzle()
    # solver = IcePuzzleSolver(grid)
    
    # Use default puzzle from your original code
    solver = IcePuzzleSolver()
    
    # Choose method: "state_space" (faster) or "z3" (more complete)
    # For Z3 method, keep max_steps low (~30) due to memory constraints
    solver.solve(method="state_space", max_steps=10000)