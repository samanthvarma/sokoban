import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import heapq
from z3 import *

# Grid constants
EMPTY = 0
WALL = 1
ROBOT = 2
GOAL = 3
BOX = 4

def create_grid():
    """Create the ice sliding puzzle grid with a box."""
    grid = np.array([
        [2, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0],
        [0, 0, 3, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 4, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
    ])
    return grid

def print_grid_values(grid):
    """Print the grid in a readable format."""
    for r in range(grid.shape[0]):
        print(' '.join(str(int(v)) for v in grid[r]))

def get_positions(grid):
    """Get the robot, goal, and box positions from the grid."""
    robot_pos = tuple(map(int, np.argwhere(grid == ROBOT)[0]))
    goal_pos = tuple(map(int, np.argwhere(grid == GOAL)[0]))
    box_pos = tuple(map(int, np.argwhere(grid == BOX)[0]))
    return robot_pos, goal_pos, box_pos

def is_valid(grid, pos):
    """Check if a position is valid (within bounds and not a wall)."""
    r, c = pos
    h, w = grid.shape
    return 0 <= r < h and 0 <= c < w and grid[r, c] != WALL

def is_obstacle(grid, r, c):
    """Check if a position is an obstacle (wall or out of bounds)."""
    h, w = grid.shape
    return r < 0 or r >= h or c < 0 or c >= w or grid[r, c] == WALL

def next_position(grid, robot_pos, box_pos, direction):
    """
    Calculate the next position after sliding on ice in the given direction.
    Returns:
        - New robot position
        - New box position
    """
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
    dr, dc = directions[direction]
    r, c = robot_pos
    box_r, box_c = box_pos
    
    # First move
    next_r, next_c = r + dr, c + dc
    
    # Check if first move hits an obstacle
    if is_obstacle(grid, next_r, next_c):
        return robot_pos, box_pos
    
    # Check if first move hits the box
    if (next_r, next_c) == box_pos:
        # Calculate where box would move
        box_next_r, box_next_c = box_r + dr, box_c + dc
        
        # If box would hit obstacle, robot can't move
        if is_obstacle(grid, box_next_r, box_next_c):
            return robot_pos, box_pos
        
        # Box moves, continue sliding both
        box_pos = (box_next_r, box_next_c)
        robot_pos = (next_r, next_c)
        
        # Continue sliding both box and robot
        while True:
            next_r, next_c = robot_pos[0] + dr, robot_pos[1] + dc
            box_next_r, box_next_c = box_pos[0] + dr, box_pos[1] + dc
            
            # If box hits obstacle, both stop
            if is_obstacle(grid, box_next_r, box_next_c):
                return robot_pos, box_pos
            
            # Both continue sliding
            robot_pos = (next_r, next_c)
            box_pos = (box_next_r, box_next_c)
    else:
        # No box hit initially, just slide robot until obstacle or box
        robot_pos = (next_r, next_c)
        while True:
            next_r, next_c = robot_pos[0] + dr, robot_pos[1] + dc
            
            # If robot would hit obstacle, stop
            if is_obstacle(grid, next_r, next_c):
                return robot_pos, box_pos
                
            # If robot would hit box, stop
            if (next_r, next_c) == box_pos:
                return robot_pos, box_pos
                
            # Continue sliding
            robot_pos = (next_r, next_c)
    
    # Should never reach here, but just in case
    return robot_pos, box_pos

def manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def a_star_search(grid):
    """A* Search algorithm for the ice sliding puzzle with a box."""
    robot_pos, goal_pos, box_pos = get_positions(grid)
    
    # State is (robot_pos, box_pos)
    # Priority queue of (priority, moves, robot_pos, box_pos, path)
    # Priority is f = g + h where g is path length and h is heuristic
    initial_state = (robot_pos, box_pos)
    open_set = [(manhattan_distance(robot_pos, goal_pos), 0, robot_pos, box_pos, [])]
    heapq.heapify(open_set)
    
    # Track visited states with the minimum number of moves to reach them
    visited = {initial_state: 0}
    
    while open_set:
        _, moves, r_pos, b_pos, path = heapq.heappop(open_set)
        
        if r_pos == goal_pos:
            return path, [(robot_pos, box_pos)] + path_positions(grid, robot_pos, box_pos, path)
        
        # Try each direction
        for d in range(4):
            next_r_pos, next_b_pos = next_position(grid, r_pos, b_pos, d)
            next_state = (next_r_pos, next_b_pos)
            
            # Skip if nothing moved (action had no effect)
            if next_state == (r_pos, b_pos):
                continue
            
            # If we haven't visited this state or found a better path
            if next_state not in visited or moves + 1 < visited[next_state]:
                visited[next_state] = moves + 1
                
                # Simple heuristic: Manhattan distance from robot to goal
                priority = moves + 1 + manhattan_distance(next_r_pos, goal_pos)
                
                heapq.heappush(open_set, (priority, moves + 1, next_r_pos, next_b_pos, path + [d]))
    
    return None, None  # No solution found

def path_positions(grid, start_robot_pos, start_box_pos, plan):
    """Compute the sequence of positions from executing a plan."""
    positions = []
    r_pos, b_pos = start_robot_pos, start_box_pos
    
    for d in plan:
        r_pos, b_pos = next_position(grid, r_pos, b_pos, d)
        positions.append((r_pos, b_pos))
    
    return positions

def solve_sat_plan(grid):
    """Solve the ice sliding puzzle with box using SAT planning."""
    height, width = grid.shape
    robot_pos, goal_pos, box_pos = get_positions(grid)
    
    # Define movement directions: up, right, down, left
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    dir_names = ["up", "right", "down", "left"]
    max_steps = 10

    solver = Solver()
    robot_vars, box_vars, actions = {}, {}, {}

    # Create variables for robot and box positions at each time step
    for t in range(max_steps + 1):
        for r in range(height):
            for c in range(width):
                robot_vars[(t, r, c)] = Bool(f"robot_{t}_{r}_{c}")
                box_vars[(t, r, c)] = Bool(f"box_{t}_{r}_{c}")
    
    # Create variables for actions at each time step
    for t in range(max_steps):
        for d in range(4):
            actions[(t, d)] = Bool(f"action_{t}_{dir_names[d]}")

    # Set initial positions
    solver.add(robot_vars[(0, *robot_pos)])
    solver.add(box_vars[(0, *box_pos)])
    for r in range(height):
        for c in range(width):
            if (r, c) != robot_pos:
                solver.add(Not(robot_vars[(0, r, c)]))
            if (r, c) != box_pos:
                solver.add(Not(box_vars[(0, r, c)]))

    def is_obstacle_sat(r, c):
        return r < 0 or r >= height or c < 0 or c >= width or grid[r, c] == 1

    def next_position_sat(r, c, d, box_r, box_c):
        """Calculate the next position after sliding on ice in direction d.
        Returns:
            - New robot position
            - Whether box moved
            - New box position
        """
        dr, dc = directions[d]
        curr_r, curr_c = r, c
        box_moved = False
        box_new_r, box_new_c = box_r, box_c
        
        # First move
        next_r, next_c = curr_r + dr, curr_c + dc
        
        # Check if first move hits an obstacle
        if is_obstacle_sat(next_r, next_c):
            return curr_r, curr_c, box_moved, box_new_r, box_new_c
        
        # Check if first move hits the box
        if (next_r, next_c) == (box_r, box_c):
            # Calculate where box would move
            box_next_r, box_next_c = box_r + dr, box_c + dc
            
            # If box would hit obstacle, robot can't move
            if is_obstacle_sat(box_next_r, box_next_c):
                return curr_r, curr_c, box_moved, box_r, box_c
            
            # Box moves, continue sliding both
            box_moved = True
            box_new_r, box_new_c = box_next_r, box_next_c
            curr_r, curr_c = next_r, next_c
            
            # Continue sliding both box and robot
            while True:
                next_r, next_c = curr_r + dr, curr_c + dc
                box_next_r, box_next_c = box_new_r + dr, box_new_c + dc
                
                # If box hits obstacle, both stop
                if is_obstacle_sat(box_next_r, box_next_c):
                    return curr_r, curr_c, box_moved, box_new_r, box_new_c
                
                # Both continue sliding
                curr_r, curr_c = next_r, next_c
                box_new_r, box_new_c = box_next_r, box_next_c
        else:
            # No box hit initially, just slide robot until obstacle or box
            curr_r, curr_c = next_r, next_c
            while True:
                next_r, next_c = curr_r + dr, curr_c + dc
                
                # If robot would hit obstacle, stop
                if is_obstacle_sat(next_r, next_c):
                    return curr_r, curr_c, box_moved, box_r, box_c
                    
                # If robot would hit box, stop
                if (next_r, next_c) == (box_r, box_c):
                    return curr_r, curr_c, box_moved, box_r, box_c
                    
                # Continue sliding
                curr_r, curr_c = next_r, next_c
        
        # Should never reach here
        return curr_r, curr_c, box_moved, box_new_r, box_new_c

    # Add constraints for each time step
    for t in range(max_steps):
        # One action per time step
        solver.add(Or([actions[(t, d)] for d in range(4)]))
        for d1 in range(4):
            for d2 in range(d1 + 1, 4):
                solver.add(Or(Not(actions[(t, d1)]), Not(actions[(t, d2)])))

        # Effects of actions on robot and box positions
        for r in range(height):
            for c in range(width):
                if is_obstacle_sat(r, c): continue
                for box_r in range(height):
                    for box_c in range(width):
                        if is_obstacle_sat(box_r, box_c) or (r, c) == (box_r, box_c): continue
                        for d in range(4):
                            r_new, c_new, box_moved, b_new_r, b_new_c = next_position_sat(r, c, d, box_r, box_c)
                            
                            # Create precondition: robot at (r,c), box at (box_r,box_c), action d
                            pre = And(robot_vars[(t, r, c)], box_vars[(t, box_r, box_c)], actions[(t, d)])
                            
                            # Add effect: robot moves to new position
                            solver.add(Implies(pre, robot_vars[(t+1, r_new, c_new)]))
                            
                            # Add effect: box moves or stays in place
                            if box_moved:
                                solver.add(Implies(pre, box_vars[(t+1, b_new_r, b_new_c)]))
                            else:
                                solver.add(Implies(pre, box_vars[(t+1, box_r, box_c)]))

        # Constraints ensuring exactly one robot and box position at each time step
        solver.add(PbEq([(robot_vars[(t+1, r, c)], 1) for r in range(height) for c in range(width) if not is_obstacle_sat(r, c)], 1))
        solver.add(PbEq([(box_vars[(t+1, r, c)], 1) for r in range(height) for c in range(width) if not is_obstacle_sat(r, c)], 1))
        
        # Robot and box cannot be in the same position
        for r in range(height):
            for c in range(width):
                if not is_obstacle_sat(r, c):
                    solver.add(Not(And(robot_vars[(t+1, r, c)], box_vars[(t+1, r, c)])))

    # Goal condition: robot must reach the goal position
    solver.add(Or([robot_vars[(t, goal_pos[0], goal_pos[1])] for t in range(1, max_steps + 1)]))

    start_time = time.time()
    result = solver.check()
    solve_time = time.time() - start_time

    if result != sat:
        return None, None, solve_time

    model = solver.model()
    robot_path = [robot_pos]
    box_path = [box_pos]
    plan = []
    
    # Extract solution
    for t in range(max_steps):
        # Find which action was taken
        for d in range(4):
            if is_true(model.evaluate(actions[(t, d)])):
                plan.append(d)
                break
        
        # Find where the robot moved
        next_robot_pos = None
        for r in range(height):
            for c in range(width):
                if is_true(model.evaluate(robot_vars[(t+1, r, c)])):
                    next_robot_pos = (r, c)
                    robot_path.append(next_robot_pos)
                    break
            if next_robot_pos:
                break
        
        # Find where the box moved
        next_box_pos = None
        for r in range(height):
            for c in range(width):
                if is_true(model.evaluate(box_vars[(t+1, r, c)])):
                    next_box_pos = (r, c)
                    box_path.append(next_box_pos)
                    break
            if next_box_pos:
                break
        
        # Check if goal reached
        if next_robot_pos == goal_pos:
            break

    return plan, robot_path, box_path, solve_time

def animate_solution(grid, robot_path, box_path, goal_pos, filename="Ice_box_path.gif"):
    """Create an animation of the robot and box paths."""
    fig, ax = plt.subplots(figsize=(10, 8))
    h, w = grid.shape

    def draw_frame(t):
        ax.clear()
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-0.5, w - 0.5); ax.set_ylim(h - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.set_title(f"Step {t}/{len(robot_path) - 1}")

        # Draw grid and obstacles
        for r in range(h):
            for c in range(w):
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor='gray'))
                if grid[r, c] == WALL:
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='black'))
                elif (r, c) == goal_pos:
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='lime'))

        # Draw robot and box
        rr, rc = robot_path[min(t, len(robot_path)-1)]
        br, bc = box_path[min(t, len(box_path)-1)]
        ax.add_patch(plt.Rectangle((bc - 0.4, br - 0.4), 0.8, 0.8, color='blue'))
        ax.add_patch(plt.Circle((rc, rr), 0.3, color='red'))

    ani = animation.FuncAnimation(fig, draw_frame, frames=len(robot_path), interval=700, repeat=True)
    ani.save(filename, writer='pillow')
    plt.close(fig)
    print(f"✅ Animation saved as '{filename}'")

def main():
    # Create the grid
    grid = create_grid()
    robot_pos, goal_pos, box_pos = get_positions(grid)
    
    print("Initial Grid:")
    print_grid_values(grid)
    print(f"Robot: {robot_pos}, Goal: {goal_pos}, Box: {box_pos}")
    
    # A* Search
    print("\n=== A* Search ===")
    start_time = time.time()
    astar_plan, astar_positions = a_star_search(grid)
    astar_time = time.time() - start_time
    
    if astar_plan:
        robot_path = [pos[0] for pos in astar_positions]  # Extract robot positions
        box_path = [pos[1] for pos in astar_positions]    # Extract box positions
        print(f"A* found a solution in {len(astar_plan)} steps")
        print(f"A* runtime: {astar_time:.6f} seconds")
        animate_solution(grid, robot_path, box_path, goal_pos, "Ice_box_path_astar.gif")
    else:
        print("A* couldn't find a solution")
    
    # SAT Planning
    print("\n=== SAT Planning ===")
    sat_plan, robot_path, box_path, sat_time = solve_sat_plan(grid)
    
    if sat_plan:
        print(f"SAT found a solution in {len(sat_plan)} steps")
        print(f"SAT runtime: {sat_time:.6f} seconds")
        animate_solution(grid, robot_path, box_path, goal_pos, "Ice_box_path_sat.gif")
    else:
        print("SAT couldn't find a solution")
    
    # Compare results
    print("\n=== Results Comparison ===")
    print(f"Algorithm | Solution Length | Runtime (seconds)")
    print(f"----------------------------------------------")
    print(f"A*        | {len(astar_plan) if astar_plan else 'N/A'}              | {astar_time:.6f}")
    print(f"SAT       | {len(sat_plan) if sat_plan else 'N/A'}              | {sat_time:.6f}")

if __name__ == "__main__":
    main()