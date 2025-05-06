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

def create_grid():
    """Create the ice sliding puzzle grid."""
    grid = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 3, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 2, 1, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ])
    return grid

def print_grid_values(grid):
    """Print the grid in a readable format."""
    for r in range(grid.shape[0]):
        print(' '.join(str(int(v)) for v in grid[r]))

def get_positions(grid):
    """Get the robot and goal positions from the grid."""
    robot_pos = tuple(map(int, np.argwhere(grid == ROBOT)[0]))
    goal_pos = tuple(map(int, np.argwhere(grid == GOAL)[0]))
    return robot_pos, goal_pos

def is_valid(grid, pos):
    """Check if a position is valid (within bounds and not a wall)."""
    r, c = pos
    h, w = grid.shape
    return 0 <= r < h and 0 <= c < w and grid[r, c] != WALL

def get_next_state(grid, pos, direction):
    """
    Get the next position after sliding in a direction.
    
    Args:
        grid: The game grid
        pos: Current position (r, c)
        direction: Direction to move (0=up, 1=right, 2=down, 3=left)
    
    Returns:
        Next position after sliding
    """
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
    dr, dc = directions[direction]
    r, c = pos
    
    # Slide until hitting a wall
    while True:
        nr, nc = r + dr, c + dc
        if not is_valid(grid, (nr, nc)):
            break
        r, c = nr, nc
    
    return (r, c)

def manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def a_star_search(grid):
    """A* Search algorithm for the ice sliding puzzle."""
    robot_pos, goal_pos = get_positions(grid)
    
    # Priority queue of (priority, moves, position, path)
    # Priority is f = g + h where g is path length and h is heuristic
    open_set = [(manhattan_distance(robot_pos, goal_pos), 0, robot_pos, [])]
    heapq.heapify(open_set)
    
    # Track visited states with the minimum number of moves to reach them
    visited = {robot_pos: 0}
    
    while open_set:
        _, moves, pos, path = heapq.heappop(open_set)
        
        if pos == goal_pos:
            return path
        
        # Try each direction
        for d in range(4):
            next_pos = get_next_state(grid, pos, d)
            
            # If we haven't visited this state or found a better path
            if next_pos not in visited or moves + 1 < visited[next_pos]:
                visited[next_pos] = moves + 1
                priority = moves + 1 + manhattan_distance(next_pos, goal_pos)
                heapq.heappush(open_set, (priority, moves + 1, next_pos, path + [d]))
    
    return None  # No solution found

def solve_sat_plan(grid):
    """Solve the ice sliding puzzle using SAT planning."""
    height, width = grid.shape
    robot_pos, goal_pos = get_positions(grid)
    
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    dir_names = ["up", "right", "down", "left"]
    max_steps = 10

    solver = Solver()
    robot_vars = {}
    actions = {}

    for t in range(max_steps + 1):
        for r in range(height):
            for c in range(width):
                robot_vars[(t, r, c)] = Bool(f"robot_{t}_{r}_{c}")

    for t in range(max_steps):
        for d in range(4):
            actions[(t, d)] = Bool(f"act_{t}_{dir_names[d]}")

    solver.add(robot_vars[(0, robot_pos[0], robot_pos[1])])
    for r in range(height):
        for c in range(width):
            if (r, c) != robot_pos:
                solver.add(Not(robot_vars[(0, r, c)]))

    def is_obstacle(r, c):
        return r < 0 or r >= height or c < 0 or c >= width or grid[r, c] == WALL

    for t in range(max_steps):
        solver.add(Or([actions[(t, d)] for d in range(4)]))
        for d1 in range(4):
            for d2 in range(d1 + 1, 4):
                solver.add(Or(Not(actions[(t, d1)]), Not(actions[(t, d2)])))

        for r in range(height):
            for c in range(width):
                if is_obstacle(r, c): continue
                for d, (dr, dc) in enumerate(directions):
                    rr, cc = r, c
                    while True:
                        next_r, next_c = rr + dr, cc + dc
                        if is_obstacle(next_r, next_c):
                            break
                        rr, cc = next_r, next_c
                    pre = And(robot_vars[(t, r, c)], actions[(t, d)])
                    solver.add(Implies(pre, robot_vars[(t + 1, rr, cc)]))
                    for r2 in range(height):
                        for c2 in range(width):
                            if (r2, c2) != (rr, cc) and not is_obstacle(r2, c2):
                                solver.add(Implies(pre, Not(robot_vars[(t + 1, r2, c2)])))

        solver.add(PbEq([(robot_vars[(t+1, r, c)], 1) for r in range(height) for c in range(width) if not is_obstacle(r, c)], 1))

    solver.add(Or([robot_vars[(t, goal_pos[0], goal_pos[1])] for t in range(1, max_steps + 1)]))

    start_time = time.time()
    result = solver.check()
    solve_time = time.time() - start_time

    if result != sat:
        return None, None, solve_time

    model = solver.model()
    robot_path = [robot_pos]
    plan = []
    for t in range(max_steps):
        for d in range(4):
            if is_true(model.evaluate(actions[(t, d)])):
                plan.append(d)
                break
        for r in range(height):
            for c in range(width):
                if is_true(model.evaluate(robot_vars[(t + 1, r, c)])):
                    robot_path.append((r, c))
                    break
            else:
                continue
            break
        if robot_path[-1] == goal_pos:
            break

    return plan, robot_path, solve_time

def compute_path(grid, plan):
    """Compute the robot's path given a plan of directions."""
    robot_pos, _ = get_positions(grid)
    path = [robot_pos]
    
    current_pos = robot_pos
    for direction in plan:
        current_pos = get_next_state(grid, current_pos, direction)
        path.append(current_pos)
    
    return path

def animate_solution(grid, robot_path, goal_pos, filename="Ice_path.gif"):
    """Create an animation of the robot's path."""
    fig, ax = plt.subplots()
    h, w = grid.shape

    def draw_frame(t):
        ax.clear()
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-0.5, w - 0.5); ax.set_ylim(-0.5, h - 0.5)
        ax.set_aspect('equal'); ax.invert_yaxis()
        ax.set_title(f"Step {t}/{len(robot_path) - 1}")

        for r in range(h):
            for c in range(w):
                if grid[r, c] == WALL:
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='black'))
                elif (r, c) == goal_pos:
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='lime'))

        rr, rc = robot_path[t]
        ax.add_patch(plt.Circle((rc, rr), 0.3, color='red'))

    ani = animation.FuncAnimation(fig, draw_frame, frames=len(robot_path), interval=700, repeat=False)
    ani.save(filename, writer='pillow')
    print(f"✅ Animation saved as '{filename}'")

def main():
    # Create the grid
    grid = create_grid()
    robot_pos, goal_pos = get_positions(grid)
    
    print("Initial Grid:")
    print_grid_values(grid)
    print(f"Robot: {robot_pos}, Goal: {goal_pos}")
    
    # A*
    print("\n=== A* Search ===")
    start_time = time.time()
    astar_plan = a_star_search(grid)
    astar_time = time.time() - start_time
    
    if astar_plan:
        astar_path = compute_path(grid, astar_plan)
        print(f"A* found a solution in {len(astar_plan)} steps")
        print(f"A* runtime: {astar_time:.6f} seconds")
        animate_solution(grid, astar_path, goal_pos, "Ice_path_astar.gif")
    else:
        print("A* couldn't find a solution")
    
    # SAT Planning
    print("\n=== SAT Planning ===")
    sat_plan, sat_path, sat_time = solve_sat_plan(grid)
    
    if sat_plan:
        print(f"SAT found a solution in {len(sat_plan)} steps")
        print(f"SAT runtime: {sat_time:.6f} seconds")
        animate_solution(grid, sat_path, goal_pos, "Ice_path_sat.gif")
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