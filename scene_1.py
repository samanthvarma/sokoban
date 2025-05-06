from z3 import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def solve_ice_with_box():
    grid = np.array([
        [2, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0],
        [0, 0, 3, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 4, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
    ])

    height, width = grid.shape
    robot_pos = tuple(map(int, np.argwhere(grid == 2)[0]))
    goal_pos = tuple(map(int, np.argwhere(grid == 3)[0]))
    box_pos = tuple(map(int, np.argwhere(grid == 4)[0]))

    print("Robot:", robot_pos)
    print("Goal:", goal_pos)
    print("Box:", box_pos)

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

    def is_obstacle(r, c):
        return r < 0 or r >= height or c < 0 or c >= width or grid[r, c] == 1

    def next_position(r, c, d, box_r, box_c):
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
        if is_obstacle(next_r, next_c):
            return curr_r, curr_c, box_moved, box_new_r, box_new_c
        
        # Check if first move hits the box
        if (next_r, next_c) == (box_r, box_c):
            # Calculate where box would move
            box_next_r, box_next_c = box_r + dr, box_c + dc
            
            # If box would hit obstacle, robot can't move
            if is_obstacle(box_next_r, box_next_c):
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
                if is_obstacle(box_next_r, box_next_c):
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
                if is_obstacle(next_r, next_c):
                    return curr_r, curr_c, box_moved, box_r, box_c
                    
                # If robot would hit box, stop
                if (next_r, next_c) == (box_r, box_c):
                    return curr_r, curr_c, box_moved, box_r, box_c
                    
                # Continue sliding
                curr_r, curr_c = next_r, next_c

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
                if is_obstacle(r, c): continue
                for box_r in range(height):
                    for box_c in range(width):
                        if is_obstacle(box_r, box_c) or (r, c) == (box_r, box_c): continue
                        for d in range(4):
                            r_new, c_new, box_moved, b_new_r, b_new_c = next_position(r, c, d, box_r, box_c)
                            
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
        solver.add(PbEq([(robot_vars[(t+1, r, c)], 1) for r in range(height) for c in range(width) if not is_obstacle(r, c)], 1))
        solver.add(PbEq([(box_vars[(t+1, r, c)], 1) for r in range(height) for c in range(width) if not is_obstacle(r, c)], 1))
        
        # Robot and box cannot be in the same position
        for r in range(height):
            for c in range(width):
                if not is_obstacle(r, c):
                    solver.add(Not(And(robot_vars[(t+1, r, c)], box_vars[(t+1, r, c)])))

    # Goal condition: robot must reach the goal position
    solver.add(Or([robot_vars[(t, goal_pos[0], goal_pos[1])] for t in range(1, max_steps + 1)]))

    print("Solving...")
    result = solver.check()
    print("SAT result:", result)

    if result != sat:
        print("No path found.")
        return None, None, None, None

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
        for r in range(height):
            for c in range(width):
                if is_true(model.evaluate(robot_vars[(t+1, r, c)])):
                    robot_path.append((r, c))
                    break
            else: continue
            break
        
        # Find where the box moved
        for r in range(height):
            for c in range(width):
                if is_true(model.evaluate(box_vars[(t+1, r, c)])):
                    box_path.append((r, c))
                    break
            else: continue
            break
        
        # Check if goal reached
        if robot_path[-1] == goal_pos:
            break

    print(f"Found path in {len(plan)} steps.")
    print("Plan:", [dir_names[d] for d in plan])
    return grid, robot_path, box_path, goal_pos

def animate_solution(grid, robot_path, box_path, goal_pos):
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
                if grid[r, c] == 1:
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='black'))
                elif (r, c) == goal_pos:
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='lime'))

        # Draw robot and box
        rr, rc = robot_path[min(t, len(robot_path)-1)]
        br, bc = box_path[min(t, len(box_path)-1)]
        ax.add_patch(plt.Rectangle((bc - 0.4, br - 0.4), 0.8, 0.8, color='blue'))
        ax.add_patch(plt.Circle((rc, rr), 0.3, color='red'))

    ani = animation.FuncAnimation(fig, draw_frame, frames=len(robot_path), interval=700, repeat=True)
    ani.save("scene_1.gif", writer='pillow')
    plt.close(fig)
    print("✅ Animation saved as 'scene_1.gif'")

if __name__ == "__main__":
    grid, robot_path, box_path, goal_pos = solve_ice_with_box()
    if grid is not None:
        animate_solution(grid, robot_path, box_path, goal_pos)