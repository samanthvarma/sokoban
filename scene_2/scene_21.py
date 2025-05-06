from z3 import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Directions: Up, Right, Down, Left
directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
dir_names = ['up', 'right', 'down', 'left']

# --- Helper functions ---
def is_wall(r, c, grid):
    h, w = grid.shape
    return r < 0 or r >= h or c < 0 or c >= w or grid[r, c] == 1

# --- Main solver ---
def solve_sokoban_symbolic():
    grid = np.array([
        [2, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0],
        [0, 0, 3, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 4, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
    ])

    h, w = grid.shape
    robot_start = tuple(map(int, np.argwhere(grid == 2)[0]))
    goal_pos = tuple(map(int, np.argwhere(grid == 3)[0]))
    box_starts = [tuple(map(int, p)) for p in np.argwhere(grid == 4)]
    num_boxes = len(box_starts)

    valid_cells = [(r, c) for r in range(h) for c in range(w) if not is_wall(r, c, grid)]

    max_steps = 100

    solver = Solver()

    # Variables
    robot_vars = {(t, r, c): Bool(f"robot_{t}_{r}_{c}") for t in range(max_steps+1) for r, c in valid_cells}
    box_vars = {(t, b, r, c): Bool(f"box_{t}_{b}_{r}_{c}") for t in range(max_steps+1) for b in range(num_boxes) for r, c in valid_cells}
    action_vars = {(t, d): Bool(f"action_{t}_{dir_names[d]}") for t in range(max_steps) for d in range(4)}

    # Initial conditions
    solver.add(robot_vars[(0, robot_start[0], robot_start[1])])
    for r, c in valid_cells:
        if (r, c) != robot_start:
            solver.add(Not(robot_vars[(0, r, c)]))

    for b, (r, c) in enumerate(box_starts):
        solver.add(box_vars[(0, b, r, c)])
        for r2, c2 in valid_cells:
            if (r2, c2) != (r, c):
                solver.add(Not(box_vars[(0, b, r2, c2)]))

    # Constraints
    for t in range(max_steps):
        solver.add(PbEq([(action_vars[(t, d)], 1) for d in range(4)], 1))  # Exactly one action

        # Robot transitions
        for r, c in valid_cells:
            for d, (dr, dc) in enumerate(directions):
                nr, nc = r + dr, c + dc

                cond_robot_at = robot_vars[(t, r, c)]
                cond_action = action_vars[(t, d)]
                move_conditions = []

                if is_wall(nr, nc, grid):
                    # Hit wall: stay
                    move_conditions.append(And(cond_robot_at, cond_action, robot_vars[(t+1, r, c)]))
                else:
                    # Check if box is there
                    box_in_front = [box_vars[(t, b, nr, nc)] for b in range(num_boxes)]
                    
                    # If pushing box
                    push_conditions = []
                    for b in range(num_boxes):
                        next_br, next_bc = nr + dr, nc + dc
                        if not is_wall(next_br, next_bc, grid):
                            no_box_blocking = And([Not(box_vars[(t, ob, next_br, next_bc)]) for ob in range(num_boxes) if ob != b])
                            push_conditions.append(And(box_vars[(t, b, nr, nc)], no_box_blocking,
                                robot_vars[(t+1, nr, nc)], box_vars[(t+1, b, next_br, next_bc)]))

                    # Slide conditions
                    slide_r, slide_c = nr, nc
                    while not is_wall(slide_r, slide_c, grid):
                        blocked = False
                        for b in range(num_boxes):
                            blocked = Or(blocked, box_vars[(t, b, slide_r, slide_c)])
                        move_conditions.append(And(cond_robot_at, cond_action, Not(blocked), robot_vars[(t+1, slide_r, slide_c)]))

                        slide_r += dr
                        slide_c += dc

                    # Add box push transitions
                    for pc in push_conditions:
                        move_conditions.append(And(cond_robot_at, cond_action, pc))

                # Only one move condition needs to happen
                if move_conditions:
                    solver.add(Or(move_conditions))

        # Frame axioms
        for b in range(num_boxes):
            for r, c in valid_cells:
                stay = []
                for d, (dr, dc) in enumerate(directions):
                    prev_r, prev_c = r - dr, c - dc
                    if (prev_r, prev_c) in valid_cells:
                        stay.append(And(box_vars[(t, b, prev_r, prev_c)], action_vars[(t, d)]))
                solver.add(Implies(Not(Or(stay)), box_vars[(t+1, b, r, c)] == box_vars[(t, b, r, c)]))

        # One robot position only
        solver.add(PbEq([(robot_vars[(t+1, r, c)], 1) for r, c in valid_cells], 1))

        # Each box at one position
        for b in range(num_boxes):
            solver.add(PbEq([(box_vars[(t+1, b, r, c)], 1) for r, c in valid_cells], 1))

        # No robot-box overlap
        for b in range(num_boxes):
            for r, c in valid_cells:
                solver.add(Not(And(robot_vars[(t+1, r, c)], box_vars[(t+1, b, r, c)])))

    # Goal condition
    solver.add(Or([robot_vars[(t, goal_pos[0], goal_pos[1])] for t in range(1, max_steps+1)]))

    print("Solving...")
    if solver.check() == sat:
        print("Solution found!")
        model = solver.model()

        robot_path = []
        box_paths = [[] for _ in range(num_boxes)]

        for t in range(max_steps+1):
            for r, c in valid_cells:
                if is_true(model.eval(robot_vars[(t, r, c)], model_completion=True)):
                    robot_path.append((r, c))
            for b in range(num_boxes):
                for r, c in valid_cells:
                    if is_true(model.eval(box_vars[(t, b, r, c)], model_completion=True)):
                        box_paths[b].append((r, c))

            if robot_path[-1] == goal_pos:
                break

        return grid, robot_path, box_paths, goal_pos
    else:
        print("No solution.")
        return None, None, None, None

# --- Animation ---
def animate(grid, robot_path, box_paths, goal_pos):
    fig, ax = plt.subplots()
    h, w = grid.shape

    def draw(t):
        ax.clear()
        ax.set_xlim(-0.5, w-0.5)
        ax.set_ylim(h-0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

        for r in range(h):
            for c in range(w):
                color = 'white'
                if grid[r, c] == 1:
                    color = 'black'
                elif (r, c) == goal_pos:
                    color = 'lime'
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color=color, ec='gray'))

        for b, path in enumerate(box_paths):
            if t < len(path):
                r, c = path[t]
                ax.add_patch(plt.Rectangle((c-0.4, r-0.4), 0.8, 0.8, color='blue'))

        if t < len(robot_path):
            r, c = robot_path[t]
            ax.add_patch(plt.Circle((c, r), 0.3, color='red'))

    ani = animation.FuncAnimation(fig, draw, frames=len(robot_path), interval=500, repeat=False)
    plt.show()

if __name__ == "__main__":
    grid, robot_path, box_paths, goal_pos = solve_sokoban_symbolic()
    if grid is not None:
        animate(grid, robot_path, box_paths, goal_pos)
