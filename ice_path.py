from z3 import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def solve_ice_robot_only():
    grid = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 3, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 2, 1, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ])

    height, width = grid.shape
    robot_pos = tuple(map(int, np.argwhere(grid == 2)[0]))
    goal_pos = tuple(map(int, np.argwhere(grid == 3)[0]))

    print("Initial Grid (no box):")
    print_grid_values(grid)
    print(f"Robot: {robot_pos}, Goal: {goal_pos}")

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
        return r < 0 or r >= height or c < 0 or c >= width or grid[r, c] == 1

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

    print("Solving (no box)...")
    result = solver.check()
    print("SAT result:", result)

    if result != sat:
        print("No path found.")
        return None, None, None

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

    print(f"Found path in {len(plan)} steps.")
    return grid, robot_path, goal_pos

def print_grid_values(grid):
    for r in range(grid.shape[0]):
        print(' '.join(str(int(v)) for v in grid[r]))

def animate_robot_only(grid, robot_path, goal_pos):
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
                if grid[r, c] == 1:
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='black'))
                elif (r, c) == goal_pos:
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='lime'))

        rr, rc = robot_path[t]
        ax.add_patch(plt.Circle((rc, rr), 0.3, color='red'))

    ani = animation.FuncAnimation(fig, draw_frame, frames=len(robot_path), interval=700, repeat=False)
    ani.save("Ice_path.gif", writer='pillow')
    print("✅ Animation saved as 'Ice_path.gif'")

if __name__ == "__main__":
    grid, robot_path, goal_pos = solve_ice_robot_only()
    if grid is not None:
        animate_robot_only(grid, robot_path, goal_pos)
