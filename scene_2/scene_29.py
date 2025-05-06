from z3 import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def solve_ice_with_boxes():
    grid = np.array([
        [2, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0],
        [0, 0, 3, 0, 4, 0, 0],
        [0, 0, 0, 0, 0, 4, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
    ])

    height, width = grid.shape
    robot_pos = tuple(map(int, np.argwhere(grid == 2)[0]))
    goal_pos = tuple(map(int, np.argwhere(grid == 3)[0]))
    box_positions = [tuple(map(int, b)) for b in np.argwhere(grid == 4)]
    num_boxes = len(box_positions)

    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    dir_names = ["up", "right", "down", "left"]
    max_steps = 30

    solver = Solver()
    robot_vars = {}
    box_vars = {}
    actions = {}

    # Variables
    for t in range(max_steps + 1):
        for r in range(height):
            for c in range(width):
                robot_vars[(t, r, c)] = Bool(f"robot_{t}_{r}_{c}")
                for b in range(num_boxes):
                    box_vars[(t, b, r, c)] = Bool(f"box_{t}_b{b}_{r}_{c}")

    for t in range(max_steps):
        for d in range(4):
            actions[(t, d)] = Bool(f"action_{t}_{dir_names[d]}")

    def is_obstacle(r, c):
        return r < 0 or r >= height or c < 0 or c >= width or grid[r, c] == 1

    def next_position(r, c, d, box_map):
        """Returns: new_r, new_c, box_moved: (idx, new_r, new_c) or None"""
        dr, dc = directions[d]
        curr_r, curr_c = r, c
        next_r, next_c = curr_r + dr, curr_c + dc

        if is_obstacle(next_r, next_c):
            return r, c, None

        for i, (br, bc) in enumerate(box_map):
            if (next_r, next_c) == (br, bc):
                box_next_r, box_next_c = br + dr, bc + dc
                if is_obstacle(box_next_r, box_next_c) or (box_next_r, box_next_c) in box_map:
                    return r, c, None
                # Start sliding both
                curr_r, curr_c = next_r, next_c
                new_box_r, new_box_c = box_next_r, box_next_c
                while True:
                    next_r, next_c = curr_r + dr, curr_c + dc
                    box_next_r, box_next_c = new_box_r + dr, new_box_c + dc
                    if is_obstacle(box_next_r, box_next_c) or (box_next_r, box_next_c) in box_map:
                        return curr_r, curr_c, (i, new_box_r, new_box_c)
                    curr_r, curr_c = next_r, next_c
                    new_box_r, new_box_c = box_next_r, box_next_c
        # No box hit initially, just slide
        while True:
            next_r, next_c = curr_r + dr, curr_c + dc
            if is_obstacle(next_r, next_c) or (next_r, next_c) in box_map:
                return curr_r, curr_c, None
            curr_r, curr_c = next_r, next_c

    # Initial positions
    solver.add(robot_vars[(0, *robot_pos)])
    for r in range(height):
        for c in range(width):
            if (r, c) != robot_pos:
                solver.add(Not(robot_vars[(0, r, c)]))
    for b, (br, bc) in enumerate(box_positions):
        solver.add(box_vars[(0, b, br, bc)])
        for r in range(height):
            for c in range(width):
                if (r, c) != (br, bc):
                    solver.add(Not(box_vars[(0, b, r, c)]))

    # Constraints
    for t in range(max_steps):
        solver.add(Or([actions[(t, d)] for d in range(4)]))
        for d1 in range(4):
            for d2 in range(d1 + 1, 4):
                solver.add(Or(Not(actions[(t, d1)]), Not(actions[(t, d2)])))

        for r in range(height):
            for c in range(width):
                if is_obstacle(r, c): continue
                for b_map in itertools.product(*[[(r2, c2) for r2 in range(height) for c2 in range(width) if not is_obstacle(r2, c2)] for _ in range(num_boxes)]):
                    if len(set(b_map)) != num_boxes or (r, c) in b_map: continue
                    for d in range(4):
                        pre = And(robot_vars[(t, r, c)],
                                  *[box_vars[(t, i, br, bc)] for i, (br, bc) in enumerate(b_map)],
                                  actions[(t, d)])
                        r_new, c_new, box_move = next_position(r, c, d, b_map)
                        solver.add(Implies(pre, robot_vars[(t+1, r_new, c_new)]))
                        for i in range(num_boxes):
                            if box_move and i == box_move[0]:
                                solver.add(Implies(pre, box_vars[(t+1, i, box_move[1], box_move[2])]))
                            else:
                                br, bc = b_map[i]
                                solver.add(Implies(pre, box_vars[(t+1, i, br, bc)]))

        solver.add(PbEq([(robot_vars[(t+1, r, c)], 1) for r in range(height) for c in range(width) if not is_obstacle(r, c)], 1))
        for i in range(num_boxes):
            solver.add(PbEq([(box_vars[(t+1, i, r, c)], 1) for r in range(height) for c in range(width) if not is_obstacle(r, c)], 1))
            for r in range(height):
                for c in range(width):
                    if not is_obstacle(r, c):
                        solver.add(Implies(box_vars[(t+1, i, r, c)], Not(robot_vars[(t+1, r, c)])))

    solver.add(Or([robot_vars[(t, goal_pos[0], goal_pos[1])] for t in range(1, max_steps + 1)]))

    print("Solving...")
    if solver.check() != sat:
        print("❌ No solution.")
        return None, None, None, None

    model = solver.model()
    robot_path = [robot_pos]
    box_paths = [ [b] for b in box_positions ]
    for t in range(max_steps):
        for r in range(height):
            for c in range(width):
                if is_true(model.evaluate(robot_vars[(t+1, r, c)])):
                    robot_path.append((r, c))
                    break
            else: continue
            break
        for i in range(num_boxes):
            for r in range(height):
                for c in range(width):
                    if is_true(model.evaluate(box_vars[(t+1, i, r, c)])):
                        box_paths[i].append((r, c))
                        break
                else: continue
                break
        if robot_path[-1] == goal_pos:
            break

    return grid, robot_path, box_paths, goal_pos

def animate_solution(grid, robot_path, box_paths, goal_pos):
    fig, ax = plt.subplots()
    h, w = grid.shape

    def draw_frame(t):
        ax.clear()
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Step {t}")
        for r in range(h):
            for c in range(w):
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, fill=False))
                if grid[r, c] == 1:
                    ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='black'))
                elif (r, c) == goal_pos:
                    ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='lime'))
        for bp in box_paths:
            br, bc = bp[min(t, len(bp)-1)]
            ax.add_patch(plt.Rectangle((bc - 0.4, br - 0.4), 0.8, 0.8, color='blue'))
        rr, rc = robot_path[min(t, len(robot_path)-1)]
        ax.add_patch(plt.Circle((rc, rr), 0.3, color='red'))

    ani = animation.FuncAnimation(fig, draw_frame, frames=len(robot_path), interval=700)
    ani.save("multi_box_scene.gif", writer='pillow')
    plt.close()
    print("✅ Animation saved as multi_box_scene.gif")

if __name__ == "__main__":
    import itertools
    grid, robot_path, box_paths, goal_pos = solve_ice_with_boxes()
    if grid is not None:
        animate_solution(grid, robot_path, box_paths, goal_pos)
