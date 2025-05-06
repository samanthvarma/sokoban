from z3 import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def solve_ice_with_multiple_boxes():
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
    box_positions = [tuple(map(int, box)) for box in np.argwhere(grid == 4)]

    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    dir_names = ["up", "right", "down", "left"]
    max_steps = 20

    solver = Solver()
    robot_vars = {}
    box_vars = {}
    actions = {}

    num_boxes = len(box_positions)

    for t in range(max_steps + 1):
        for r in range(height):
            for c in range(width):
                robot_vars[(t, r, c)] = Bool(f"robot_{t}_{r}_{c}")
                for b in range(num_boxes):
                    box_vars[(t, b, r, c)] = Bool(f"box_{t}_{b}_{r}_{c}")

    for t in range(max_steps):
        for d in range(4):
            actions[(t, d)] = Bool(f"action_{t}_{dir_names[d]}")

    def is_obstacle(r, c):
        return r < 0 or r >= height or c < 0 or c >= width or grid[r, c] == 1

    # Initial state
    solver.add(robot_vars[(0, *robot_pos)])
    for r in range(height):
        for c in range(width):
            if (r, c) != robot_pos:
                solver.add(Not(robot_vars[(0, r, c)]))

    for b in range(num_boxes):
        br, bc = box_positions[b]
        solver.add(box_vars[(0, b, br, bc)])
        for r in range(height):
            for c in range(width):
                if (r, c) != (br, bc):
                    solver.add(Not(box_vars[(0, b, r, c)]))

    def next_pos(r, c, d, box_locs):
        dr, dc = directions[d]
        cur_r, cur_c = r, c
        next_r, next_c = cur_r + dr, cur_c + dc

        # If next tile is wall
        if is_obstacle(next_r, next_c):
            return cur_r, cur_c, box_locs.copy()

        # If next tile is a box
        box_index = None
        for idx, (br, bc) in enumerate(box_locs):
            if (next_r, next_c) == (br, bc):
                box_index = idx
                break

        if box_index is not None:
            # Try to push the box
            new_box_r, new_box_c = next_r + dr, next_c + dc
            if is_obstacle(new_box_r, new_box_c) or (new_box_r, new_box_c) in box_locs:
                return cur_r, cur_c, box_locs.copy()  # Cannot push
            # Move robot and box, continue sliding both
            box_locs = box_locs.copy()
            box_locs[box_index] = (new_box_r, new_box_c)
            cur_r, cur_c = next_r, next_c
            while True:
                nr, nc = cur_r + dr, cur_c + dc
                nbr, nbc = box_locs[box_index][0] + dr, box_locs[box_index][1] + dc
                if is_obstacle(nbr, nbc) or (nbr, nbc) in box_locs:
                    break
                cur_r, cur_c = nr, nc
                box_locs[box_index] = (nbr, nbc)
            return cur_r, cur_c, box_locs

        else:
            # No box, slide until wall or box
            while True:
                nr, nc = cur_r + dr, cur_c + dc
                if is_obstacle(nr, nc) or (nr, nc) in box_locs:
                    break
                cur_r, cur_c = nr, nc
            return cur_r, cur_c, box_locs.copy()

    # Constraints over time
    for t in range(max_steps):
        # Only one action
        solver.add(Or([actions[(t, d)] for d in range(4)]))
        for d1 in range(4):
            for d2 in range(d1 + 1, 4):
                solver.add(Or(Not(actions[(t, d1)]), Not(actions[(t, d2)])))

        for r in range(height):
            for c in range(width):
                if is_obstacle(r, c): continue
                for box_positions_t in itertools.product(
                    *[[ (r1, c1) for r1 in range(height) for c1 in range(width)
                        if not is_obstacle(r1, c1) and (r1, c1) != (r, c)] for _ in range(num_boxes)]
                ):
                    if len(set(box_positions_t)) != num_boxes:
                        continue  # skip overlapping box configs

                    preconds = [robot_vars[(t, r, c)]] + [
                        box_vars[(t, b, br, bc)] for b, (br, bc) in enumerate(box_positions_t)
                    ]

                    for d in range(4):
                        new_r, new_c, new_boxes = next_pos(r, c, d, list(box_positions_t))
                        if is_obstacle(new_r, new_c): continue

                        postconds = [robot_vars[(t+1, new_r, new_c)]] + [
                            box_vars[(t+1, b, br, bc)] for b, (br, bc) in enumerate(new_boxes)
                        ]

                        solver.add(Implies(And(*preconds, actions[(t, d)]), And(*postconds)))

        # Enforce one robot position
        solver.add(PbEq([(robot_vars[(t+1, r, c)], 1)
                         for r in range(height)
                         for c in range(width)
                         if not is_obstacle(r, c)], 1))

        # Enforce one position per box
        for b in range(num_boxes):
            solver.add(PbEq([(box_vars[(t+1, b, r, c)], 1)
                             for r in range(height)
                             for c in range(width)
                             if not is_obstacle(r, c)], 1))

    # Goal: robot must reach the goal at some point
    solver.add(Or([robot_vars[(t, goal_pos[0], goal_pos[1])] for t in range(1, max_steps + 1)]))

    print("Solving...")
    result = solver.check()
    print("Result:", result)

    if result != sat:
        print("❌ No solution found.")
        return None, None, None, None

    model = solver.model()
    robot_path = []
    box_paths = [[] for _ in range(num_boxes)]

    for t in range(max_steps + 1):
        for r in range(height):
            for c in range(width):
                if is_true(model.evaluate(robot_vars[(t, r, c)])):
                    robot_path.append((r, c))
                    break

        for b in range(num_boxes):
            for r in range(height):
                for c in range(width):
                    if is_true(model.evaluate(box_vars[(t, b, r, c)])):
                        box_paths[b].append((r, c))
                        break

    print("✅ Found path with multiple boxes.")
    return grid, robot_path, box_paths, goal_pos

def animate_solution(grid, robot_path, box_paths, goal_pos):
    fig, ax = plt.subplots(figsize=(8, 6))
    h, w = grid.shape

    def draw(t):
        ax.clear()
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-0.5, w - 0.5); ax.set_ylim(h - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.set_title(f"Step {t}/{len(robot_path)-1}")

        for r in range(h):
            for c in range(w):
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, fill=False, edgecolor='gray'))
                if grid[r, c] == 1:
                    ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='black'))
                elif (r, c) == goal_pos:
                    ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='lime'))

        rr, rc = robot_path[t]
        ax.add_patch(plt.Circle((rc, rr), 0.3, color='red'))

        for b in range(len(box_paths)):
            br, bc = box_paths[b][t]
            ax.add_patch(plt.Rectangle((bc-0.4, br-0.4), 0.8, 0.8, color='blue'))

    anim = animation.FuncAnimation(fig, draw, frames=len(robot_path), interval=700, repeat=True)
    anim.save("scene_28.gif", writer='pillow')
    plt.close()
    print("🎥 Animation saved as 'scene_28.gif'")

if __name__ == "__main__":
    import itertools
    grid, robot_path, box_paths, goal_pos = solve_ice_with_multiple_boxes()
    if grid is not None:
        animate_solution(grid, robot_path, box_paths, goal_pos)
