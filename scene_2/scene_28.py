# -*- coding: utf-8 -*-
from z3 import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Directions
DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_NAMES = ["up", "right", "down", "left"]

def solve_ice_with_boxes():
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

    H, W = grid.shape
    max_steps = 30
    robot_pos = tuple(map(int, np.argwhere(grid == 2)[0]))
    goal_pos = tuple(map(int, np.argwhere(grid == 3)[0]))
    box_positions = [tuple(map(int, pos)) for pos in np.argwhere(grid == 4)]
    num_boxes = len(box_positions)

    solver = Solver()
    robot_vars = {}
    box_vars = {}
    actions = {}

    def is_obstacle(r, c):
        return r < 0 or r >= H or c < 0 or c >= W or grid[r, c] == 1

    def next_position(r, c, d, box_r, box_c):
        dr, dc = DIRECTIONS[d]
        curr_r, curr_c = r, c
        box_moved = False
        box_new_r, box_new_c = box_r, box_c
        next_r, next_c = curr_r + dr, curr_c + dc

        if is_obstacle(next_r, next_c):
            return curr_r, curr_c, box_moved, box_new_r, box_new_c

        if (next_r, next_c) == (box_r, box_c):
            box_next_r, box_next_c = box_r + dr, box_c + dc
            if is_obstacle(box_next_r, box_next_c):
                return curr_r, curr_c, box_moved, box_r, box_c
            box_moved = True
            box_new_r, box_new_c = box_next_r, box_next_c
            curr_r, curr_c = next_r, next_c
            while True:
                next_r, next_c = curr_r + dr, curr_c + dc
                box_next_r, box_next_c = box_new_r + dr, box_new_c + dc
                if is_obstacle(box_next_r, box_next_c):
                    return curr_r, curr_c, box_moved, box_new_r, box_new_c
                curr_r, curr_c = next_r, next_c
                box_new_r, box_new_c = box_next_r, box_next_c
        else:
            curr_r, curr_c = next_r, next_c
            while True:
                next_r, next_c = curr_r + dr, curr_c + dc
                if is_obstacle(next_r, next_c) or (next_r, next_c) == (box_r, box_c):
                    return curr_r, curr_c, box_moved, box_r, box_c
                curr_r, curr_c = next_r, next_c

    for t in range(max_steps + 1):
        for r in range(H):
            for c in range(W):
                robot_vars[(t, r, c)] = Bool(f"robot_{t}_{r}_{c}")
                for b in range(num_boxes):
                    box_vars[(t, b, r, c)] = Bool(f"box_{t}_b{b}_{r}_{c}")

    for t in range(max_steps):
        for d in range(4):
            actions[(t, d)] = Bool(f"action_{t}_{DIR_NAMES[d]}")

    solver.add(robot_vars[(0, *robot_pos)])
    for r in range(H):
        for c in range(W):
            if (r, c) != robot_pos:
                solver.add(Not(robot_vars[(0, r, c)]))

    for b, (br, bc) in enumerate(box_positions):
        solver.add(box_vars[(0, b, br, bc)])
        for r in range(H):
            for c in range(W):
                if (r, c) != (br, bc):
                    solver.add(Not(box_vars[(0, b, r, c)]))

    for t in range(max_steps):
        solver.add(Or([actions[(t, d)] for d in range(4)]))
        for d1 in range(4):
            for d2 in range(d1 + 1, 4):
                solver.add(Or(Not(actions[(t, d1)]), Not(actions[(t, d2)])))

        for r in range(H):
            for c in range(W):
                if is_obstacle(r, c): continue
                for b in range(num_boxes):
                    for box_r in range(H):
                        for box_c in range(W):
                            if is_obstacle(box_r, box_c) or (r, c) == (box_r, box_c): continue
                            for d in range(4):
                                r_new, c_new, box_moved, b_new_r, b_new_c = next_position(r, c, d, box_r, box_c)
                                pre = And(robot_vars[(t, r, c)], box_vars[(t, b, box_r, box_c)], actions[(t, d)])
                                solver.add(Implies(pre, robot_vars[(t+1, r_new, c_new)]))
                                for b2 in range(num_boxes):
                                    if b2 == b:
                                        if box_moved:
                                            solver.add(Implies(pre, box_vars[(t+1, b2, b_new_r, b_new_c)]))
                                        else:
                                            solver.add(Implies(pre, box_vars[(t+1, b2, box_r, box_c)]))
                                    else:
                                        for r2 in range(H):
                                            for c2 in range(W):
                                                solver.add(Implies(pre, box_vars[(t+1, b2, r2, c2)] == box_vars[(t, b2, r2, c2)]))

        solver.add(PbEq([(robot_vars[(t+1, r, c)], 1) for r in range(H) for c in range(W) if not is_obstacle(r, c)], 1))
        for b in range(num_boxes):
            solver.add(PbEq([(box_vars[(t+1, b, r, c)], 1) for r in range(H) for c in range(W) if not is_obstacle(r, c)], 1))
        for r in range(H):
            for c in range(W):
                if not is_obstacle(r, c):
                    for b in range(num_boxes):
                        solver.add(Not(And(robot_vars[(t+1, r, c)], box_vars[(t+1, b, r, c)])))

    solver.add(Or([robot_vars[(t, goal_pos[0], goal_pos[1])] for t in range(1, max_steps + 1)]))

    print("Solving...")
    if solver.check() != sat:
        print("❌ No solution found")
        return None, None, None, None

    model = solver.model()
    robot_path = [robot_pos]
    box_paths = [[pos] for pos in box_positions]
    for t in range(max_steps):
        for r in range(H):
            for c in range(W):
                if is_true(model.evaluate(robot_vars[(t+1, r, c)])):
                    robot_path.append((r, c))
                    break
            else: continue
            break

        for b in range(num_boxes):
            for r in range(H):
                for c in range(W):
                    if is_true(model.evaluate(box_vars[(t+1, b, r, c)])):
                        box_paths[b].append((r, c))
                        break
                else: continue
                break

        if robot_path[-1] == goal_pos:
            break

    print(f"✅ Found path with {len(robot_path)-1} steps")
    return grid, robot_path, box_paths, goal_pos

def animate_solution(grid, robot_path, box_paths, goal_pos):
    fig, ax = plt.subplots(figsize=(10, 8))
    H, W = grid.shape

    def draw(t):
        ax.clear()
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5)
        ax.set_title(f"Step {t}/{len(robot_path)-1}")

        for r in range(H):
            for c in range(W):
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor='gray'))
                if grid[r, c] == 1:
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='black'))
                elif (r, c) == goal_pos:
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='lime'))

        for box_path in box_paths:
            br, bc = box_path[min(t, len(box_path)-1)]
            ax.add_patch(plt.Rectangle((bc - 0.4, br - 0.4), 0.8, 0.8, color='blue'))
        rr, rc = robot_path[min(t, len(robot_path)-1)]
        ax.add_patch(plt.Circle((rc, rr), 0.3, color='red'))

    ani = animation.FuncAnimation(fig, draw, frames=len(robot_path), interval=700, repeat=True)
    ani.save("scene_1.gif", writer='pillow')
    plt.close()
    print("✅ Animation saved as 'scene_1.gif'")

if __name__ == "__main__":
    grid, robot_path, box_paths, goal_pos = solve_ice_with_boxes()
    if grid is not None:
        animate_solution(grid, robot_path, box_paths, goal_pos)
