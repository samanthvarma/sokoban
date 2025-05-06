# -*- coding: utf-8 -*-
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
        # (Initialization code remains the same as before)
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

        # Calculate valid cells (excluding walls)
        self.valid_cells = []
        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r, c] != 1: # Check only for walls initially
                     self.valid_cells.append((r, c))

        print(f"Puzzle initialized with {self.num_boxes} boxes")
        print(f"Robot starts at {self.robot_pos}")
        print(f"Goal at {self.goal_pos}")
        print(f"Boxes at {self.box_positions}")

    def is_obstacle(self, r, c, current_grid):
        """Check if a cell is a wall or out of bounds."""
        if r < 0 or r >= self.height or c < 0 or c >= self.width or current_grid[r, c] == 1:
            return True
        return False

    def calculate_stop_position(self, start_r, start_c, dr, dc, current_grid, current_box_positions, ignore_box_idx=None):
        """
        Find where a SINGLE object (robot OR box) stops sliding.
        It stops *in* the cell before the obstacle.
        Used primarily for robot sliding freely or potentially for pre-computation.
        """
        curr_r, curr_c = start_r, start_c
        while True:
            next_r, next_c = curr_r + dr, curr_c + dc

            # Check for grid boundaries or static walls
            if self.is_obstacle(next_r, next_c, current_grid):
                return (curr_r, curr_c) # Stop before the wall/boundary

            # Check for collision with ANY box (unless we are ignoring one)
            hit_another_box = False
            for idx, (br, bc) in enumerate(current_box_positions):
                # When called for a sliding robot (ignore_box_idx is None), it checks all boxes.
                # When called for a sliding box (e.g. precomputation), it ignores itself.
                if idx == ignore_box_idx:
                    continue
                if (next_r, next_c) == (br, bc):
                    hit_another_box = True
                    break

            if hit_another_box:
                return (curr_r, curr_c) # Stop before the other box

            # If no obstacle, continue sliding to the next cell
            curr_r, curr_c = next_r, next_c

    def next_position(self, robot_r, robot_c, direction_idx, current_box_positions, current_grid):
        """
        Calculate the final position of the robot and any moved box after one action.
        Handles sliding robot and pushed sliding boxes according to the rules.
        Robot moves with the box and stops in the cell the box occupied just before stopping.
        """
        dr, dc = directions[direction_idx]
        original_robot_pos = (robot_r, robot_c)

        # Calculate robot's immediate next step position
        robot_next_r, robot_next_c = robot_r + dr, robot_c + dc

        # --- Case 1: Robot hits a wall immediately ---
        if self.is_obstacle(robot_next_r, robot_next_c, current_grid):
            return original_robot_pos, []

        # --- Check if the next step hits a box ---
        hit_box_idx = -1
        box_start_pos = None
        for idx, (br, bc) in enumerate(current_box_positions):
            if (robot_next_r, robot_next_c) == (br, bc):
                hit_box_idx = idx
                box_start_pos = (br, bc)
                break

        # --- Case 2: Robot hits a box (Revised Logic) ---
        if hit_box_idx != -1:
            # Check if the space IMMEDIATELY BEYOND the box is blocked (wall or ANOTHER box)
            box_after_r, box_after_c = box_start_pos[0] + dr, box_start_pos[1] + dc
            blocked = False
            if self.is_obstacle(box_after_r, box_after_c, current_grid):
                blocked = True
            else:
                for idx, (other_br, other_bc) in enumerate(current_box_positions):
                    if idx != hit_box_idx and (box_after_r, box_after_c) == (other_br, other_bc):
                        blocked = True
                        break

            if blocked:
                # Box cannot be pushed, so robot stays put
                return original_robot_pos, []
            else:
                # Simulate the slide step-by-step for robot and pushed box together
                current_robot_pos = box_start_pos # Robot moves into the box's initial cell
                current_box_pos = (box_start_pos[0] + dr, box_start_pos[1] + dc) # Box moves one step initially

                while True:
                    # Where will the box move next?
                    next_box_r, next_box_c = current_box_pos[0] + dr, current_box_pos[1] + dc

                    # Is the next box position blocked by a wall/boundary?
                    if self.is_obstacle(next_box_r, next_box_c, current_grid):
                        final_robot_pos = current_robot_pos
                        final_box_pos = current_box_pos
                        break # Box stops here

                    # Is the next box position blocked by ANOTHER box?
                    hit_another_box = False
                    for idx, (other_br, other_bc) in enumerate(current_box_positions):
                        if idx != hit_box_idx and (next_box_r, next_box_c) == (other_br, other_bc):
                            hit_another_box = True
                            break

                    if hit_another_box:
                        final_robot_pos = current_robot_pos
                        final_box_pos = current_box_pos
                        break # Box stops here

                    # If not blocked, advance both robot and box
                    current_robot_pos = current_box_pos       # Robot moves to where the box was
                    current_box_pos = (next_box_r, next_box_c) # Box moves to the next cell

                # Return the final calculated positions
                # Check if any movement actually occurred compared to the initial push state
                if final_robot_pos == box_start_pos and final_box_pos == (box_start_pos[0] + dr, box_start_pos[1] + dc):
                     # This means the box moved exactly one step and stopped. Robot is where the box started.
                     pass # This is a valid move
                elif final_robot_pos == original_robot_pos:
                    # Should not happen if blocked==False, but as a safeguard
                    # If robot ended up where it started, means no effective move happened.
                     return original_robot_pos, []


                # Only return a move if the final state is different from the start
                if final_robot_pos == original_robot_pos and final_box_pos == box_start_pos:
                    return original_robot_pos, []
                else:
                    return final_robot_pos, [(hit_box_idx, final_box_pos)]

        # --- Case 3: Robot slides into empty space ---
        else:
            # Robot slides freely from its CURRENT position
            # Use calculate_stop_position for this simpler case
            final_robot_pos = self.calculate_stop_position(
                robot_r, robot_c, dr, dc, current_grid,
                current_box_positions # Checks collision with ALL boxes
            )

            if final_robot_pos == original_robot_pos:
                return original_robot_pos, [] # No movement
            else:
                return final_robot_pos, []

    # --- Rest of the class methods remain the same ---
    # create_state_space_graph, extract_path, solve_with_state_space_search,
    # solve_with_z3, solve, animate_solution

    def create_state_space_graph(self, max_states=50000):
        """
        Create a state space graph for more efficient planning.
        Each node is a state (robot_pos, tuple(sorted(box_positions))).
        Each edge is an action that transitions between states.
        (Code is identical to previous version, uses the updated next_position)
        """
        print("Building state space graph...")
        graph = {}
        backtrack = {}
        initial_state = (self.robot_pos, tuple(sorted(self.box_positions)))
        queue = [initial_state]
        visited = {initial_state}
        goal_state = None
        state_count = 0
        while queue and state_count < max_states:
            current_state = queue.pop(0)
            state_count += 1
            if state_count % 5000 == 0:
                print(f"Processed {state_count} states, queue size: {len(queue)}")

            robot_pos, box_positions_tuple = current_state
            box_positions_list = list(box_positions_tuple)

            if robot_pos == self.goal_pos:
                 # Found a potential goal state
                 if goal_state is None: # Keep the first one found by BFS
                      goal_state = current_state
                      print(f"Goal reached after exploring {state_count} states!")
                      # We can break here for BFS shortest path, or continue for Z3 precomputation
                      # break # Uncomment for pure BFS shortest path

            graph[current_state] = {}
            for d in range(4):
                (new_robot_pos, moved_boxes) = self.next_position(
                    robot_pos[0], robot_pos[1], d, box_positions_list, self.grid
                )

                new_box_positions_list = list(box_positions_list)
                for box_idx, new_pos in moved_boxes:
                    # The index corresponds to the sorted list
                    new_box_positions_list[box_idx] = new_pos

                next_state_box_tuple = tuple(sorted(new_box_positions_list)) # Ensure canonical state
                next_state = (new_robot_pos, next_state_box_tuple)

                if next_state != current_state:
                    graph[current_state][d] = next_state
                    if next_state not in visited:
                        visited.add(next_state)
                        queue.append(next_state)
                        backtrack[next_state] = (current_state, d)

        print(f"State space exploration complete. Explored {state_count} states.")
        print(f"Total unique states visited: {len(visited)}")

        # Find the shortest goal state reachable if multiple goal states exist in visited
        shortest_len = float('inf')
        final_goal_state = None
        if goal_state: # If we found one during BFS
             path_len = 0
             temp = goal_state
             while temp in backtrack:
                  temp, _ = backtrack[temp]
                  path_len += 1
             shortest_len = path_len
             final_goal_state = goal_state

        # Check all visited states just in case BFS was allowed to continue past the first goal
        for state in visited:
             if state[0] == self.goal_pos:
                  path_len = 0
                  temp = state
                  has_path = True
                  while temp in backtrack:
                       temp, _ = backtrack[temp]
                       path_len += 1
                  # Ensure it's actually reachable from start (initial state has no backtrack entry)
                  if temp != initial_state and state != initial_state :
                      has_path = False # Should not happen if state is in visited and not initial

                  if has_path and path_len < shortest_len:
                       shortest_len = path_len
                       final_goal_state = state

        if final_goal_state:
            print(f"Shortest path goal state found: {final_goal_state} with length {shortest_len}")
        else:
             # Check if start state is goal state
             if initial_state[0] == self.goal_pos:
                  print("Start state is the goal state.")
                  final_goal_state = initial_state
             else:
                  print("Goal state not found or not reachable.")


        return graph, backtrack, final_goal_state


    def extract_path(self, backtrack, goal_state):
        """Extract path from goal to start using backtrack dictionary."""
        # Check if goal is reachable or is the start state
        if goal_state is None:
            print("Error: Goal state is None.")
            return None, None, None
        if goal_state not in backtrack and goal_state != (self.robot_pos, tuple(sorted(self.box_positions))):
             print(f"Error: Goal state {goal_state} not found in backtrack keys and is not the initial state.")
             return None, None, None

        # Handle goal == start state
        initial_state_tuple = (self.robot_pos, tuple(sorted(self.box_positions)))
        if goal_state == initial_state_tuple:
             return [self.robot_pos], [[bp] for bp in self.box_positions], [] # Path of length 0

        current = goal_state
        num_boxes = len(self.box_positions)

        # Paths stored in reverse (end to start)
        robot_path_rev = [current[0]]
        # Store paths based on the *sorted* order used in states
        box_paths_rev_dict = {idx: [current[1][idx]] for idx in range(num_boxes)}
        actions_rev = []

        while current in backtrack:
            prev_state, action = backtrack[current]
            robot_path_rev.append(prev_state[0])
            # Append previous box positions based on sorted order
            for box_idx in range(num_boxes):
                box_paths_rev_dict[box_idx].append(prev_state[1][box_idx])
            actions_rev.append(action)
            current = prev_state

            # Safety break
            if len(actions_rev) > len(backtrack) * 2: # Generous limit
                 print("Error: Path extraction seems stuck in a loop.")
                 return None, None, None

        # Reverse paths and actions
        robot_path = list(reversed(robot_path_rev))
        actions = list(reversed(actions_rev))

        # Reverse the individual sorted box paths
        sorted_box_paths = [list(reversed(box_paths_rev_dict[i])) for i in range(num_boxes)]

        # --- Map sorted box paths back to original box order ---
        # Create a mapping from the initial sorted position tuple to the original index
        initial_unsorted_boxes = self.box_positions
        initial_sorted_boxes_tuple = tuple(sorted(initial_unsorted_boxes))
        # Need a robust way to know which *original* box corresponds to the i-th element
        # in the sorted tuple at the start. Let's map initial position to original index.
        initial_pos_to_orig_idx = {pos: idx for idx, pos in enumerate(initial_unsorted_boxes)}

        # Determine the order of original indices corresponding to the initial sorted tuple
        initial_sorted_orig_indices = [initial_pos_to_orig_idx[pos] for pos in initial_sorted_boxes_tuple]

        # Create the final paths list, ordered by original index
        final_box_paths = [[] for _ in range(num_boxes)]
        path_length = len(robot_path)

        for step in range(path_length):
            # For each step, assign the position from the i-th sorted path
            # to the box whose *original* index corresponds to the i-th position
            # in the *initial* sorted list.
            for i in range(num_boxes):
                original_box_index = initial_sorted_orig_indices[i]
                final_box_paths[original_box_index].append(sorted_box_paths[i][step])

        return robot_path, final_box_paths, actions




    def solve_with_z3(self, max_steps=20):
        """ Solve using Z3 SMT solver """
        # (Identical to previous version - uses updated next_position via precomputation)
        # Note: Z3 precomputation implicitly uses the new next_position logic
        print(f"Initializing Z3 solver with max_steps={max_steps}...")
        solver = Solver()
        solver.set(timeout=120000) # 120 second timeout

        # --- Step 1: Precompute State Space ---
        state_to_id = {}
        id_to_state = {}
        transitions = {} # (state_id, action) -> next_state_id
        initial_state = (self.robot_pos, tuple(sorted(self.box_positions)))
        state_to_id[initial_state] = 0
        id_to_state[0] = initial_state
        next_state_id = 1
        queue = [initial_state]
        visited = {initial_state}
        print("Precomputing state space for Z3...")
        processed_count = 0
        while queue:
            current_state = queue.pop(0)
            current_id = state_to_id[current_state]
            processed_count += 1
            if processed_count % 10000 == 0: print(f" Precomputing... processed {processed_count} states.")

            robot_pos, box_positions_tuple = current_state
            box_positions_list = list(box_positions_tuple)

            for d in range(4):
                (new_robot_pos, moved_boxes) = self.next_position(
                    robot_pos[0], robot_pos[1], d, box_positions_list, self.grid
                )
                new_box_positions_list = list(box_positions_list)
                for box_idx, new_pos in moved_boxes:
                     new_box_positions_list[box_idx] = new_pos # Index is correct for sorted list

                next_state_box_tuple = tuple(sorted(new_box_positions_list))
                next_state = (new_robot_pos, next_state_box_tuple)

                if next_state not in state_to_id:
                    state_to_id[next_state] = next_state_id
                    id_to_state[next_state_id] = next_state
                    next_id = next_state_id
                    next_state_id += 1
                    if next_state not in visited:
                        visited.add(next_state)
                        queue.append(next_state)
                else:
                    next_id = state_to_id[next_state]

                if next_id != current_id:
                    transitions[(current_id, d)] = next_id

        print(f"Precomputed {len(state_to_id)} unique states")
        num_states = len(state_to_id)
        if num_states == 0:
             print("Error: No states generated during precomputation.")
             return None, None, None


        # --- Step 2: Define Z3 Variables ---
        states = {} # (t, state_id) -> Bool
        actions = {} # (t, action) -> Bool
        for t in range(max_steps + 1):
            for state_id in range(num_states): states[(t, state_id)] = Bool(f"state_{t}_{state_id}")
        for t in range(max_steps):
            for d in range(4): actions[(t, d)] = Bool(f"action_{t}_{d}")

        # --- Step 3: Add Z3 Constraints ---
        solver.add(states[(0, 0)]) # Initial state
        for state_id in range(1, num_states): solver.add(Not(states[(0, state_id)]))

        for t in range(max_steps):
            solver.add(PbEq([(actions[(t, d)], 1) for d in range(4)], 1)) # Exactly one action
            # Transition constraints
            for current_id in range(num_states):
                for d in range(4):
                    condition = And(states[(t, current_id)], actions[(t, d)])
                    next_id = transitions.get((current_id, d), current_id) # Stay if no transition
                    solver.add(Implies(condition, states[(t + 1, next_id)]))
            # Ensure exactly one state is true at t+1 (redundant but can help)
            solver.add(PbEq([(states[(t + 1, sid)], 1) for sid in range(num_states)], 1))


        # Goal constraint
        goal_state_ids = [sid for sid, state in id_to_state.items() if state[0] == self.goal_pos]
        if not goal_state_ids:
             print("Error: No goal state ID found during precomputation.")
             # Check if start is goal
             if id_to_state[0][0] == self.goal_pos:
                  print("Start state is the goal.")
                  return [self.robot_pos], [[bp] for bp in self.box_positions], []
             return None, None, None

        goal_constraints = [states[(t, gid)] for t in range(1, max_steps + 1) for gid in goal_state_ids]
        if not goal_constraints:
             print("Warning: No goal constraints generated (max_steps might be 0?)")
             # Check if start is goal
             if id_to_state[0][0] == self.goal_pos:
                  print("Start state is the goal.")
                  return [self.robot_pos], [[bp] for bp in self.box_positions], []
             return None, None, None
        solver.add(Or(goal_constraints))


        # --- Step 4: Solve ---
        print("Solving with Z3...")
        start_time = time.time(); result = solver.check(); end_time = time.time()
        print(f"Z3 Solving took {end_time - start_time:.2f} seconds. Result: {result}")

        # --- Step 5: Extract Solution ---
        if result == sat:
            model = solver.model()
            final_time = -1; goal_reached_id = -1
            for t in range(max_steps + 1): # Check from t=0
                for gid in goal_state_ids:
                    if is_true(model.eval(states[(t, gid)], model_completion=True)):
                        final_time = t; goal_reached_id = gid; break
                if final_time != -1: break

            if final_time == -1: print("Error: Solver sat but no goal state found in model."); return None, None, None
            print(f"Goal reached at step {final_time}")
            if final_time == 0: return [self.robot_pos], [[bp] for bp in self.box_positions], []


            # Reconstruct path
            actual_actions = []
            path_states_ids = [-1] * (final_time + 1)
            path_states_ids[final_time] = goal_reached_id
            current_state_id = goal_reached_id

            for t in range(final_time - 1, -1, -1):
                found_prev = False
                # Find the state and action at time t that led to current_state_id at t+1
                possible_prev_states = [sid for sid in range(num_states) if is_true(model.eval(states[(t, sid)], model_completion=True))]
                possible_actions = [d for d in range(4) if is_true(model.eval(actions[(t, d)], model_completion=True))]

                if not possible_prev_states or not possible_actions:
                     print(f"Error reconstructing path at step {t+1}: No active state/action found at step {t}.")
                     return None, None, None # Path reconstruction failed

                prev_state_id = possible_prev_states[0] # Should only be one
                action_taken = possible_actions[0]      # Should only be one

                # Verify transition
                expected_next = transitions.get((prev_state_id, action_taken), prev_state_id)
                if expected_next == current_state_id:
                     actual_actions.append(action_taken)
                     path_states_ids[t] = prev_state_id
                     current_state_id = prev_state_id
                     found_prev = True
                else:
                     print(f"Error reconstructing path at step {t+1}: Transition mismatch.")
                     print(f" State[{t}]={prev_state_id}, Action[{t}]={action_taken} -> Expected {expected_next}, but got State[{t+1}]={current_state_id}")
                     # Attempt to find *any* state/action pair that works (might indicate solver issue)
                     found_alternative = False
                     for pid_alt in range(num_states):
                          if not is_true(model.eval(states[(t, pid_alt)], model_completion=True)): continue
                          for d_alt in range(4):
                               if not is_true(model.eval(actions[(t, d_alt)], model_completion=True)): continue
                               expected_next_alt = transitions.get((pid_alt, d_alt), pid_alt)
                               if expected_next_alt == path_states_ids[t+1]: # Use the known next state ID
                                    print(f" Found alternative: State[{t}]={pid_alt}, Action[{t}]={d_alt} -> {expected_next_alt}")
                                    actual_actions.append(d_alt)
                                    path_states_ids[t] = pid_alt
                                    current_state_id = pid_alt
                                    found_alternative = True
                                    break
                          if found_alternative: break
                     if not found_alternative:
                          print(" Critical Error: Could not find valid previous step.")
                          return None, None, None # Path reconstruction failed

            actual_actions.reverse()
            print(f"Solution length: {len(actual_actions)} steps")
            print(f"Actions: {[dir_names[a] for a in actual_actions]}")

            # Generate paths from state IDs
            robot_path = [id_to_state[sid][0] for sid in path_states_ids if sid !=-1]
            num_boxes = len(self.box_positions)
            sorted_box_paths_dict = {i: [] for i in range(num_boxes)}
            for state_id in path_states_ids:
                 if state_id != -1:
                      sorted_box_pos_tuple = id_to_state[state_id][1]
                      for box_idx in range(num_boxes):
                           sorted_box_paths_dict[box_idx].append(sorted_box_pos_tuple[box_idx])

            # Map sorted box paths back to original box order (using the logic from extract_path)
            initial_unsorted_boxes = self.box_positions
            initial_sorted_boxes_tuple = tuple(sorted(initial_unsorted_boxes))
            initial_pos_to_orig_idx = {pos: idx for idx, pos in enumerate(initial_unsorted_boxes)}
            initial_sorted_orig_indices = [initial_pos_to_orig_idx[pos] for pos in initial_sorted_boxes_tuple]
            final_box_paths = [[] for _ in range(num_boxes)]
            path_length_found = len(robot_path)
            for step in range(path_length_found):
                 for i in range(num_boxes):
                      original_box_index = initial_sorted_orig_indices[i]
                      final_box_paths[original_box_index].append(sorted_box_paths_dict[i][step])


            return robot_path, final_box_paths, actual_actions
        else:
            print("No solution found by Z3.")
            return None, None, None

    def solve(self, method="state_space", max_steps=50):
        """Main solving entry point."""
        # (Identical to previous version)
        start_time = time.time()
        robot_path, box_paths, actions = None, None, None
        if method == "state_space":
            robot_path, box_paths, actions = self.solve_with_state_space_search(max_states=max_steps)
        elif method == "z3":
            robot_path, box_paths, actions = self.solve_with_z3(max_steps=max_steps)
        else:
            raise ValueError(f"Unknown solving method: {method}")
        end_time = time.time()
        print(f"Total solving time: {end_time - start_time:.2f} seconds")

        if robot_path and actions is not None:
             # Basic validation: Path lengths should match
            if len(robot_path) == len(actions) + 1 and all(len(bp) == len(actions) + 1 for bp in box_paths):
                print("-" * 20 + "\nSolution Summary:\n" + "-" * 20)
                print(f" Method: {method}")
                print(f" Length: {len(actions)} steps")
                print(f" Actions: {[dir_names[a] for a in actions]}")
                print(f" Final Robot Pos: {robot_path[-1]} (Goal: {self.goal_pos})")
                final_box_poses = [bp[-1] for bp in box_paths]
                print(f" Final Box Pos: {final_box_poses}")
                print("-" * 20)
                self.animate_solution(robot_path, box_paths, filename=f"scene_25{method}.gif")
                return True
            else:
                print("Error: Path lengths mismatch after solving.")
                print(f" Robot path len: {len(robot_path)}")
                print(f" Box paths lens: {[len(bp) for bp in box_paths]}")
                print(f" Actions len: {len(actions)}")
                return False
        else:
             print("No solution found or path extraction failed.")
             return False


    def animate_solution(self, robot_path, box_paths, filename="solution.gif"):
        """Create an animation of the solution."""
        # (Identical to previous version)
        if not robot_path or not box_paths or not box_paths[0]:
            print("No valid path found to animate.")
            return False

        fig, ax = plt.subplots(figsize=(8, 8)) # Slightly smaller default size
        box_colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow'] # More colors

        max_frames = len(robot_path)
        num_boxes_in_path = len(box_paths)

        def draw_frame(t):
            ax.clear()
            ax.set_xticks(np.arange(-0.5, self.width -0.5, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, self.height-0.5, 1), minor=True)
            ax.grid(which='minor', color='grey', linestyle='-', linewidth=0.5)
            ax.tick_params(which='minor', size=0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(-0.5, self.width - 0.5)
            ax.set_ylim(self.height - 0.5, -0.5) # Inverted Y for matrix display
            ax.set_aspect('equal')
            ax.set_title(f"Step {t}/{max_frames - 1}")

            # Draw grid cells (walls and goal)
            for r in range(self.height):
                for c in range(self.width):
                    if self.grid[r, c] == 1: # Wall
                        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='black', zorder=1))
                    elif (r, c) == self.goal_pos: # Goal
                        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='lime', alpha=0.4, zorder=0))

            # Draw boxes
            for i in range(num_boxes_in_path):
                box_path = box_paths[i]
                if t < len(box_path):
                    br, bc = box_path[t]
                    color = box_colors[i % len(box_colors)]
                    ax.add_patch(plt.Rectangle((bc - 0.4, br - 0.4), 0.8, 0.8, color=color, label=f'Box {i+1}' if t==0 else "", zorder=2))

            # Draw robot
            if t < len(robot_path):
                rr, rc = robot_path[t]
                ax.add_patch(plt.Circle((rc, rr), 0.35, color='red', label='Robot' if t==0 else "", zorder=3))

            # Legend
            if t == 0 and (self.num_boxes > 0 or True): # Show robot label even if no boxes
                 handles, labels = ax.get_legend_handles_labels()
                 if handles: # Only show legend if there's something to label
                      ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)


        ani = animation.FuncAnimation(fig, draw_frame, frames=max_frames, interval=400, repeat=True, repeat_delay=1500)
        try:
            if not filename.lower().endswith(".gif"): filename += ".gif"
            ani.save("scene_25.gif", writer='pillow', dpi=90)
            print(f"✅ Animation saved as 'scene_25.gif'")
        except Exception as e:
            print(f"❌ Failed to save animation: {e}\n  (Ensure 'pillow' is installed: pip install pillow)")
        plt.close(fig)
        return True

# (create_complex_puzzle function remains the same)
def create_complex_puzzle():
    """Create a more complex puzzle for testing."""
    grid = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 2, 0, 0, 1, 4, 0, 0, 1],
        [1, 0, 0, 0, 4, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 4, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 3, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ])
    return grid


if __name__ == "__main__":
    # --- Configuration ---
    use_complex = False
    # solve_method = "state_space" # Faster for simpler puzzles, finds shortest path
    solve_method = "z3"         # Can handle more complex states/longer paths, but slower setup

    # Parameters (adjust based on method and puzzle complexity)
    state_limit = 20000 # Max states for BFS (state_space method)
    plan_length = 30    # Max steps for Z3 plan (z3 method)
    # --------------------

    print(f"Using {'Complex' if use_complex else 'Default'} Puzzle")
    print(f"Solver Method: {solve_method}")

    grid_to_use = create_complex_puzzle() if use_complex else None
    solver = IcePuzzleSolver(grid=grid_to_use)

    if solve_method == "state_space":
        print(f"State Space Search Limit: {state_limit} states")
        solver.solve(method="state_space", max_steps=state_limit)
    elif solve_method == "z3":
        print(f"Z3 Max Plan Length: {plan_length} steps")
        solver.solve(method="z3", max_steps=plan_length)

    print("\nScript finished.")