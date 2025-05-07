from collections import defaultdict
from grid_utils import unmark_block
from visualization import visualize_grid
import random

LEDGE_TILES = {'_', '⤓', '↥', '⬒', '☆'}
VALID_MOVE_TILES = {'O', '_', '\\', '/', '⤓', '↥', '⬒', '█', '^', 'Φ'}

def choose_action(state, q_table, epsilon, width, grid):
    # available actions based on the state type
    if isinstance(state[0], str) and state[0] == 'block':
        available_actions = list(range(width))
    elif isinstance(state[0], tuple): # ledge state
        ledge_start_x, ledge_y = state[0]
        # check if ledge_y is valid before accessing grid
        available_actions = [col for col in range(width) if grid.get((col, ledge_y)) in LEDGE_TILES]
    else:
        print(f"Warning: Unrecognized state format for action selection: {state}")
        available_actions = list(range(width))
        
    # check state exists in the q_table (initialize if not)
    if state not in q_table:
        q_table[state] = defaultdict(float)
        for act in available_actions:
             q_table[state][act] = 0.0 

    current_q_actions = q_table[state]

    for act in available_actions:
        if act not in current_q_actions:
             current_q_actions[act] = 0.0

    if random.random() < epsilon:
        return random.choice(available_actions)  # explore
    else:
        # choose the action with the highest Q-value from available actions
        q_values = q_table[state]
        max_q = -float('inf')
        best_actions = []
        
        for act in available_actions: 
            q_val = q_values.get(act, 0.0) # set to 0 if action not seen before in this state
            if q_val > max_q:
                max_q = q_val
                best_actions = [act]
            elif q_val == max_q:
                best_actions.append(act)
        
        if not best_actions: # if all available actions have Q=-inf
             return random.choice(available_actions) # select random available action
             
        return random.choice(best_actions) # choose randomly among best actions

# find the state key ((start_x, y), frozenset(buttons)) for the ledge the ball is currently on
def find_ledge_state_key(x, y, grid, pressed_buttons):
    # search left from current x to find the start of the connected ledge segment
    ledge_start_x = x
    while ledge_start_x >= 0 and grid.get((ledge_start_x, y)) in LEDGE_TILES:
        ledge_start_x -= 1
    ledge_start_x += 1 # correct start position
    
    # check if the found start is actually a ledge tile
    if grid.get((ledge_start_x, y)) in LEDGE_TILES:
         return ((ledge_start_x, y), frozenset(pressed_buttons))
    else:
         # this is when the ball is on a pipe tile adjacent to a ledge and not technically on the ledge start itself
         # return None if we can't confirm the start point
         print(f"Debug: Could not confirm ledge start for ({x},{y}). Found start_x={ledge_start_x}, tile={grid.get((ledge_start_x, y))}")
         return None 
    
def is_block_row(grid, x, y):
    return grid.get((x, y)) == '█' or (
        grid.get((x, y)) in {'⤓', '↥'} and (
            grid.get((x - 1, y), '') == '█' or grid.get((x + 1, y), '') == '█'
        )
    )

def get_valid_diagonal_moves(grid, x, y, width, height):
    """
    Returns a list of valid diagonal positions to move to from (x, y),
    checking if the tile is within the board and is in VALID_MOVE_TILES.
    """
    moves = []
    for dx in [-1, 1]:  # check both left and right diagonals
        nx, ny = x + dx, y - 1
        if 0 <= nx < width and 0 <= ny < height and grid.get((nx, ny)) in VALID_MOVE_TILES:
            moves.append((nx, ny))
    return moves

def identify_decision_state(x, y, grid, pressed_buttons):
    tile = grid.get((x, y), '')
    if tile in LEDGE_TILES:
        return find_ledge_state_key(x, y, grid, frozenset(pressed_buttons))
    if tile == '█' or (tile in {'⤓', '↥'} and (grid.get((x - 1, y)) == '█' or grid.get((x + 1, y)) == '█')):
        return (('block', y), frozenset(pressed_buttons))
    return None

def handle_blocks(grid, x, y, width, exploration_rate, tracker_dict, q_table, pressed_buttons, state_action_pairs=None):
    """
    Handles movement and decision logic when the agent is on a block row.

    Parameters:
    - choose_action_func: function used to select action
    - tracker_dict: must contain "block_row_tracker" and "pipe_tracker"
    - q_table: Q-table (can be online or target in DQN)
    - state_action_pairs: optional list for v1 to record transitions
    """
    state = (('block', y), frozenset(pressed_buttons))
    tracker_dict['block_row_tracker'][state] += 1

    while True:
        action = choose_action(state, q_table, exploration_rate, width, grid)

        if state_action_pairs is not None:  # v1 logs states
            state_action_pairs.append((state, action))

        x = action
        if (x, y) in tracker_dict["pipes"]:
            destination = tracker_dict["pipes"][(x, y)]
            tracker_dict["pipe_tracker"][(x, y)] += 1
            return destination[0], destination[1]

        return x, y

def drop_ball(
    grid, width, height, start_x, buckets, target_bucket,
    mode,  # "q" or "dqn"
    exploration_rate,
    q_table,
    trackers,
    # optional extras for DQN
    extra=None,
    visualize=False
):
    """
    Unified drop function for both Q-learning and DQN.

    Args:
        grid, width, height, start_x, buckets, target_bucket: shared environment info
        mode: "q" or "dqn"
        exploration_rate: epsilon-greedy parameter
        q_table: Q-table (dict for Q, neural net or dict for DQN)
        trackers: dict with keys: blocks, pipes, ledge_tracker, pipe_tracker, button_tracker, spike_tracker, bucket_tracker
        extra: DQN-specific dict with keys (optional):
            - replay_buffer
            - should_learn
            - learn
            - Experience
            - q_table_target
            - soft_update_alpha
            - update_target_network
            - batch_size
            - total_decision_steps (passed as reference)

    Returns:
        If mode == "q": (state_action_pairs, reward)
        If mode == "dqn": (reward, final_bucket)
    """
    x, y = start_x, height - 1
    pressed_buttons = set()
    stars_collected = set()

    if mode == "q":
        state_action_pairs = []

    last_state, last_action = None, None
    done = False
    reward = 0

    while y > 0:
        if visualize:
            visualize_grid(grid, width, height, ball_position=(x, y), buckets=buckets)

        tile = grid.get((x, y), ' ')
        is_ledge = tile in LEDGE_TILES
        is_block = tile == '█' or (tile in {'⤓', '↥'} and (grid.get((x - 1, y)) == '█' or grid.get((x + 1, y)) == '█'))

        if is_ledge or is_block:
            state = identify_decision_state(x, y, grid, pressed_buttons)

            if isinstance(state[0], tuple) and state[0][0] == "block":
                x, y = handle_blocks(grid, x, y, width, exploration_rate, {
                    "block_row_tracker": trackers["block_row_tracker"],
                    "pipe_tracker": trackers["pipe_tracker"],
                    "pipes": trackers["pipes"]
                }, q_table, pressed_buttons,
                    state_action_pairs if mode == "q" else None
                )
                continue

            # log visit
            trackers["ledge_tracker"][state] += 1

            # Q-learning: record state
            if mode == "q":
                action = choose_action(state, q_table, exploration_rate, width, grid)
                state_action_pairs.append((state, action))
            else:
                # DQN logic
                action = choose_action(state, q_table, exploration_rate, width, grid)
                if last_state is not None:
                    extra["replay_buffer"].append(extra["Experience"](last_state, last_action, 0, state, False))
                    if extra["should_learn"]() and len(extra["replay_buffer"]) >= extra["batch_size"]:
                        extra["learn"](grid, width, extra["learning_rate"], extra["discount_factor"])
                        if extra["total_decision_steps"][0] % extra["target_update_frequency"] == 0:
                            extra["update_target_network"](q_table, extra["q_table_target"], extra["soft_update_alpha"])
                last_state = state
                last_action = action
                extra["total_decision_steps"][0] += 1

            # if the agent lands on a button tile
            if grid.get((action, y)) == '⬒':
                grid[(action, y)] = '_'  # press the button by converting it to a normal ledge

                button_to_block = trackers.get("button_to_block_map", {})  # get map of button positions to block row Y values
                row_y = button_to_block.get((action, y))  # find which block row this button should unmark
                if row_y is not None:
                    unmark_block(grid, row_y, trackers["blocks"])  # restore the original tiles in that block row

                trackers["button_tracker"][(action, y)] += 1  # increment press count for this button
                pressed_buttons.add((action, y))  # record that this button was pressed in the current episode
                x = action  # move the ball to the button's column
                trackers["ledge_tracker"][state] -= 1  # undo visit count to this ledge since ball stays on the row
                continue  # re-evaluate tile after button logic (don't fall yet)
            
            # Bonus Star
            if grid.get((action, y)) == '☆':
                trackers["bonus_star_tracker"][(action, y)] += 1
                grid[(action, y)] = '_'  # convert to normal ledge
                pressed_buttons.add((action, y))  # optional for state tracking
                stars_collected.add((action, y))  # track that it was picked up
                trackers["ledge_tracker"][state] -= 1
                continue  # stay on same row

            # Pipe
            if (action, y) in trackers["pipes"]:
                trackers["pipe_tracker"][(action, y)] += 1
                x, y = trackers["pipes"][(action, y)]
                continue

            # Fall off ledge
            x = action
            while (x, y - 1) in grid and grid[(x, y - 1)] == ' ':
                y -= 1
            y -= 1
            continue

        # Slide
        if tile in {'\\', '/'}:
            x += 1 if tile == '\\' else -1
            y -= 1
            if not (0 <= x < width and 0 <= y < height):
                break
            continue

        # Spike
        if tile == '^':
            trackers["spike_tracker"][y] += 1
            reward -= 1
            done = True
            if mode == "dqn" and last_state is not None:
                extra["replay_buffer"].append(extra["Experience"](last_state, last_action, reward, None, done))
                if extra["should_learn"]() and len(extra["replay_buffer"]) >= extra["batch_size"]:
                    extra["learn"](grid, width, extra["learning_rate"], extra["discount_factor"])
                    if extra["total_decision_steps"][0] % extra["target_update_frequency"] == 0:
                        extra["update_target_network"](q_table, extra["q_table_target"], extra["soft_update_alpha"])
            return (state_action_pairs, reward, stars_collected) if mode == "q" else (reward, -1, stars_collected)

        # Diagonal fall
        moves = get_valid_diagonal_moves(grid, x, y, width, height)
        if moves:
            x, y = random.choice(moves)
        else:
            y -= 1

    # Episode end
    bucket = buckets.get(x, -1)
    if bucket != -1:
        trackers["bucket_tracker"][bucket] += 1
        if bucket == target_bucket:
            reward += 1 * (3 ** len(stars_collected))


    if mode == "dqn" and last_state is not None:
        done = True
        extra["replay_buffer"].append(extra["Experience"](last_state, last_action, reward, None, done))
        if extra["should_learn"]() and len(extra["replay_buffer"]) >= extra["batch_size"]:
            extra["learn"](grid, width, extra["learning_rate"], extra["discount_factor"])
            if extra["total_decision_steps"][0] % extra["target_update_frequency"] == 0:
                extra["update_target_network"](q_table, extra["q_table_target"], extra["soft_update_alpha"])

    return (state_action_pairs, reward, stars_collected) if mode == "q" else (reward, bucket, stars_collected)

def initialize_trackers(include_q_table=False):
    # special maps
    pipes = {}   # (x, y) -> (x, y) pipe destinations
    blocks = {}  # row_y -> {x: original tile}

    # trackers
    bucket_tracker = defaultdict(int) # maps bucket index -> number of landings (range(num_buckets))
    ledge_tracker = defaultdict(int)  # maps (start_x, y) of ledge -> number of visits
    spike_tracker = defaultdict(int)  # maps spike row y -> number of hits
    pipe_tracker = defaultdict(int)  # maps (x, y) of pipe entry/exit -> number of uses
    button_tracker = defaultdict(int)  # maps (x, y) of button tile -> number of presses
    block_row_tracker = defaultdict(int)  # maps (block_row_y, pressed_buttons) -> number of visits
    button_to_block_map = {}  # (x, y) of button -> row_y it unmarks
    bonus_star_tracker = defaultdict(int)  # (x, y) of bonus star -> times collected

    # shared dictionary for use in board building and stats
    trackers = {
        "blocks": blocks,
        "pipes": pipes,
        "bucket_tracker": bucket_tracker,
        "button_tracker": button_tracker,
        "pipe_tracker": pipe_tracker,
        "spike_tracker": spike_tracker,
        "ledge_tracker": ledge_tracker,
        "block_row_tracker": block_row_tracker,
        "button_to_block_map": button_to_block_map,
        "bonus_star_tracker": bonus_star_tracker,
    }

    return trackers
