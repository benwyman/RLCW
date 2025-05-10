from collections import defaultdict
from grid_utils import unmark_block
from visualization import visualize_grid
import random
import math

LEDGE_TILES = {'_', '⤓', '↥', '⬒', '☆'}
VALID_MOVE_TILES = {'O', '_', '\\', '/', '⤓', '↥', '⬒', '█', '^', 'Φ', '☆'}
STEP_PENALTY = 0.005
MAX_STEPS = 15000

def choose_action(state, q_table, epsilon, width, grid, policy="epsilon", temperature=1.0):
    # available actions based on the state type
    if isinstance(state[0], str) and state[0] == 'block':
        available_actions = list(range(width))
    elif isinstance(state[0], tuple):  # ledge state
        ledge_start_x, ledge_y = state[0]
        available_actions = [col for col in range(width) if grid.get((col, ledge_y)) in {'_', '⬒', '☆', '↥', '⤓'}]
    else:
        print(f"Warning: Unrecognized state format for action selection: {state}")
        available_actions = list(range(width))
        
    # initialize Q-values if not present
    if state not in q_table:
        q_table[state] = defaultdict(float)
    current_q_actions = q_table[state]
    for act in available_actions:
        if act not in current_q_actions:
            current_q_actions[act] = 0.0

    # === ε-GREEDY POLICY ===
    if policy == "epsilon":
        if random.random() < epsilon:
            return random.choice(available_actions)
        else:
            max_q = max(current_q_actions[a] for a in available_actions)
            best_actions = [a for a in available_actions if current_q_actions[a] == max_q]
            return random.choice(best_actions)

    # === SOFTMAX POLICY ===
    elif policy == "softmax":
        logits = [current_q_actions[a] / temperature for a in available_actions]
        max_logit = max(logits)
        exps = [math.exp(l - max_logit) for l in logits]  # numerically stable softmax
        total = sum(exps)
        probs = [e / total for e in exps]
        return random.choices(available_actions, weights=probs, k=1)[0]

    else:
        raise ValueError(f"Unknown policy type: {policy}")

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
    if is_block_row(grid, x, y):
        return (('block', y), frozenset(pressed_buttons))
    return None

def handle_blocks(grid, x, y, width, exploration_rate, tracker_dict, q_table, pressed_buttons, episode, reward, state_action_pairs=None, step_counter=None, policy="epsilon", temperature=1.0):
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
        action = choose_action(state, q_table, exploration_rate, width, grid, policy=policy, temperature=temperature)
        

        if state_action_pairs is not None:
            state_action_pairs.append((state, action))

        if step_counter is not None:
            step_counter[0] += 1  # increment shared step count

        x = action
        reward -= STEP_PENALTY # step penalty
        if (x, y) in tracker_dict["pipes"]:
            destination = tracker_dict["pipes"][(x, y)]
            tracker_dict["pipe_tracker"][(x, y)] += 1
            return destination[0], destination[1], reward

        return x, y, reward

def drop_ball(
    grid, width, height, start_x, buckets, target_bucket,
    exploration_rate,
    q_table,
    trackers,
    episode,
    visualize=False,
    policy="epsilon",
    temperature=1.0,
):
    """
    Unified drop function for both Q-learning and DQN.

    Args:
        grid, width, height, start_x, buckets, target_bucket: shared environment info
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
        (state_action_pairs, reward)
    """
    x, y = start_x, height - 1
    pressed_buttons = set()
    stars_collected = set()

    # Fake 10 stars if none exist on the board (e.g. default map)
    if all(grid.get(pos) != '☆' for pos in grid):
        stars_collected.update({("fake", i) for i in range(10)})
    state_action_pairs = []

    done = False
    reward = 0
    step_counter = [0]

    while y > 0:
        reward -= STEP_PENALTY # step penalty

        if visualize:
            visualize_grid(grid, width, height, ball_position=(x, y), buckets=buckets)

        tile = grid.get((x, y), ' ')
        is_ledge = tile in LEDGE_TILES
        is_block = is_block_row(grid, x, y)

        if is_ledge or is_block:
            state = identify_decision_state(x, y, grid, pressed_buttons)

            if isinstance(state[0], tuple) and state[0][0] == "block":
                x, y, reward = handle_blocks(
                    grid, x, y, width, exploration_rate,
                    {
                        "block_row_tracker": trackers["block_row_tracker"],
                        "pipe_tracker": trackers["pipe_tracker"],
                        "pipes": trackers["pipes"]
                    },
                    q_table, pressed_buttons, episode, reward,
                    state_action_pairs,
                    step_counter=step_counter,
                    policy=policy,
                    temperature=temperature
                )
                continue

            # log visit
            trackers["ledge_tracker"][state] += 1

            # Q-learning: record state
            action = choose_action(state, q_table, exploration_rate, width, grid, policy=policy, temperature=temperature)
            state_action_pairs.append((state, action))
            step_counter[0] += 1  # increment shared step count
            if step_counter[0] >= MAX_STEPS:
                return (state_action_pairs, reward, stars_collected, None, step_counter[0])

            # if the agent lands on a button tile
            if grid.get((action, y)) == '⬒':
                num_buttons_pressed = sum(1 for pos in pressed_buttons if pos in trackers["button_tracker"])
                if len(stars_collected) >= num_buttons_pressed + 1:
                    grid[(action, y)] = '_'
                    row_y = trackers.get("button_to_block_map", {}).get((action, y))
                    if row_y is not None:
                        reward += 1
                        unmark_block(grid, row_y, trackers["blocks"])
                    trackers["button_tracker"][(action, y)] += 1
                    pressed_buttons.add((action, y))
                    x = action
                    trackers["ledge_tracker"][state] -= 1
                continue  # re-evaluate tile after button logic (don't fall yet)
            
            # Bonus Star
            if grid.get((action, y)) == '☆':
                trackers["bonus_star_tracker"][(action, y)] += 1
                grid[(action, y)] = '_'  # convert to normal ledge
                pressed_buttons.add((action, y))  # optional for state tracking
                stars_collected.add((action, y))  # track that it was picked up
                reward += 1
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
            reward -= 10
            done = True
            return (state_action_pairs, reward, stars_collected, None, step_counter[0])

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
           reward += 10
        else:
            reward -= 10

    return (state_action_pairs, reward, stars_collected, bucket, step_counter[0])

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
