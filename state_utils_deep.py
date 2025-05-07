from collections import defaultdict, deque
from grid_utils import unmark_block
from visualization import visualize_grid
import random
import heapq
import torch
from torch import nn

# === Constants and Globals ===
LEDGE_TILES = {'_', '⤓', '↥', '⬒', '☆'}
VALID_MOVE_TILES = {'O', '_', '\\', '/', '⤓', '↥', '⬒', '█', '^', 'Φ'}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
step_penalty = 0.01

# === Action Selection ===
def choose_action(state, model, epsilon, width, grid):
    # determine available actions
    if isinstance(state[0], str) and state[0] == 'block':
        available_actions = list(range(width))
    elif isinstance(state[0], tuple):  # ledge state
        ledge_start_x, ledge_y = state[0]
        available_actions = [col for col in range(width) if grid.get((col, ledge_y)) in LEDGE_TILES]
    else:
        print(f"Warning: Unrecognized state format for action selection: {state}")
        available_actions = list(range(width))

    if random.random() < epsilon:
        return random.choice(available_actions)

    with torch.no_grad():
        state_tensor = preprocess_state(state, width)
        q_values = model(state_tensor.to(device)).squeeze(0)

    # select among available actions
    best_value = -float("inf")
    best_actions = []
    for a in available_actions:
        val = q_values[a].item()
        if val > best_value:
            best_value = val
            best_actions = [a]
        elif val == best_value:
            best_actions.append(a)

    return random.choice(best_actions)

# === State Identification ===

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

# === Block Row Logic ===

def handle_blocks(grid, x, y, width, exploration_rate, tracker_dict, q_model, pressed_buttons, extra):
    """
    Handles movement and decision logic when the agent is on a block row.

    Parameters:
    - choose_action_func: function used to select action
    - tracker_dict: must contain "block_row_tracker" and "pipe_tracker"
    - q_model: Q-table (can be online or target in DQN)
    """
    state = (('block', y), frozenset(pressed_buttons))
    tracker_dict['block_row_tracker'][state] += 1

    while True:
        action = choose_action(state, q_model, exploration_rate, width, grid)

        x = action
        reward = -step_penalty
        log_transition(state, action, reward, state, extra)
        maybe_learn(extra, q_model, width, grid)

        if (x, y) in tracker_dict["pipes"]:
            destination = tracker_dict["pipes"][(x, y)]
            tracker_dict["pipe_tracker"][(x, y)] += 1
            return destination[0], destination[1]
        
        return x, y

# === State Encoding for Neural Network ===
def encode_state(state, width):
    button_vector = torch.zeros(width * 5)
    for bx, by in state[1]:
        index = (by * width + bx) % (width * 5)
        button_vector[index] = 1

    key = state[0]

    if isinstance(key, tuple) and isinstance(key[0], int):
        x, y = key
        return torch.tensor([0, x, y], dtype=torch.float32).unsqueeze(0), button_vector.unsqueeze(0)

    elif isinstance(key, tuple) and key[0] == 'block':
        y = key[1]
        return torch.tensor([1, 0, y], dtype=torch.float32).unsqueeze(0), button_vector.unsqueeze(0)

    else:
        raise ValueError("Unrecognized state format.")

def preprocess_state(state, width):
    base, buttons = encode_state(state, width)
    return torch.cat([base, buttons], dim=1)

# === Experience and Learning Helpers ===

def log_transition(prev_state, prev_action, reward, next_state, extra, done=False, boost=False):
    if prev_state is None:
        return
    exp = extra["Experience"](prev_state, prev_action, reward, next_state, done)
    if boost:
        extra["replay_buffer"].add(exp, td_error=20.0)  # star/spike reward
    else:
        extra["replay_buffer"].add(exp)

def maybe_learn(extra, q_model, width, grid):
    if extra["should_learn"]() and len(extra["replay_buffer"]) >= extra["batch_size"]:
        extra["learn"](grid, width, extra["learning_rate"], extra["discount_factor"], extra["episode"])
        if extra["total_decision_steps"][0] % extra["target_update_frequency"] == 0:
            extra["update_target_network"](q_model, extra["target_net"], extra["soft_update_alpha"])

# === Experience and Learning Helpers ===

def drop_ball(
    grid, width, height, start_x, buckets, target_bucket,
    exploration_rate,
    q_model,
    trackers,
    extra=None,
    visualize=False
):
    """

    Args:
        grid, width, height, start_x, buckets, target_bucket: shared environment info
        exploration_rate: epsilon-greedy parameter
        q_model: Q-table (dict for Q, neural net or dict for DQN)
        trackers: dict with keys: blocks, pipes, ledge_tracker, pipe_tracker, button_tracker, spike_tracker, bucket_tracker
        extra: DQN-specific dict with keys (optional):
            - replay_buffer
            - should_learn
            - learn
            - Experience
            - target_net
            - soft_update_alpha
            - update_target_network
            - batch_size
            - total_decision_steps (passed as reference)
    """
    x, y = start_x, height - 1
    pressed_buttons = set()
    stars_collected = set()

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
                }, q_model, pressed_buttons, extra)
                continue

            # log visit
            trackers["ledge_tracker"][state] += 1

            # DQN logic
            action = choose_action(state, q_model, exploration_rate, width, grid)
            if last_state is not None:
                reward -= step_penalty  # step penalty
                log_transition(last_state, last_action, reward, state, extra)
                maybe_learn(extra, q_model, width, grid)
            last_state = state
            last_action = action
            extra["total_decision_steps"][0] += 1

            if grid.get((action, y)) == '⬒':
                num_buttons_pressed = sum(1 for pos in pressed_buttons if pos in trackers["button_tracker"])
                if len(stars_collected) >= num_buttons_pressed + 1:
                    grid[(action, y)] = '_'
                    row_y = trackers.get("button_to_block_map", {}).get((action, y))
                    if row_y is not None:
                        unmark_block(grid, row_y, trackers["blocks"])
                        # visualize_grid(grid, width, height, ball_position=(x, y), buckets=buckets)
                    trackers["button_tracker"][(action, y)] += 1
                    pressed_buttons.add((action, y))
                    x = action
                    trackers["ledge_tracker"][state] -= 1
                continue
            
            # Bonus Star
            if grid.get((action, y)) == '☆':
                trackers["bonus_star_tracker"][(action, y)] += 1
                grid[(action, y)] = '_'
                pressed_buttons.add((action, y))
                stars_collected.add((action, y))
                # visualize_grid(grid, width, height, ball_position=(x, y), buckets=buckets)

                # log star reward as its own transition
                reward += 10
                log_transition(last_state, last_action, reward, state, extra, boost=True)

                # stay on same row
                trackers["ledge_tracker"][state] -= 1
                continue

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
            print(f"⚠️ Spike hit at ({x}, {y}) — applying penalty {reward}", flush=True)
            log_transition(last_state, last_action, reward, None, extra, done=True, boost=True)
            maybe_learn(extra, q_model, width, grid)
            return (reward, -1, stars_collected)

        # Diagonal fall
        moves = get_valid_diagonal_moves(grid, x, y, width, height)
        if moves:
            x, y = random.choice(moves)
        else:
            y -= 1

    # Episode end
    bucket = buckets.get(x, -1)
    if bucket != -1:
        reward += 0
        trackers["bucket_tracker"][bucket] += 1
        if bucket == target_bucket:
           reward += 0

    done = True
    log_transition(last_state, last_action, reward, None, extra, done=True)
    maybe_learn(extra, q_model, width, grid)

    return (reward, bucket, stars_collected)

# === Tracker Initialization ===

def initialize_trackers():
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

# === Prioritized Experience Replay Buffer ===
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.8):
        self.capacity = capacity                      # max number of experiences to store
        self.buffer = []                              # list to store experiences
        self.priorities = []                          # list of priorities (same order as buffer)
        self.next_index = 0                           # circular index for overwriting old data
        self.alpha = alpha                            # how much prioritization is used (0 = uniform, 1 = full prioritization)

    def __len__(self):
        return len(self.buffer)                       # return current number of experiences

    def add(self, experience, td_error=1.0):
        # compute priority using TD error and alpha
        priority = (abs(td_error) + 1e-5) ** self.alpha

        if len(self.buffer) < self.capacity:
            # add new experience and its priority if buffer isn't full
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            # overwrite oldest experience and priority (circular buffer)
            self.buffer[self.next_index] = experience
            self.priorities[self.next_index] = priority
            self.next_index = (self.next_index + 1) % self.capacity  # wrap around

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return []

        # convert priorities to probabilities
        priorities = torch.tensor(self.priorities, dtype=torch.float32)
        probs = priorities / priorities.sum()

        # sample indices based on priority probabilities
        indices = torch.multinomial(probs, batch_size, replacement=True)

        # compute importance-sampling weights to correct for non-uniform sampling
        total = len(self.buffer)
        weights = (total * probs[indices]).pow(-beta)  # beta controls bias correction
        weights /= weights.max()  # normalize for stability

        # gather samples from buffer using selected indices
        samples = [self.buffer[i] for i in indices]

        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        # update priorities for sampled transitions using new TD errors
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error.item()) ** self.alpha

