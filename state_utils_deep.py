from collections import defaultdict, deque
from grid_utils import unmark_block
from visualization import visualize_grid
import random
import heapq
import torch
from torch import nn

# === Constants and Globals ===
LEDGE_TILES = {'_', '⤓', '↥', '⬒', '☆'}
VALID_MOVE_TILES = {'O', '_', '\\', '/', '⤓', '↥', '⬒', '█', '^', 'Φ', '☆'}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STEP_PENALTY = 0.005

# === Q-Network Definition ===
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        # input layer to hidden
        self.fc1 = nn.Linear(input_dim, 1024)
        # hidden to hidden
        self.fc2 = nn.Linear(1024, 512)
        # hidden to output (Q-value for each action)
        self.fc3 = nn.Linear(512, output_dim)

    def forward(self, x):
        # pass through network with ReLU activations
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))

# === Action Selection ===
def choose_action(state, model, epsilon, width, height, grid):
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
        state_tensor = preprocess_state(state, width, height)
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

def handle_blocks(grid, x, y, width, height, exploration_rate, tracker_dict, q_model, pressed_buttons, extra, reward, step_counter=None):
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
        action = choose_action(state, q_model, exploration_rate, width, height, grid)
        step_counter[0] += 1

        x = action
        log_transition(state, action, reward, state, extra)
        reward -= STEP_PENALTY # step penalty

        if (x, y) in tracker_dict["pipes"]:
            destination = tracker_dict["pipes"][(x, y)]
            tracker_dict["pipe_tracker"][(x, y)] += 1
            return destination[0], destination[1]
        
        return x, y, reward

# === State Encoding for Neural Network ===
def preprocess_state(state, width, height):
    button_vector = torch.zeros(width * height)
    for bx, by in state[1]:
        index = by * width + bx
        button_vector[index] = 1

    key = state[0]

    if isinstance(key, tuple) and isinstance(key[0], int):
        x, y = key
        base = torch.tensor([0, x, y], dtype=torch.float32).unsqueeze(0)
    elif isinstance(key, tuple) and key[0] == 'block':
        y = key[1]
        base = torch.tensor([1, 0, y], dtype=torch.float32).unsqueeze(0)
    else:
        raise ValueError("Unrecognized state format.")

    return torch.cat([base, button_vector.unsqueeze(0)], dim=1)

# === Experience and Learning Helpers ===

def log_transition(prev_state, prev_action, reward, next_state, extra, done=False, boost=False):
    if prev_state is None:
        return
    exp = extra["Experience"](prev_state, prev_action, reward, next_state, done)
    if boost:
        extra["replay_buffer"].add(exp, td_error=20.0)  # star/spike reward
    else:
        extra["replay_buffer"].add(exp)

# === Learning Step Function ===
def learn(
    episode,                     # current episode index
    total_decision_steps,        # running count of decision steps
    replay_buffer,               # prioritized experience replay buffer
    batch_size,                  # number of experiences to sample for each update
    online_net,                  # current Q-network (being trained)
    target_net,                  # target Q-network (used for stable value targets)
    optimizer,                   # optimizer used to update online_net
    discount_factor,             # gamma: discount future rewards
    soft_update_alpha,           # tau: how slowly target_net updates toward online_net
    target_update_frequency,     # how often to update target network
    width, height,               # board dimensions (used to encode state)
    episodes                     # total number of episodes (for beta schedule)
):
    # === 1. Skip learning until buffer has enough samples ===
    if len(replay_buffer) < batch_size:
        return  # wait until enough experiences are stored before learning

    # === 2. Schedule importance-sampling correction (beta increases from beta_start to 1) ===
    beta_start = 0                # initial beta value (under-corrects for bias)
    beta_end = 1.0                # final beta value (full correction)
    beta = beta_start + (beta_end - beta_start) * (episode / episodes)  # gradually increase beta as training progresses

    # === 3. Sample prioritized batch with importance-sampling weights ===
    batch, indices, weights = replay_buffer.sample(batch_size, beta)  # sample prioritized experiences
    weights = weights.unsqueeze(1).to(device)  # reshape to [batch_size, 1] and move to device (CPU/GPU)

    # === 4. Unpack experience components ===
    state_batch = torch.cat([preprocess_state(s, width, height) for s, _, _, _, _ in batch]).to(device)  # combine state tensors
    action_batch = torch.tensor([a for _, a, _, _, _ in batch], dtype=torch.int64).unsqueeze(1).to(device)  # extract actions
    reward_batch = torch.tensor([r for _, _, r, _, _ in batch], dtype=torch.float32).unsqueeze(1).to(device)  # extract rewards
    done_batch = torch.tensor([d for _, _, _, _, d in batch], dtype=torch.float32).unsqueeze(1).to(device)  # extract done flags

    # === 5. Process next states, skipping terminal ones ===
    next_states = [s for _, _, _, s, _ in batch]  # get all next states
    non_final_mask = torch.tensor([s is not None for s in next_states], dtype=torch.bool)  # mask for non-terminal states
    non_final_next_states = torch.cat([preprocess_state(s, width, height) for s in next_states if s is not None]).to(device)  # encode non-terminal next states

    # === 6. Q-values for current actions ===
    q_values = online_net(state_batch).gather(1, action_batch)  # get Q-values for chosen actions from online network

    # === 7. Compute target Q-values using Double DQN logic ===
    target_q_values = torch.zeros(batch_size, 1, device=device)  # initialize target Q-values to zero
    if non_final_next_states.size(0) > 0:  # if there are any non-terminal next states
        next_actions = online_net(non_final_next_states).argmax(dim=1).unsqueeze(1)  # pick best next action using online net
        target_q = target_net(non_final_next_states).gather(1, next_actions)  # get target Q-values from target net
        target_q_values[non_final_mask] = target_q.detach()  # assign to valid indices (detached from computation graph)

    # === 8. Bellman update: expected Q = r + γ * Q'(s', a') ===
    expected_q = reward_batch + discount_factor * target_q_values * (1 - done_batch)  # compute Bellman target

    # === 9. Compute TD error and apply importance sampling correction ===
    td_errors = q_values - expected_q  # temporal difference error
    loss = (td_errors.pow(2) * weights).mean()  # MSE loss scaled by importance sampling weights

    # === 10. Optimize online network ===
    optimizer.zero_grad()       # clear previous gradients
    loss.backward()             # compute new gradients via backpropagation
    torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=5.0)  # prevent exploding gradients
    optimizer.step()            # apply gradients to update network weights

    # === 11. Periodically soft update target network ===
    if total_decision_steps[0] % target_update_frequency == 0:  # every N steps...
        update_target_network(online_net, target_net, soft_update_alpha)  # slowly blend weights into target net

    # === 12. Update priorities in replay buffer ===
    replay_buffer.update_priorities(indices, td_errors.squeeze().detach())  # adjust sample priorities using TD error

    return loss.item()  # return scalar loss value for logging


# === Soft Target Network Update ===
def update_target_network(online_model, target_model, alpha):
    for target_param, online_param in zip(target_model.parameters(), online_model.parameters()):
        target_param.data.copy_(alpha * online_param.data + (1 - alpha) * target_param.data)

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

    # Fake 10 stars if none exist on the board (e.g. default map)
    if all(grid.get(pos) != '☆' for pos in grid):
        stars_collected.update({("fake", i) for i in range(10)})

    last_state, last_action = None, None
    reward = 0
    
    step_counter = [0]  # wrapped in list so it's mutable

    while y > 0:
        reward -= STEP_PENALTY # step penalty
        if visualize:
            visualize_grid(grid, width, height, ball_position=(x, y), buckets=buckets)

        tile = grid.get((x, y), ' ')
        is_ledge = tile in LEDGE_TILES
        is_block = tile == '█' or (tile in {'⤓', '↥'} and (grid.get((x - 1, y)) == '█' or grid.get((x + 1, y)) == '█'))

        if is_ledge or is_block:
            state = identify_decision_state(x, y, grid, pressed_buttons)

            if isinstance(state[0], tuple) and state[0][0] == "block":
                x, y = handle_blocks(grid, x, y, width, height, exploration_rate, {
                    "block_row_tracker": trackers["block_row_tracker"],
                    "pipe_tracker": trackers["pipe_tracker"],
                    "pipes": trackers["pipes"]
                }, q_model, pressed_buttons, extra, reward, step_counter=step_counter)
                continue

            # log visit
            trackers["ledge_tracker"][state] += 1

            # DQN logic
            action = choose_action(state, q_model, exploration_rate, width, height, grid)
            if last_state is not None:
                log_transition(last_state, last_action, reward, state, extra)
                step_counter[0] += 1
            last_state = state
            last_action = action
            extra["total_decision_steps"][0] += 1
            if extra["total_decision_steps"][0] % extra["target_update_frequency"] == 0:
                extra["update_target_network"](extra["q_model"], extra["target_net"], extra["soft_update_alpha"])

            if grid.get((action, y)) == '⬒':
                num_buttons_pressed = sum(1 for pos in pressed_buttons if pos in trackers["button_tracker"])
                if len(stars_collected) >= num_buttons_pressed + 1:
                    grid[(action, y)] = '_'
                    row_y = trackers.get("button_to_block_map", {}).get((action, y))
                    if row_y is not None:
                        unmark_block(grid, row_y, trackers["blocks"])
                        reward += 1
                        log_transition(last_state, last_action, reward, state, extra, boost=True)
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
                #print(f"[Step {step_counter[0]}] Collected star at ({action}, {y})")
                # visualize_grid(grid, width, height, ball_position=(x, y), buckets=buckets)

                # log star reward as its own transition
                reward += 1
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
            reward -= 1
            # print(f"Spike hit at ({x}, {y}) — applying penalty {reward}", flush=True)
            log_transition(last_state, last_action, reward, None, extra, done=True, boost=True)
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
           reward += 100

    done = True
    log_transition(last_state, last_action, reward, None, extra, done=True)

    return (reward, bucket, stars_collected, step_counter[0])

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
    def __init__(self, capacity, alpha=0.9, epsilon=1e-5, recency_bonus=0.01):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.timestamps = []  # used to track recency
        self.next_index = 0
        self.alpha = alpha          # prioritization strength
        self.epsilon = epsilon      # small constant to ensure non-zero probabilities
        self.recency_bonus = recency_bonus  # added to favor newer entries
        self.time = 0              # global time counter for freshness

    def __len__(self):
        return len(self.buffer)

    def add(self, experience, td_error=1.0):
        # calculate base priority
        base_priority = (abs(td_error) + self.epsilon) ** self.alpha

        # apply freshness bonus
        recency_weight = 1.0 + self.recency_bonus * self.time
        priority = base_priority * recency_weight

        # insert experience and priority
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)
            self.timestamps.append(self.time)
        else:
            self.buffer[self.next_index] = experience
            self.priorities[self.next_index] = priority
            self.timestamps[self.next_index] = self.time
            self.next_index = (self.next_index + 1) % self.capacity

        self.time += 1  # increment time counter for freshness

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return [], [], []

        # convert priorities to probabilities
        priorities = torch.tensor(self.priorities, dtype=torch.float32)
        probs = priorities / priorities.sum()

        # sample indices with replacement based on priority
        indices = torch.multinomial(probs, batch_size, replacement=True)

        # compute importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]).pow(-beta)
        weights /= weights.max()  # normalize for stability

        samples = [self.buffer[i] for i in indices]
        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            base_priority = (abs(td_error.item()) + self.epsilon) ** self.alpha
            recency_weight = 1.0 + self.recency_bonus * self.timestamps[idx]
            new_priority = base_priority * recency_weight
            self.priorities[idx] = new_priority
