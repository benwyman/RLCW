# === Imports ===
import random
from collections import defaultdict, deque, namedtuple
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from state_utils_deep import (
    drop_ball, initialize_trackers, 
    PrioritizedReplayBuffer, preprocess_state
)
from board_builder import build_board
from visualization import visualize_grid

# === Device Configuration ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Q-Network Definition ===
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        # input layer to hidden
        self.fc1 = nn.Linear(input_dim, 128)
        # hidden to hidden
        self.fc2 = nn.Linear(128, 128)
        # hidden to output (Q-value for each action)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        # pass through network with ReLU activations
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))

# === Training Hyperparameters ===
learning_rate = 0.001               # optimizer learning rate
discount_factor = 0.99              # gamma: importance of future rewards
exploration_rate = 1.0              # epsilon: initial exploration probability
exploration_decay = 0.995           # decay per episode
min_exploration = 0.01              # lower bound for epsilon
episodes = 1000                     # total training episodes
initial_free_exploration = 0        # episode cutoff for exploration decay

# === DQN-specific Parameters ===
update_frequency = 4                # learn every N decisions
target_update_frequency = 100       # soft update every N decisions
soft_update_alpha = 0.1             # tau: target network smoothing factor

# === Prioritized Experience Replay Setup ===
Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])
replay_buffer = PrioritizedReplayBuffer(capacity=10000)  # custom buffer with priorities
batch_size = 64                    # number of experiences to sample

# === Training Objectives ===
target_bucket = 2                  # desired goal bucket

# === Initialize Stats and Environment ===
trackers = initialize_trackers()   # global stat trackers for board elements
episode_rewards_history = []       # stores total reward for each episode
most_recent_rewards = deque(maxlen=100)  # sliding window for recent episode rewards
total_stars_collected = 0          # tracks cumulative star count
total_decision_steps = [0]         # decision step count (mutable list)
episode_losses = []                # average loss per episode (for plotting)

# === Initialize Game Board ===
grid, buckets, width, height = build_board("hard", trackers)

# === Define Neural Network Dimensions ===
input_dim = 3 + (width * 5)         # [type, x, y] + flattened button vector
output_dim = width                  # one Q-value per column action

# === Instantiate Online and Target Networks ===
online_net = QNetwork(input_dim, output_dim).to(device)
target_net = QNetwork(input_dim, output_dim).to(device)
target_net.load_state_dict(online_net.state_dict())  # sync initially
target_net.eval()                  # put target net in inference mode

# === Optimizer ===
optimizer = optim.Adam(online_net.parameters(), lr=learning_rate)

# === Learning Step Function ===
def learn(grid, width, learning_rate, discount_factor, episode):
    # skip if not enough experiences yet
    if len(replay_buffer) < batch_size:
        return

    # compute beta (importance sampling adjustment) for this episode
    beta = min(1.0, (episode / episodes) ** 0.6)

    # sample batch from buffer
    batch, indices, weights = replay_buffer.sample(batch_size, beta)
    weights = weights.unsqueeze(1).to(device)

    # separate experience components
    state_batch = torch.cat([preprocess_state(s, width) for s, _, _, _, _ in batch]).to(device)
    action_batch = torch.tensor([a for _, a, _, _, _ in batch], dtype=torch.int64).unsqueeze(1).to(device)
    reward_batch = torch.tensor([r for _, _, r, _, _ in batch], dtype=torch.float32).unsqueeze(1).to(device)
    done_batch = torch.tensor([d for _, _, _, _, d in batch], dtype=torch.float32).unsqueeze(1).to(device)

    # preprocess next states and mask out terminal transitions
    next_states = [s for _, _, _, s, _ in batch]
    non_final_mask = torch.tensor([s is not None for s in next_states], dtype=torch.bool)
    non_final_next_states = torch.cat([preprocess_state(s, width) for s in next_states if s is not None]).to(device)

    # compute current Q-values
    q_values = online_net(state_batch).gather(1, action_batch)

    # compute target Q-values
    target_q_values = torch.zeros(batch_size, 1, device=device)
    if non_final_next_states.size(0) > 0:
        next_actions = online_net(non_final_next_states).argmax(dim=1).unsqueeze(1)
        target_q = target_net(non_final_next_states).gather(1, next_actions)
        target_q_values[non_final_mask] = target_q.detach()

    # Bellman target
    expected_q = reward_batch + discount_factor * target_q_values * (1 - done_batch)

    # TD error and loss
    td_errors = q_values - expected_q
    loss = (td_errors.pow(2) * weights).mean()

    # optimize
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=5.0)
    optimizer.step()

    # update priorities
    replay_buffer.update_priorities(indices, td_errors.squeeze().detach())

    return loss.item()

# === Soft Target Network Update ===
def update_target_network(online_model, target_model, alpha):
    for target_param, online_param in zip(target_model.parameters(), online_model.parameters()):
        target_param.data.copy_(alpha * online_param.data + (1 - alpha) * target_param.data)

# === Learning Condition Function ===
def should_learn():
    return total_decision_steps[0] % update_frequency == 0

grid, buckets, width, height = build_board("hard", trackers)
start_x = random.randint(0, width - 1)
visualize_grid(grid, width, height, ball_position=(start_x, height - 1), buckets=buckets)

# === Training Loop ===
for episode in range(episodes):
    # regenerate a fresh board per episode
    grid, buckets, width, height = build_board("hard", trackers)
    start_x = random.randint(0, width - 1)

    # track how many steps were taken this episode
    steps_before = total_decision_steps[0]

    print(f"=== [Episode {episode + 1}] Starting episode logic...", flush=True)

    # drop the ball and get result of episode
    episode_final_reward, final_bucket, stars_collected = drop_ball(
        grid=grid,
        width=width,
        height=height,
        start_x=start_x,
        buckets=buckets,
        target_bucket=target_bucket,
        exploration_rate=exploration_rate,
        q_model=online_net,
        trackers=trackers,
        extra={
            "replay_buffer": replay_buffer,
            "should_learn": should_learn,
            "learn": learn,
            "Experience": Experience,
            "target_net": target_net,
            "soft_update_alpha": soft_update_alpha,
            "update_target_network": update_target_network,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "discount_factor": discount_factor,
            "total_decision_steps": total_decision_steps,
            "target_update_frequency": target_update_frequency,
            "episode": episode,
        },
        visualize=False  # set True to debug
    )

    print(f"=== [Episode {episode + 1}] Finished episode logic.", flush=True)

    # post-episode stats
    steps_taken = total_decision_steps[0] - steps_before
    total_stars_collected += len(stars_collected)
    episode_rewards_history.append(episode_final_reward)
    most_recent_rewards.append(episode_final_reward)

    # perform batch learning steps after episode
    losses = []
    if len(replay_buffer) > batch_size:
        for _ in range(2):  # optional: increase for more updates
            print("      >> Learning...", flush=True)
            loss = learn(grid, width, learning_rate, discount_factor, episode)
            print(f"      << Finished learning. Loss: {loss}", flush=True)
            if loss is not None:
                losses.append(loss)

    # track average loss this episode
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    episode_losses.append(avg_loss)

    # decay exploration
    if episode >= initial_free_exploration:
        exploration_rate = max(min_exploration, exploration_rate * exploration_decay)

    # episode summary
    print(f"[Episode {episode+1}] Reward: {episode_final_reward} | Bucket: {final_bucket} | Stars: {len(stars_collected)} | Steps: {steps_taken} | ε: {exploration_rate:.2f}", flush=True)

    # periodic performance print
    if (episode + 1) % 100 == 0:
        avg_reward = sum(most_recent_rewards) / len(most_recent_rewards)
        avg_stars = total_stars_collected / (episode + 1)
        if losses:
            avg_loss = sum(losses) / len(losses)
            print(f"Ep {episode + 1} | Avg R (last 100): {avg_reward:.2f} | Stars: {avg_stars:.2f} | Loss: {avg_loss:.4f} | ε: {exploration_rate:.2f}", flush=True)
        else:
            print(f"Ep {episode + 1} | Avg R (last 100): {avg_reward:.2f} | Stars: {avg_stars:.2f} | ε: {exploration_rate:.2f}", flush=True)

import csv

with open("training_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Episode", "Reward", "AvgLoss"])
    for i, (reward, loss) in enumerate(zip(episode_rewards_history, episode_losses)):
        writer.writerow([i + 1, reward, loss])

import matplotlib.pyplot as plt

# Load from file or just use the in-memory lists
plt.plot(episode_rewards_history, label="Reward")
plt.plot(episode_losses, label="Avg Loss")
plt.xlabel("Episode")
plt.ylabel("Value")
plt.title("Training Progress")
plt.legend()
plt.grid(True)
plt.show()
