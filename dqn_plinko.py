# === Imports ===
import random
from collections import defaultdict, deque, namedtuple
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from state_utils_deep import (
    drop_ball, initialize_trackers, 
    PrioritizedReplayBuffer, preprocess_state, learn,
    update_target_network, QNetwork
)
from board_builder import build_board
from visualization import visualize_grid
import seaborn as sns  # for nice styling
sns.set(style="whitegrid")  # apply seaborn style

# === Device Configuration ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Training Hyperparameters ===
learning_rate = 1e-5               # optimizer learning rate
discount_factor = 0.99              # gamma: importance of future rewards
exploration_rate = 1.0              # epsilon: initial exploration probability
exploration_decay = 0.999           # decay per episode
min_exploration = 0.01              # lower bound for epsilon
episodes = 1000                     # total training episodes
initial_free_exploration = 0        # episode cutoff for exploration decay

# === DQN-specific Parameters ===
target_update_frequency = 50       # soft update every N decisions
soft_update_alpha = 0.1             # tau: target network smoothing factor

# === Prioritized Experience Replay Setup ===
Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])
replay_buffer = PrioritizedReplayBuffer(capacity=10000)  # custom buffer with priorities
batch_size = 32                    # number of experiences to sample

# === Training Objectives ===
target_bucket = 2                  # desired goal bucket
map_name = "easy"

# === Initialize Stats and Environment ===
trackers = initialize_trackers()   # global stat trackers for board elements
episode_rewards_history = []       # stores total reward for each episode
most_recent_rewards = deque(maxlen=100)  # sliding window for recent episode rewards
total_stars_collected = 0          # tracks cumulative star count
total_decision_steps = [0]         # decision step count (mutable list)
episode_losses = []                # average loss per episode (for plotting)
episode_accuracy = []  # track success (target bucket hit) per episode

# === Initialize Game Board ===
grid, buckets, width, height = build_board(map_name, trackers)

# === Define Neural Network Dimensions ===
input_dim = 3 + (width * height)         # [type, x, y] + flattened button vector
output_dim = width                  # one Q-value per column action

# === Instantiate Online and Target Networks ===
online_net = QNetwork(input_dim, output_dim).to(device)
target_net = QNetwork(input_dim, output_dim).to(device)
target_net.load_state_dict(online_net.state_dict())  # sync initially
target_net.eval()                  # put target net in inference mode

# === Optimizer ===
optimizer = optim.Adam(online_net.parameters(), lr=learning_rate)

grid, buckets, width, height = build_board(map_name, trackers)
start_x = random.randint(0, width - 1)
visualize_grid(grid, width, height, ball_position=(start_x, height - 1), buckets=buckets)
most_recent_steps = deque(maxlen=100)

# === Training Loop ===
for episode in range(episodes):
    # regenerate a fresh board per episode
    grid, buckets, width, height = build_board(map_name, trackers)
    start_x = random.randint(0, width - 1)

    # track how many steps were taken this episode
    steps_before = total_decision_steps[0]

    # drop the ball and get result of episode
    episode_final_reward, final_bucket, stars_collected, steps_taken = drop_ball(
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
            "Experience": Experience,
            "target_net": target_net,
            "soft_update_alpha": soft_update_alpha,
            "update_target_network": update_target_network,
            "total_decision_steps": total_decision_steps,
            "target_update_frequency": target_update_frequency,
            "episode": episode,
            "q_model": online_net
        },
        # visualize=False  # set True to debug
        # visualize=(episode == episodes - 1)
    )

    # post-episode stats
    steps_taken = total_decision_steps[0] - steps_before
    most_recent_steps = deque(maxlen=100) if episode == 0 else most_recent_steps
    most_recent_steps.append(steps_taken)
    total_stars_collected += len(stars_collected)
    episode_rewards_history.append(episode_final_reward)
    most_recent_rewards.append(episode_final_reward)
    episode_accuracy.append(1 if final_bucket == target_bucket else 0)  # success = 1, failure = 0

    # perform batch learning steps after episode
    losses = []
    if len(replay_buffer) > batch_size:
        for _ in range(10):  # optional: increase for more updates
            loss = learn(
                episode,
                total_decision_steps,
                replay_buffer,
                batch_size,
                online_net,
                target_net,
                optimizer,
                discount_factor,
                soft_update_alpha,
                target_update_frequency,
                width,
                height,
                episodes
            )

            if loss is not None:
                losses.append(loss)

    # track average loss this episode
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    episode_losses.append(avg_loss)

    # decay exploration
    if episode >= initial_free_exploration:
        exploration_rate = max(min_exploration, exploration_rate * exploration_decay)

    # episode summary
    # print(f"[Episode {episode+1}] Reward: {episode_final_reward} | Bucket: {final_bucket} | Stars: {len(stars_collected)} | Steps: {steps_taken} | ε: {exploration_rate:.2f}", flush=True)

    # periodic performance print
    if (episode + 1) % 50 == 0:
        avg_reward = sum(most_recent_rewards) / len(most_recent_rewards)
        avg_stars = total_stars_collected / (episode + 1)
        avg_loss = sum(losses) / len(losses)
        avg_steps = sum(most_recent_steps) / len(most_recent_steps)
        print(f"Ep {episode + 1} | Avg R (last 100): {avg_reward:.2f} | Stars: {avg_stars:.2f} | Steps: {avg_steps:.2f} | Loss: {avg_loss:.4f} | ε: {exploration_rate:.2f}", flush=True)

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

def smooth(series, weight=0.9):
    smoothed = []
    last = series[0]
    for point in series:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

# smooth and plot reward
plt.figure(figsize=(10, 5))
plt.plot(smooth(episode_rewards_history), label="Smoothed Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Smoothed Reward over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("smoothed_reward.png")

# optionally: smooth and plot loss
plt.figure(figsize=(10, 5))
plt.plot(smooth(episode_losses), label="Smoothed Loss", color="orange")
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.title("Smoothed Loss over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("smoothed_loss.png")

# smooth and plot accuracy
plt.figure(figsize=(10, 5))
plt.plot(smooth(episode_accuracy), label="Smoothed Accuracy", color="green")
plt.xlabel("Episode")
plt.ylabel("Target Bucket Accuracy")
plt.title("Smoothed Accuracy over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("smoothed_accuracy.png")