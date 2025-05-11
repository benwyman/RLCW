# === Imports ===
import random
from collections import defaultdict, deque, namedtuple
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

from state_utils_deep import (
    drop_ball, initialize_trackers, 
    PrioritizedReplayBuffer, preprocess_state, learn,
    update_target_network, QNetwork
)
from board_builder import build_board
from visualization import visualize_grid

# === Seaborn styling ===
sns.set(style="whitegrid")

# === Device Configuration ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Training Hyperparameters ===
learning_rate_values = [0.1, 1e-5]               # optimizer learning rate
discount_factor = 0.99              # gamma: importance of future rewards
exploration_decay_values = [0.995, 0.999, 0.9999]            # decay per episode
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

# === Store all results ===
all_results = []
total_trials = len(learning_rate_values) * len(exploration_decay_values)
trial_id = 0

# === Grid Search ===
for learning_rate, exploration_decay in itertools.product(learning_rate_values, exploration_decay_values):
    trial_id += 1
    print(f"\n[Trial {trial_id}/{total_trials}] learning_rate={learning_rate}, exploration_decay={exploration_decay}")
    exploration_rate = 1.0              #  reset epsilon: initial exploration probability

    # === Initialize Stats and Environment ===
    trackers = initialize_trackers()   # global stat trackers for board elements
    episode_rewards_history = []       # stores total reward for each episode
    most_recent_rewards = deque(maxlen=100)  # sliding window for recent episode rewards
    total_stars_collected = 0          # tracks cumulative star count
    total_decision_steps = [0]         # decision step count (mutable list)
    episode_losses = []                # average loss per episode (for plotting)
    episode_success_history = []       # success tracking for accuracy
    episode_steps_history = []         # track steps taken per episode (added)

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

    start_x = random.randint(0, width - 1)
    # visualize_grid(grid, width, height, ball_position=(start_x, height - 1), buckets=buckets)
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

        # === Track success for accuracy ===
        is_success = (final_bucket == target_bucket)  # success if the agent reaches the target bucket
        episode_success_history.append(is_success)

        # === Track steps taken per episode ===
        episode_steps_history.append(steps_taken)

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
        if (episode + 1) % 100 == 0:
            avg_reward = sum(most_recent_rewards) / len(most_recent_rewards)
            avg_stars = total_stars_collected / (episode + 1)
            avg_loss = sum(losses) / len(losses)
            avg_steps = sum(most_recent_steps) / len(most_recent_steps)
            print(f"Ep {episode + 1} | Avg R (last 100): {avg_reward:.2f} | Stars: {avg_stars:.2f} | Steps: {avg_steps:.2f} | Loss: {avg_loss:.4f} | ε: {exploration_rate:.2f}", flush=True)

    all_results.append({
    "learning_rate": learning_rate,
    "exploration_decay": exploration_decay,
    "rewards": episode_rewards_history,
    "accuracy": episode_success_history,  # Store accuracy
    "steps": episode_steps_history,
    })

# === Graphing Functions ===
def smooth(series, weight=0.9):
    smoothed = []
    last = series[0]
    for point in series:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_top_configs(configs, metric_key, ylabel, title):
    plt.figure(figsize=(10, 6))
    for idx, config in enumerate(configs, 1):
        series = config[metric_key]
        smoothed = smooth(series)
        label = f"Top {idx}: α={config['learning_rate']}, decay={config['exploration_decay']}"
        plt.plot(smoothed, label=label)
        plt.text(len(smoothed) - 1, smoothed[-1], f"{smoothed[-1]:.2f}", fontsize=8, ha='left', va='center')

    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"top3_{metric_key}.png")


def plot_metric_by_hyperparam_smoothed(all_results, metric_key, hyperparam_key, ylabel, title_prefix):
    grouped = defaultdict(list)
    for result in all_results:
        key = result[hyperparam_key]
        grouped[key].append(result[metric_key])

    plt.figure(figsize=(10, 6))
    for value, series_list in grouped.items():
        avg_series = [sum(ep) / len(ep) for ep in zip(*series_list)]
        smoothed = smooth(avg_series)
        plt.plot(smoothed, label=f"{hyperparam_key}={value}")
        plt.text(len(smoothed) - 1, smoothed[-1], f"{smoothed[-1]:.2f}", fontsize=8, ha='left', va='center')

    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(f"{title_prefix} by {hyperparam_key}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{metric_key}_smoothed_{hyperparam_key}.png")

def plot_cumulative_accuracy_smoothed(all_results, hyperparam_key):
    grouped = defaultdict(list)
    for result in all_results:
        key = result[hyperparam_key]
        grouped[key].append(result["accuracy"])

    plt.figure(figsize=(10, 6))
    for value, acc_lists in grouped.items():
        avg_acc = []
        for episodes in zip(*acc_lists):
            cum_sum = 0
            running_acc = []
            for i, success in enumerate(episodes, 1):
                cum_sum += success
                running_acc.append(cum_sum / i)
            avg_acc.append(running_acc[-1])
        smoothed = smooth(avg_acc)
        plt.plot(smoothed, label=f"{hyperparam_key}={value}")
        plt.text(len(smoothed) - 1, smoothed[-1], f"{smoothed[-1]:.2f}", fontsize=8, ha='left', va='center')

    plt.xlabel("Episode")
    plt.ylabel("Cumulative Accuracy")
    plt.title(f"Smoothed Target Bucket Accuracy by {hyperparam_key}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"accuracy_smoothed_{hyperparam_key}.png")

# === Smoothed per-hyperparameter plots ===
plot_metric_by_hyperparam_smoothed(all_results, "steps", "learning_rate", "Steps", "Smoothed Steps per Episode")
plot_metric_by_hyperparam_smoothed(all_results, "steps", "exploration_decay", "Steps", "Smoothed Steps per Episode")

plot_metric_by_hyperparam_smoothed(all_results, "rewards", "learning_rate", "Reward", "Smoothed Reward per Episode")
plot_metric_by_hyperparam_smoothed(all_results, "rewards", "exploration_decay", "Reward", "Smoothed Reward per Episode")

plot_cumulative_accuracy_smoothed(all_results, "learning_rate")
plot_cumulative_accuracy_smoothed(all_results, "exploration_decay")

# === Top 3 config plots ===
plot_top_configs(top_configs, "steps", "Steps", "Top 3 Configs: Steps per Episode")
plot_top_configs(top_configs, "rewards", "Reward", "Top 3 Configs: Reward per Episode")
plot_top_configs(top_configs, "accuracy", "Cumulative Accuracy", "Top 3 Configs: Accuracy Over Time")