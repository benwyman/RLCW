import random
from collections import defaultdict
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from grid_utils import unmark_block
from state_utils import (
    identify_decision_state, choose_action, is_block_row,
    get_valid_diagonal_moves, handle_blocks, drop_ball, initialize_trackers
)
from board_builder import build_board

# === Seaborn styling ===
sns.set(style="whitegrid")

# === Parameter grid ===
alpha_values = [0.01, 0.05, 0.1, 0.3, 0.5]
gamma_values = [0.90, 0.95, 0.98, 0.99, 0.999]
decay_values = [0.95, 0.98, 0.99, 0.995, 0.999]
policy_types = ["epsilon", "softmax"]

# === Static config ===
episodes = 2000
min_exploration = 0.01
initial_free_exploration = 0
target_bucket = 2
map_name = "hard"
softmax_temp = 1.0

# === Store all results ===
all_results = []
total_trials = len(alpha_values) * len(gamma_values) * len(decay_values) * len(policy_types)
trial_id = 1

# === Grid Search ===
for alpha, gamma, decay, policy in itertools.product(alpha_values, gamma_values, decay_values, policy_types):
    print(f"\n[Trial {trial_id}/{total_trials}] α={alpha}, γ={gamma}, decay={decay}, policy={policy}")
    q_table = defaultdict(dict)
    trackers = initialize_trackers()
    episode_rewards_history = []
    episode_success_history = []
    episode_steps_history = []
    episode_spike_history = []
    exploration_rate = 1.0
    for episode in range(episodes):
        grid, buckets, width, height = build_board(map_name, trackers)
        start_x = random.randint(0, width - 1)

        if policy == "softmax":
            temperature = 5 if episode < 200 else softmax_temp  # high, but finite
        else:
            temperature = softmax_temp if policy == "softmax" else None

        state_action_pairs, reward, stars_collected, final_bucket, steps_taken = drop_ball(
            grid=grid,
            width=width,
            height=height,
            start_x=start_x,
            buckets=buckets,
            target_bucket=target_bucket,
            exploration_rate=exploration_rate,
            q_table=q_table,
            trackers=trackers,
            episode=episode,
            policy=policy,
            temperature=temperature,
            visualize=False,
        )

        spike_hits = sum(trackers["spike_tracker"].values())
        episode_spike_history.append(spike_hits)
        
        is_success = (final_bucket == target_bucket)
        episode_success_history.append(is_success)

        for state, action in reversed(state_action_pairs):
            current_q = q_table[state][action]
            best_future_q = max(q_table.get(state, {}).values(), default=0)
            q_table[state][action] = current_q + alpha * (reward + gamma * best_future_q - current_q)

        episode_rewards_history.append(reward)
        episode_steps_history.append(steps_taken)

        if episode >= initial_free_exploration:
            exploration_rate = max(min_exploration, exploration_rate * decay)

        # episode summary
        # print(f"[Episode {episode+1}] Reward: {reward} | Bucket: {final_bucket} | Stars: {len(stars_collected)} | Steps: {steps_taken} | ε: {exploration_rate:.2f}", flush=True)

        # print progress
        """
        if (episode + 1) % 100 == 0:
            recent_rewards = episode_rewards_history[-100:]
            recent_steps = episode_steps_history[-100:]
            recent_successes = episode_success_history[-100:]

            avg_reward = sum(recent_rewards) / len(recent_rewards)
            avg_steps = sum(recent_steps) / len(recent_steps)
            avg_accuracy = sum(recent_successes) / len(recent_successes)

            print(
                f"Episode {episode + 1} | "
                f"Avg Reward (Last 100): {avg_reward:.2f} | "
                f"Accuracy: {avg_accuracy:.2f} | "
                f"Avg Steps: {avg_steps:.2f} | "
                f"ε: {exploration_rate:.2f} | "
                f"Q-States: {len(q_table)}"
            )
        """


    trial_id += 1
    all_results.append({
        "alpha": alpha,
        "gamma": gamma,
        "decay": decay,
        "policy": policy,
        "rewards": episode_rewards_history,
        "steps": episode_steps_history,
        "accuracy": episode_success_history,
        "spikes": episode_spike_history
    })

def smooth(series, weight=0.9):
    smoothed = []
    last = series[0]
    for point in series:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

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

def plot_cumulative_spikes_smoothed(all_results, hyperparam_key):
    grouped = defaultdict(list)
    for result in all_results:
        key = result[hyperparam_key]
        grouped[key].append(result["spikes"])

    plt.figure(figsize=(10, 6))
    for value, spike_lists in grouped.items():
        avg_cum_spikes = []
        for episodes in zip(*spike_lists):
            cum_sum = 0
            running_cum = []
            for i, spike_count in enumerate(episodes, 1):
                cum_sum += spike_count
                running_cum.append(cum_sum)
            avg_cum_spikes.append(running_cum[-1])
        smoothed = smooth(avg_cum_spikes)
        plt.plot(smoothed, label=f"{hyperparam_key}={value}")
        plt.text(len(smoothed) - 1, smoothed[-1], f"{smoothed[-1]:.0f}", fontsize=8, ha='left', va='center')

    plt.xlabel("Episode")
    plt.ylabel("Cumulative Spikes Hit")
    plt.title(f"Cumulative Spikes Hit by {hyperparam_key}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"spikes_cumulative_{hyperparam_key}.png")

def get_top_configs(all_results, metric_key="rewards", top_n=3):
    scored = []
    for result in all_results:
        score = np.mean(result[metric_key][-100:])
        scored.append((score, result))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [entry[1] for entry in scored[:top_n]]

def plot_top_configs(configs, metric_key, ylabel, title):
    plt.figure(figsize=(10, 6))
    for idx, config in enumerate(configs, 1):
        series = config[metric_key]
        smoothed = smooth(series)
        label = f"Top {idx}: α={config['alpha']}, γ={config['gamma']}, decay={config['decay']}, π={config['policy']}"
        plt.plot(smoothed, label=label)
        plt.text(len(smoothed) - 1, smoothed[-1], f"{smoothed[-1]:.2f}", fontsize=8, ha='left', va='center')

    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"top3_{metric_key}.png")

# === Smoothed per-hyperparameter plots ===
plot_metric_by_hyperparam_smoothed(all_results, "steps", "alpha", "Steps", "Smoothed Steps per Episode")
plot_metric_by_hyperparam_smoothed(all_results, "steps", "gamma", "Steps", "Smoothed Steps per Episode")
plot_metric_by_hyperparam_smoothed(all_results, "steps", "decay", "Steps", "Smoothed Steps per Episode")
plot_metric_by_hyperparam_smoothed(all_results, "steps", "policy", "Steps", "Smoothed Steps per Episode")

plot_metric_by_hyperparam_smoothed(all_results, "rewards", "alpha", "Reward", "Smoothed Reward per Episode")
plot_metric_by_hyperparam_smoothed(all_results, "rewards", "gamma", "Reward", "Smoothed Reward per Episode")
plot_metric_by_hyperparam_smoothed(all_results, "rewards", "decay", "Reward", "Smoothed Reward per Episode")
plot_metric_by_hyperparam_smoothed(all_results, "rewards", "policy", "Reward", "Smoothed Reward per Episode")

plot_cumulative_accuracy_smoothed(all_results, "alpha")
plot_cumulative_accuracy_smoothed(all_results, "gamma")
plot_cumulative_accuracy_smoothed(all_results, "decay")
plot_cumulative_accuracy_smoothed(all_results, "policy")

plot_cumulative_spikes_smoothed(all_results, "alpha")
plot_cumulative_spikes_smoothed(all_results, "gamma")
plot_cumulative_spikes_smoothed(all_results, "decay")
plot_cumulative_spikes_smoothed(all_results, "policy")

# === Top 3 config plots ===
top_configs = get_top_configs(all_results, "rewards", top_n=3)
plot_top_configs(top_configs, "steps", "Steps", "Top 3 Configs: Steps per Episode")
plot_top_configs(top_configs, "rewards", "Reward", "Top 3 Configs: Reward per Episode")
plot_top_configs(top_configs, "accuracy", "Cumulative Accuracy", "Top 3 Configs: Accuracy Over Time")
