import random
from collections import defaultdict, deque
from grid_utils import unmark_block
from state_utils import identify_decision_state, choose_action, is_block_row, get_valid_diagonal_moves, handle_blocks, drop_ball, initialize_trackers
from board_builder import build_board
from visualization import print_training_stats, visualize_grid

# declaration of tracker dictionaries
trackers = initialize_trackers()

q_table = defaultdict(dict)
episode_rewards_history = []
most_recent_rewards = deque(maxlen=100)
most_recent_steps = deque(maxlen=100)
total_stars_collected = 0

# q-learning parameters
learning_rate = 0.1
discount_factor = 0.99
exploration_rate = 1.0  # start fully exploratory
exploration_decay = 0.999  # reduce randomness over time
min_exploration = 0.01  # smallest possible exploration rate
episodes = 2000  # number of training episodes
initial_free_exploration = 0

# train agent
target_bucket = 2  # the bucket the agent should aim for
map_name = "hard"

grid, buckets, width, height = build_board(map_name, trackers)

start_x = random.randint(0, width - 1)

visualize_grid(grid, width, height, ball_position=(start_x, height - 1), buckets=buckets)

for episode in range(episodes):
    grid, buckets, width, height = build_board(map_name, trackers)

    start_x = random.randint(0, width - 1)
    
    # visualize_grid(grid, width, height, ball_position=(start_x, height - 1), buckets=buckets)

    state_action_pairs, reward, stars_collected, final_bucket, steps_taken = drop_ball(
        grid=grid,
        width=width,
        height=height,
        start_x=start_x,
        buckets=buckets,
        target_bucket=target_bucket,
        mode="q",
        exploration_rate=exploration_rate,
        q_table=q_table,
        trackers=trackers,
        episode=episode,
        # visualize=(episode == episodes - 1)
        visualize=False,
    )

    total_stars_collected += len(stars_collected)

    # update Q-table using recorded decisions
    for state, action in reversed(state_action_pairs):
        current_q = q_table[state][action]
        best_future_q = max(q_table.get(state, {}).values(), default=0)
        q_table[state][action] = current_q + learning_rate * (reward + discount_factor * best_future_q - current_q)

    episode_rewards_history.append(reward) # save the final reward
    most_recent_rewards.append(reward)
    most_recent_steps.append(steps_taken)

    # decay exploration rate
    if episode >= initial_free_exploration:
        exploration_rate = max(min_exploration, exploration_rate * exploration_decay)

    # episode summary
    # print(f"[Episode {episode+1}] Reward: {reward} | Bucket: {final_bucket} | Stars: {len(stars_collected)} | Steps: {steps_taken} | ε: {exploration_rate:.2f}", flush=True)

    # print progress
    if (episode + 1) % 100 == 0:
        avg_reward = sum(most_recent_rewards) / len(most_recent_rewards)
        avg_stars = total_stars_collected / (episode + 1)
        avg_steps = sum(most_recent_steps) / len(most_recent_steps)
        print(f"Episode {episode + 1} | Avg Reward (Last 100): {avg_reward:.2f} | Avg Stars: {avg_stars:.2f} | Avg Steps: {avg_steps:.2f} | Exploration Rate: {exploration_rate:.2f} | Q-States: {len(q_table)}")

print_training_stats(
    trackers,
    q_table
)
