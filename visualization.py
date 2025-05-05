import pandas as pd
from IPython.display import display

def visualize_grid(grid, width, height, ball_position=None, buckets=None):
    x_labels = "   " + " ".join(str(i % 10) for i in range(width))  # x-axis labels
    print(x_labels)  # print top x-axis

    for y in range(height - 1, -1, -1):
        row = f"{y:2} "  # y-axis label
        for x in range(width):
            if ball_position and (x, y) == ball_position:
                row += "X "
            else:
                tile = grid.get((x, y), ' ')
                row += f"{tile} "
        print(row)  # print full row

    bucket_row = "   "
    if buckets:
        for x in range(width):
            bucket_row += str(buckets.get(x, ' ')) + " "
    else:
        bucket_row += "  " * width
    print(bucket_row)  # print bucket labels
    print(x_labels)  # print bottom x-axis
    print("===" + "=" * (2 * width))  # horizontal divider

def print_training_stats(trackers, q_table):
    print("\nLedge State Visit Statistics (Top 15):")
    sorted_ledges = sorted(trackers["ledge_tracker"].items(), key=lambda item: item[1], reverse=True)
    for state, count in sorted_ledges[:15]:
        print(f"State {state} visited {count} times")

    print("\nBlock Row State Visit Statistics (Top 15):")
    sorted_blocks = sorted(trackers["block_row_tracker"].items(), key=lambda item: item[1], reverse=True)
    for state, count in sorted_blocks[:15]:
        print(f"State {state} visited {count} times")

    print("\nBucket Landing Statistics:")
    total_landings = sum(trackers["bucket_tracker"].values())
    for bucket_id, count in sorted(trackers["bucket_tracker"].items()):
        if count > 0:
            percent = (count / total_landings * 100) if total_landings > 0 else 0
            print(f"Bucket {bucket_id}: {count} landings ({percent:.1f}%)")

    print("\nSpike Hit Statistics:")
    for y, count in sorted(trackers["spike_tracker"].items()):
        if count > 0:
            print(f"Spike row {y} hit {count} times")

    print("\nPipe Usage Statistics:")
    for (x, y), count in sorted(trackers["pipe_tracker"].items()):
        if count > 0:
            print(f"Pipe at ({x}, {y}) used {count} times")

    print("\nButton Press Statistics:")
    for (x, y), count in sorted(trackers["button_tracker"].items()):
        if count > 0:
            print(f"Button at ({x}, {y}) pressed {count} times")

    # Q-table without sorting
    q_data_for_df = {}
    for state, actions in q_table.items():
        state_display = (state[0], tuple(sorted(list(state[1]))))
        q_data_for_df[state_display] = actions

    q_df = pd.DataFrame(q_data_for_df).T

    if not q_df.empty:
        q_df.index = pd.MultiIndex.from_tuples(
            q_df.index,
            names=["position", "buttons_pressed"]
        )

        index_df = q_df.index.to_frame(index=False)

        def get_y(pos):
            if isinstance(pos, tuple):
                if pos[0] == 'block':
                    return pos[1]
                elif isinstance(pos[1], int):
                    return pos[1]
            return -1

        index_df["y_value"] = index_df["position"].apply(get_y)
        index_df = index_df.sort_values("y_value", ascending=False).drop(columns="y_value")
        q_df = q_df.loc[pd.MultiIndex.from_frame(index_df)]

        print("\nOnline Q-Table (Sorted by descending Y in position):")
        with pd.option_context("display.multi_sparse", False):
            display(q_df.head(20))
    else:
        print("\nQ-Table is empty.")
