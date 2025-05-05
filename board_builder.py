from grid_utils import (
    generate_grid,
    mark_ledge,
    mark_slide,
    mark_spike,
    mark_pipe,
    mark_block,
    mark_buckets,
)

def build_board(name, width, height, trackers):
    """
    Builds a board by name.

    Parameters:
    - name (str): which board setup to build
    - width (int): width of the grid
    - height (int): height of the grid
    - trackers (dict): contains all shared tracker/global dicts

    Returns:
    - grid (dict): the board grid
    - buckets (dict): maps x to bucket index
    """
    # unpack tracker dicts
    blocks = trackers["blocks"]
    pipes = trackers["pipes"]
    button_tracker = trackers["button_tracker"]
    pipe_tracker = trackers["pipe_tracker"]
    spike_tracker = trackers["spike_tracker"]
    ledge_tracker = trackers["ledge_tracker"]

    # clear global dicts
    blocks.clear()
    pipes.clear()

    grid = generate_grid(width, height)

    if name == "default":
        mark_ledge(grid, 2, 5, 27, button_tracker, ledge_tracker)
        mark_ledge(grid, 6, 4, 24, button_tracker, ledge_tracker)
        mark_ledge(grid, 1, 6, 21, button_tracker, ledge_tracker)
        mark_ledge(grid, 9, 5, 19, button_tracker, ledge_tracker)
        mark_ledge(grid, 3, 7, 17, button_tracker, ledge_tracker)
        mark_ledge(grid, 7, 6, 15, button_tracker, ledge_tracker)
        mark_ledge(grid, 0, 5, 13, button_tracker, ledge_tracker)
        mark_ledge(grid, 9, 6, 11, button_tracker, ledge_tracker)
        mark_ledge(grid, 0, 9, 9, button_tracker, ledge_tracker, button_x=0)
        mark_ledge(grid, 5, 6, 7, button_tracker, ledge_tracker)
        mark_ledge(grid, 4, 5, 5, button_tracker, ledge_tracker)
        mark_ledge(grid, 8, 4, 3, button_tracker, ledge_tracker)
        mark_ledge(grid, 4, 8, 2, button_tracker, ledge_tracker)

        mark_slide(grid, 0, 28, 4, "forward")
        mark_slide(grid, 13, 23, 3, "backward")
        mark_slide(grid, 14, 16, 4, "backward")
        mark_slide(grid, 0, 12, 2, "forward")

        mark_spike(grid, 5, 4, 18, spike_tracker)
        mark_spike(grid, 7, 2, 6, spike_tracker)

        mark_pipe(grid, 6, 27, 24, pipes, pipe_tracker)
        mark_pipe(grid, 9, 24, 19, pipes, pipe_tracker)
        mark_pipe(grid, 4, 9, 5, pipes, pipe_tracker)

        mark_block(grid, width, 5, blocks)

    elif name == "flat":
        mark_ledge(grid, 0, width, height - 2, button_tracker, ledge_tracker)
        mark_spike(grid, 0, width, 1, spike_tracker)

    else:
        raise ValueError(f"Unknown board name: {name}")

    buckets = mark_buckets(width, 5)
    return grid, buckets
