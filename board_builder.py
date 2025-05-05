from grid_utils import (
    generate_grid,
    mark_ledge,
    mark_slide,
    mark_spike,
    mark_pipe,
    mark_block,
    mark_buckets,
)

def build_board(name, trackers):
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

    if name == "default":
        width, height = 15, 30
        grid = generate_grid(width, height)
        mark_ledge(grid, 2, 5, 27, button_tracker, ledge_tracker)
        mark_ledge(grid, 6, 4, 24, button_tracker, ledge_tracker)
        mark_ledge(
            grid, 1, 6, 21, 
            button_tracker, ledge_tracker, 
            button_x=1, block_row_y=19, 
            button_to_block_map=trackers["button_to_block_map"]
        )
        mark_ledge(grid, 9, 5, 19, button_tracker, ledge_tracker)
        mark_ledge(grid, 3, 7, 17, button_tracker, ledge_tracker)
        mark_ledge(grid, 7, 6, 15, button_tracker, ledge_tracker)
        mark_ledge(grid, 0, 5, 13, button_tracker, ledge_tracker)
        mark_ledge(grid, 9, 6, 11, button_tracker, ledge_tracker)
        mark_ledge(
            grid, 0, 9, 9, 
            button_tracker, ledge_tracker, 
            button_x=0, block_row_y=5, 
            button_to_block_map=trackers["button_to_block_map"])
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
        mark_block(grid, width, 19, blocks)

    elif name == "hard":
        width, height = 25, 100
        grid = generate_grid(width, height)

        # Ledges
        mark_ledge(grid, 23, 2, 98, button_tracker, ledge_tracker)
        mark_ledge(grid, 0, 5, 95, button_tracker, ledge_tracker)
        mark_ledge(grid, 10, 6, 95, button_tracker, ledge_tracker)
        mark_ledge(grid, 3, 8, 91, button_tracker, ledge_tracker)
        mark_ledge(grid, 15, 9, 89, button_tracker, ledge_tracker)
        mark_ledge(grid, 2, 5, 85, button_tracker, ledge_tracker, button_x=4, block_row_y=80, button_to_block_map=trackers["button_to_block_map"])
        mark_ledge(grid, 15, 5, 83, button_tracker, ledge_tracker)
        mark_ledge(grid, 6, 4, 80, button_tracker, ledge_tracker)
        mark_ledge(grid, 0, 8, 77, button_tracker, ledge_tracker)
        mark_ledge(grid, 10, 5, 75, button_tracker, ledge_tracker, button_x=14, block_row_y=70, button_to_block_map=trackers["button_to_block_map"])
        mark_ledge(grid, 0, 7, 74, button_tracker, ledge_tracker)
        mark_ledge(grid, 7, 5, 70, button_tracker, ledge_tracker)
        mark_ledge(grid, 4, 6, 67, button_tracker, ledge_tracker)
        mark_ledge(grid, 14, 7, 65, button_tracker, ledge_tracker)
        mark_ledge(grid, 0, 4, 63, button_tracker, ledge_tracker)
        mark_ledge(grid, 10, 6, 61, button_tracker, ledge_tracker, button_x=14, block_row_y=60, button_to_block_map=trackers["button_to_block_map"])
        mark_ledge(grid, 20, 4, 59, button_tracker, ledge_tracker)
        mark_ledge(grid, 1, 6, 56, button_tracker, ledge_tracker)
        mark_ledge(grid, 14, 7, 55, button_tracker, ledge_tracker)
        mark_ledge(grid, 5, 8, 53, button_tracker, ledge_tracker)
        mark_ledge(grid, 12, 5, 50, button_tracker, ledge_tracker)
        mark_ledge(grid, 3, 5, 47, button_tracker, ledge_tracker, button_x=7, block_row_y=45, button_to_block_map=trackers["button_to_block_map"])
        mark_ledge(grid, 15, 5, 44, button_tracker, ledge_tracker)
        mark_ledge(grid, 8, 4, 42, button_tracker, ledge_tracker)
        mark_ledge(grid, 18, 4, 40, button_tracker, ledge_tracker)
        mark_ledge(grid, 0, 5, 37, button_tracker, ledge_tracker)
        mark_ledge(grid, 10, 8, 35, button_tracker, ledge_tracker)
        mark_ledge(grid, 5, 5, 32, button_tracker, ledge_tracker)
        mark_ledge(grid, 15, 7, 31, button_tracker, ledge_tracker)
        mark_ledge(grid, 20, 4, 29, button_tracker, ledge_tracker, button_x=23, block_row_y=25, button_to_block_map=trackers["button_to_block_map"])
        mark_ledge(grid, 2, 5, 27, button_tracker, ledge_tracker)
        mark_ledge(grid, 8, 4, 24, button_tracker, ledge_tracker)
        mark_ledge(grid, 14, 6, 21, button_tracker, ledge_tracker)
        mark_ledge(grid, 0, 5, 18, button_tracker, ledge_tracker)
        mark_ledge(grid, 10, 5, 17, button_tracker, ledge_tracker)
        mark_ledge(grid, 20, 4, 15, button_tracker, ledge_tracker)
        mark_ledge(grid, 6, 5, 12, button_tracker, ledge_tracker)
        mark_ledge(grid, 3, 6, 10, button_tracker, ledge_tracker)
        mark_ledge(grid, 14, 5, 8, button_tracker, ledge_tracker)
        mark_ledge(grid, 0, 4, 6, button_tracker, ledge_tracker)
        mark_ledge(grid, 7, 4, 4, button_tracker, ledge_tracker)
        mark_ledge(grid, 20, 4, 3, button_tracker, ledge_tracker)
        mark_ledge(grid, 0, 4, 2, button_tracker, ledge_tracker)
        mark_ledge(grid, 8, 5, 1, button_tracker, ledge_tracker)
        mark_ledge(grid, 18, 4, 1, button_tracker, ledge_tracker)


        # Slides
        mark_slide(grid, 0, 93, 4, "forward")   
        mark_slide(grid, 24, 88, 5, "backward")  
        mark_slide(grid, 5, 58, 4, "forward")     
        mark_slide(grid, 18, 41, 5, "backward")   
        mark_slide(grid, 10, 23, 4, "forward")    

        # Spikes
        mark_spike(grid, 7, 3, 38, spike_tracker)
        mark_spike(grid, 16, 1, 20, spike_tracker)

        # Spikes (additional)
        mark_spike(grid, 3, 3, 76, spike_tracker)   # under ledge at 77
        mark_spike(grid, 7, 3, 66, spike_tracker)   # under ledge at 67
        mark_spike(grid, 3, 2, 55, spike_tracker)   # under ledge at 56
        mark_spike(grid, 4, 1, 46, spike_tracker)  # under ledge at 47
        mark_spike(grid, 1, 2, 36, spike_tracker)   # under ledge at 37
        mark_spike(grid, 9, 2, 16, spike_tracker)   # under ledge at 18

        # Pipes: button ledges to block rows
        mark_pipe(grid, 10, 91, 80, pipes, pipe_tracker)
        mark_pipe(grid, 11, 75, 70, pipes, pipe_tracker)
        mark_pipe(grid, 20, 65, 60, pipes, pipe_tracker)
        mark_pipe(grid, 1, 56, 45, pipes, pipe_tracker)
        mark_pipe(grid, 14, 35, 25, pipes, pipe_tracker)

        # More Pipes
        mark_pipe(grid, 23, 98, 89, pipes, pipe_tracker)
        mark_pipe(grid, 18, 89, 83, pipes, pipe_tracker)
        mark_pipe(grid, 20, 59, 55, pipes, pipe_tracker)
        mark_pipe(grid, 16, 55, 50, pipes, pipe_tracker)
        mark_pipe(grid, 16, 55, 50, pipes, pipe_tracker)
        mark_pipe(grid, 16, 55, 50, pipes, pipe_tracker)
        mark_pipe(grid, 18, 44, 40, pipes, pipe_tracker)
        mark_pipe(grid, 20, 40, 31, pipes, pipe_tracker)
        mark_pipe(grid, 2, 18, 6, pipes, pipe_tracker)
        mark_pipe(grid, 0, 6, 2, pipes, pipe_tracker)
        mark_pipe(grid, 12, 17, 1, pipes, pipe_tracker)
        mark_pipe(grid, 18, 8, 1, pipes, pipe_tracker)

        # Blocks (already connected via button above and pipe)
        mark_block(grid, width, 80, blocks)
        mark_block(grid, width, 70, blocks)
        mark_block(grid, width, 60, blocks)
        mark_block(grid, width, 45, blocks)
        mark_block(grid, width, 25, blocks)

        # Bonus Stars (optional rewards)
        bonus_positions = [(24, 98), (0, 74), (0, 63), (24, 59), (19, 44), (0, 18)]

        for x, y in bonus_positions:
            grid[(x, y)] = '☆'
            if (x, y) not in trackers["bonus_star_tracker"]:
                trackers["bonus_star_tracker"][(x, y)] = 0

    else:
        raise ValueError(f"Unknown board name: {name}")

    buckets = mark_buckets(width, 5)
    return grid, buckets, width, height
