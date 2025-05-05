def generate_grid(width, height):
    grid = {}
    for y in range(height):
        for x in range(width):
            if (y % 2 == 0 and x % 2 == 1) or (y % 2 == 1 and x % 2 == 0):
                grid[(x, height - 1 - y)] = 'O'  # place pegs in a checkered pattern
            else:
                grid[(x, height - 1 - y)] = ' '  # empty spaces between pegs
    return grid

def mark_ledge(grid, start_x, length, ledge_y, button_tracker, ledge_tracker, button_x=None):
    # place a horizontal ledge starting at start_x on row ledge_y
    for x in range(start_x, start_x + length): 
        if x == button_x:
            grid[(x, ledge_y)] = '⬒'  # mark a special button tile
            button_tracker[(x, ledge_y)]  # initialize button in tracker
        else:
            grid[(x, ledge_y)] = '_'  # normal ledge tile
    ledge_tracker[((start_x, ledge_y), frozenset())] # initialize ledge visit tracker

def mark_spike(grid, start_x, length, spike_y, spike_tracker):
    for x in range(start_x, start_x + length):
        grid[(x, spike_y)] = '^'
    spike_tracker[spike_y]  # auto-initializes to 0 if not already set

def mark_pipe(grid, x, y1, y2, pipes, pipe_tracker):
    # mark a vertical pipe that connects y1 and y2 at column x
    top = max(y1, y2)
    bottom = min(y1, y2)

    for y in range(bottom, top + 1):
        if y == top:
            grid[(x, y)] = '⤓'  # down pipe entrance
        elif y == bottom:
            grid[(x, y)] = '↥'  # up pipe entrance
        else:
            tile = grid.get((x, y), ' ')
            grid[(x, y)] = 'Φ' if tile == 'O' else '|'  # middle of the pipe

    # connect both ends in the pipes map
    pipes[(x, top)] = (x, bottom)
    pipes[(x, bottom)] = (x, top)

    # start tracking usage of this pipe
    pipe_tracker[(x, top)]
    pipe_tracker[(x, bottom)]

def mark_slide(grid, start_x, start_y, length, direction):
    slide_char = '\\' if direction == "forward" else '/'
    x, y = start_x, start_y

    for _ in range(length):
        if (x, y) in grid and grid[(x, y)] == 'O':
            grid[(x, y)] = slide_char  # replace pegs with slides

        # replace diagonally in the selected direction
        if direction == "forward":
            x += 1
            y -= 1
        else:
            x -= 1
            y -= 1

def mark_block(grid, width, row_y, blocks):
    if row_y in blocks:
        return  # skip if already marked

    blocks[row_y] = {}  # store original row tiles
    for x in range(width):
        current_tile = grid.get((x, row_y), ' ')
        if current_tile not in {'↥', 'Φ', '⤓', '|'}:  # skip if tile is part of a pipe
            blocks[row_y][x] = current_tile  # remember what was here
            grid[(x, row_y)] = '█'  # mark block tile

def unmark_block(grid, row_y, blocks):
    if row_y not in blocks:
        return  # nothing to unmark

    for x, original_char in blocks[row_y].items():
        grid[(x, row_y)] = original_char  # restore original tile
    del blocks[row_y]  # remove from block tracker

def mark_buckets(width, num_buckets):
    buckets = {}  # maps x to bucket index
    base_size = width // num_buckets  # base size for each bucket
    extra = width % num_buckets  # leftover columns
    middle_bucket = num_buckets // 2  # middle bucket index
    start_x = 0  # starting column for current bucket

    for i in range(num_buckets):
        # add 1 to size if extra columns remain and it's not the middle bucket
        size = base_size + (1 if extra > 0 and i != middle_bucket else 0)
        for x in range(start_x, start_x + size):
            buckets[x] = i  # map each column to bucket index
        start_x += size  # move to next start column
        if extra > 0 and i != middle_bucket:
            extra -= 1  # use up one extra column

    return buckets