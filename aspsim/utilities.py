import numpy as np


def calc_block_sizes(num_samples, start_idx, block_size):
    left_in_block = block_size - start_idx
    sample_counter = 0
    block_sizes = []
    while sample_counter < num_samples:
        block_len = np.min((num_samples - sample_counter, left_in_block))
        block_sizes.append(block_len)
        sample_counter += block_len
        left_in_block -= block_len
        if left_in_block == 0:
            left_in_block = block_size
    return block_sizes