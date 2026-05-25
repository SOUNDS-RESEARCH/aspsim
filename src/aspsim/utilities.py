"""Miscellaneous utility functions that does not currently fit into another module."""

import numpy as np


def calc_block_sizes(num_samples: int, start_idx: int, block_size: int):
    """Calculate block sizes for processing samples in blocks.

    Parameters
    ----------
    num_samples : int
        Total number of samples to process
    start_idx : int
        Starting index of the first block.
    block_size : int
        Size of each block. The last block may be smaller if num_samples is not divisible by block_size.

    Returns
    -------
    block_sizes : list of int
        List of block sizes, where the sum of block_sizes is equal to num_samples.
    """
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
