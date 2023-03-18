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


def flatten_dict(dict_to_flatten, parent_key="", sep="~"):
    items = []
    for key, value in dict_to_flatten.items():
        new_key = parent_key + sep + key if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, sep).items())
        else:
            items.append((new_key, value))
    new_dict = dict(items)
    return new_dict


def restack_dict(dict_to_stack, sep="~"):
    """Only accepts dicts of depth 2.
    All elements must be of that depth."""
    extracted_data = {}
    for multi_key in dict_to_stack.keys():
        key_list = multi_key.split(sep)
        if len(key_list) > 2:
            raise NotImplementedError
        if key_list[0] not in extracted_data:
            extracted_data[key_list[0]] = {}
        extracted_data[key_list[0]][key_list[1]] = dict_to_stack[multi_key]
    return extracted_data

