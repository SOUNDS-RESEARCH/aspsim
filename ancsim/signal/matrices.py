import numpy as np
import numba as nb
import scipy.signal as spsig
import scipy.linalg as splinalg


def to_length(ar, length, axis=-1):
    """Either truncates or pads ndarray at the end with zeros, 
        so that the length along axis is equal to length."""
    end = min(length, ar.shape[axis])
    slices = [slice(None) for _ in range(ar.ndim)]
    slices[axis] = slice(0,end)
    ar = ar[tuple(slices)]

    pad_tuple = [(0,0) for _ in range(ar.ndim)]
    pad_tuple[axis] = (0, length-end)
    ar = np.pad(ar, pad_tuple)
    return ar

def block_diag(arrays):
    """Creates a block diagonal matrix
        - arrays is a list of ndarrays
        - The operation is performed independently for the first dimension. 
        The length of the first dimension must naturally be the same.
        - arr.ndim must be 3 or more
        (to make a block_diag with ndim=2, just use scipy.block_diag)"""
    assert all([ar.ndim == arrays[0].ndim for ar in arrays])
    assert all([ar.shape[:-2] == arrays[0].shape[:-2] for ar in arrays])
    assert all([ar.dtype == arrays[0].dtype for ar in arrays])
    tot_cols = np.sum([ar.shape[-1] for ar in arrays])
    tot_rows = np.sum([ar.shape[-2] for ar in arrays])
    bc_dim = arrays[0].shape[:-2]
    
    diag_mat = np.zeros(np.concatenate((bc_dim, [tot_rows, tot_cols])), dtype=arrays[0].dtype)

    idx = np.zeros(2, dtype=int)
    for ar in arrays:
        idx_shift = ar.shape[-2:]
        diag_mat[...,idx[0]:idx[0]+idx_shift[0], 
                    idx[1]:idx[1]+idx_shift[1]] = ar

        idx += idx_shift
    return diag_mat

def block_of_toeplitz(block_of_col, block_of_row=None):
    """ block_of_col and block_of_row are of shapes
        (M, N, K_col) and (M, N, K_row)
        
        The values in the last axis for each (m,n) block is turned
        into a toeplitz matrix. The output is a single matrix of shape
        (M*K_col N*K_row)

        If only block_of_col is supplied, the toeplitz 
        matrices are constructed symmetric
    """
    if block_of_row is None:
        if np.iscomplexobj(block_of_col):
            block_of_row = np.conj(block_of_col)
        else:
            block_of_row = block_of_col

    assert block_of_col.shape[:2] == block_of_row.shape[:2]
    len_col = block_of_col.shape[2]
    len_row = block_of_row.shape[2]

    block_mat = np.zeros((block_of_col.shape[0]*len_col, 
                        block_of_col.shape[1]*len_row), 
                        dtype=block_of_col.dtype)
    for m in range(block_of_col.shape[0]):
        for n in range(block_of_col.shape[1]):
            block_mat[m*len_col:(m+1)*len_col, 
                      n*len_row:(n+1)*len_row] = splinalg.toeplitz(block_of_col[m,n,:], 
                                                                   block_of_row[m,n,:])
    return block_mat

def block_transpose(matrix, block_size, out=None):
    """
    Transposes each block individually in a block matrix. 
    block_size is an integer defining the size of the square blocks. 
    Requires square blocks, but not a square block matrix. 

    B = [
        A_11^T, ..., A_1r^T

        A_r1^T, ..., A_rc^T
    ]

    if out is supplied, the transposed matrix will be written into its memory
    """
    assert matrix.shape[0] % block_size == 0
    assert matrix.shape[1] % block_size == 0
    num_rows = matrix.shape[0] // block_size
    num_cols = matrix.shape[1] // block_size

    if out is not None:
        transposed_mat = out
    else:
        transposed_mat = np.zeros((matrix.shape[1], matrix.shape[0]))

    for r in range(num_rows):
        for c in range(num_cols):
            transposed_mat[r*block_size:(r+1)*block_size, c*block_size:(c+1)*block_size] = \
                    matrix[r*block_size:(r+1)*block_size, c*block_size:(c+1)*block_size].T
    return transposed_mat