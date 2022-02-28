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



def block_diag_multiply(mat, block_left=None, block_right=None, out_matrix=None):
    """
    block_left is size (a, a)
    block_right is size (a, a)
    mat is size (ab, ab) for some integer b
    
    performs the operation block_diag_left @ mat @ block_diag_right, 
    where block_diag is a matrix with multiple copies of block on the diagonal
    block_diag = I_b kron block
    """
    if block_left is not None and block_right is not None:
        assert all([block_left.shape[0] == b for b in (*block_left.shape, *block_right.shape)])
        block_size = block_left.shape[0]
    elif block_left is not None:
        assert block_left.shape[0] == block_left.shape[1]
        block_size = block_left.shape[0]
    elif block_right is not None:
        assert block_right.shape[0] == block_right.shape[1]
        block_size = block_right.shape[0]
    else:
        raise ValueError("Neither block_left or block_right provided")
    
    assert mat.shape[0] == mat.shape[1]
    mat_size = mat.shape[0]
    assert mat_size % block_size == 0
    num_blocks = mat_size // block_size
    
    if out_matrix is None:
        out_matrix = np.zeros((mat_size, mat_size))

    if block_left is not None and block_right is not None:
        for i in range(num_blocks):
            for j in range(num_blocks):
                out_matrix[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = \
                    block_left @ mat[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] @ block_right
    elif block_left is not None:
        for i in range(num_blocks):
            for j in range(num_blocks):
                out_matrix[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = \
                    block_left @ mat[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
    elif block_right is not None:
        for i in range(num_blocks):
            for j in range(num_blocks):
                out_matrix[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = \
                    mat[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] @ block_right

    return out_matrix