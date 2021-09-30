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

#@nb.jit
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



def corr_matrix(seq1, seq2, lag1, lag2):
    """seq1 has shape (sumCh, numCh1, numSamples) or (numCh1, numSamples)
        seq2 has shape (sumCh, numCh2, numSamples) or (numCh2, numSamples)
    
        Outputs a correlation matrix of size 
        (numCh1*lag1, numCh2*lag2)
        
        Sums over the first dimension"""
    assert seq1.ndim == seq2.ndim
    if seq1.ndim == 2:
        seq1 = np.expand_dims(seq1, 0)
        seq2 = np.expand_dims(seq2, 0)
    assert seq1.shape[0] == seq2.shape[0]
    assert seq1.shape[2] == seq2.shape[2]
    sum_ch = seq1.shape[0]
    seq_len = seq1.shape[2]

    if seq_len < max(lag1, lag2):
        pad_len = max(lag1, lag2) - seq_len
        seq1 = np.pad(seq1, ((0,0), (0,0), (0,pad_len)))
        seq2 = np.pad(seq2, ((0,0), (0,0), (0,pad_len)))

    R = np.zeros((seq1.shape[1]*lag1, seq2.shape[1]*lag2))
    for i in range(sum_ch):
        R += _corr_matrix(seq1[i,...], seq2[i,...], lag1, lag2) / sum_ch
    return R 

def _corr_matrix(seq1, seq2, lag1, lag2):
    """seq1 has shape (numChannels_1, numSamples)
        seq2 has shape (numChannels_2, numSamples)
    
        Outputs a correlation matrix of size 
        (numChannels_1*lag1, numChannels_2*lag2)"""
    assert seq1.ndim == seq2.ndim == 2
    assert seq1.shape[-1] == seq2.shape[-1]
    seqLen = seq1.shape[-1]
    numChannels1 = seq1.shape[0]
    numChannels2 = seq2.shape[0]

    corr = np.zeros((numChannels1, numChannels2, 2*seqLen-1))
    for i in range(numChannels1):
        for j in range(numChannels2):
            corr[i,j,:] = spsig.correlate(seq1[i,:], seq2[j,:], mode="full")
    corr /= seqLen
    corrMid = seqLen - 1
    R = np.zeros((seq1.shape[0]*lag1, seq2.shape[0]*lag2))
    for c1 in range(seq1.shape[0]):
        for l1 in range(lag1):
            rowIdx = c1*lag1 + l1
            R[rowIdx,:] = corr[c1, :, corrMid-l1:corrMid-l1+lag2].ravel()
    return R