import numpy as np
import scipy.signal as spsig
import scipy.linalg as splinalg

import ancsim.signal.matrices as mat



def autocorrelation(sig, max_lag, interval):
    """
    Returns the autocorrelation of a multichannel signal

    corr[j,k,i] = r_jk(i) = E[s_j(n)s_k(n-i)]
    output is for positive indices, i=0,...,max_lag-1
    for negative correlation values for r_jk, use r_kj instead
    Because r_jk(i) = r_kj(-i)
    """
    #Check if the correlation is normalized
    num_channels = sig.shape[0]
    corr = np.zeros((num_channels, num_channels, max_lag))

    for i in range(num_channels):
        for j in range(num_channels):
            corr[i,j,:] = spsig.correlate(np.flip(sig[i,interval[0]-max_lag+1:interval[1]]), 
                                            np.flip(sig[j,interval[0]:interval[1]]), "valid")
    return corr

def corr_matrix_from_autocorrelation(corr):
    """The correlation is of shape (num_channels, num_channels, max_lag)
        corr[i,j,k] = E[x_i(n)x_j(n-k)], for k = 0,...,max_lag-1
        That means corr[i,j,k] == corr[j,i,-k]

        The output is a correlation matrix of shape (num_channels*max_lag, num_channels*max_lag)
        Each block R_ij is defined as
        [r_ij(0) ... r_ij(max_lag-1)  ]
        [                             ]
        [r_ij(-max_lag+1) ... r_ij(0) ]

        which is part of the full matrix as 
        [R_11 ... R_1M  ]
        [               ]
        [R_M1 ... R_MM  ]

        So that the full R = E[X(n) X^T(n)], 
        where X(n) = [X^T_1(n), ... , X^T_M]^T
        and X_i = [x_i(n), ..., x_i(n-max_lag)]^T
    """
    #num_channels = corr.shape[0]
    #assert num_channels == corr.shape[1]
    #max_lag = corr.shape[-1]
    corr_mat = mat.block_of_toeplitz(np.moveaxis(corr, 0,1), corr)
    return corr_mat
    

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
    """seq1 has shape (num_channels_1, numSamples)
        seq2 has shape (num_channels_2, numSamples)
    
        Outputs a correlation matrix of size 
        (num_channels_1*lag1, num_channels_2*lag2)"""
    assert seq1.ndim == seq2.ndim == 2
    assert seq1.shape[-1] == seq2.shape[-1]
    seqLen = seq1.shape[-1]
    num_channels1 = seq1.shape[0]
    num_channels2 = seq2.shape[0]

    corr = np.zeros((num_channels1, num_channels2, 2*seqLen-1))
    for i in range(num_channels1):
        for j in range(num_channels2):
            corr[i,j,:] = spsig.correlate(seq1[i,:], seq2[j,:], mode="full")
    corr /= seqLen
    corrMid = seqLen - 1
    R = np.zeros((seq1.shape[0]*lag1, seq2.shape[0]*lag2))
    for c1 in range(seq1.shape[0]):
        for l1 in range(lag1):
            rowIdx = c1*lag1 + l1
            R[rowIdx,:] = corr[c1, :, corrMid-l1:corrMid-l1+lag2].ravel()
    return R


def cos_similary(vec1, vec2):
    """Computes <vec1, vec2> / (||vec1|| ||vec2||)
        which is cosine of the angle between the two vectors. 
        1 is paralell vectors, 0 is orthogonal, and -1 is opposite directions

        If the arrays have more than 1 axis, it will be flattened.
    """
    assert vec1.shape == vec2.shape
    vec1 = np.ravel(vec1)
    vec2 = np.ravel(vec2)
    ip = vec1.T @ vec2
    norms = np.linalg.norm(vec1) *np.linalg.norm(vec2)
    return ip / norms

def corr_matrix_distance(mat1, mat2):
    """Computes the correlation matrix distance, as defined in:
    Correlation matrix distaince, a meaningful measure for evaluation of 
    non-stationary MIMO channels - Herdin, Czink, Ozcelik, Bonek
    
    0 means that the matrices are equal up to a scaling
    1 means that they are maximally different (orthogonal in NxN dimensional space)
    """
    assert mat1.shape == mat2.shape
    norm1 = np.linalg.norm(mat1, ord="fro", axis=(-2,-1))
    norm2 = np.linalg.norm(mat2, ord="fro", axis=(-2,-1))
    return 1 - np.trace(mat1 @ mat2) / (norm1 * norm2)