import numpy as np
import numexpr as ne
import scipy.signal as spsig
import scipy.linalg as splinalg

import ancsim.signal.matrices as mat
import ancsim.signal.filterclasses as fc


def cov_est_oas(sample_cov, n, verbose=False):
    """
    Implements the OAS covariance estimator with shrinkage from
    Shrinkage Algorithms for MMSE Covariance Estimation

    n is the number of sample vectors that the sample covariance is 
    constructed from

    Assumes Gaussian sample distribution
    """
    assert sample_cov.ndim == 2
    assert sample_cov.shape[0] == sample_cov.shape[1]
    p = sample_cov.shape[-1]
    tr_s = np.trace(sample_cov)
    tr_s_square = np.trace(sample_cov @ sample_cov)

    rho_num = ((-1) / p) *  tr_s_square + tr_s**2
    rho_denom = ((n - 1) / p) * (tr_s_square - (tr_s**2 / p))
    rho = rho_num / rho_denom
    rho_oas = min(rho, 1)

    reg_factor = rho_oas * tr_s / p
    cov_est = (1-rho_oas) * sample_cov + reg_factor * np.eye(p)

    if verbose:
        print(f"OAS covariance estimate is {1-rho_oas} * sample_cov + {reg_factor} * I")
    return cov_est



class SampleCorrelation:
    """
    Estimates the correlation matrix between two random vectors 
    v(n) & w(n) by iteratively computing (1/N) sum_{n=0}^{N-1} v(n) w^T(n) 
    The goal is to estimate R = E[v(n) w^T(n)]

    If delay is supplied, it will calculate E[v(n) w(n-delay)^T]

    Only the internal state will be changed by calling update()
    In order to update self.corr_mat, get_corr() must be called
    """
    def __init__(self, forget_factor, size, delay=0):
        """
        size : scalar integer, correlation matrix is size x size. 
                or tuple of length 2, correlation matrix is size
        forget_factor : scalar between 0 and 1. 
            1 is straight averaging, increasing time window
            0 will make matrix only dependent on the last sample
        """
        if not isinstance(size, (list, tuple, np.ndarray)):
            size = (size, size)
        self.corr_mat = np.zeros(size)
        self.avg = fc.MovingAverage(forget_factor, size)
        self._preallocated_update = np.zeros_like(self.avg.state)
        self._old_vec = np.zeros((size[1], 1))
        self.delay = delay
        self._saved_vecs = np.zeros((size[1], delay))
        self.n = 0

    def update(self, vec1, vec2=None):
        """Update the correlation matrix with a new sample vector
            If only one is provided, the autocorrelation is computed
            If two are provided, the cross-correlation is computed
        """
        if vec2 is None:
            vec2 = vec1

        if self.delay > 0:
            idx = self.n % self.delay
            self._old_vec[...] = self._saved_vecs[:,idx:idx+1]
            self._saved_vecs[:,idx:idx+1] = vec2
            vec2 = self._old_vec

        if self.n >= self.delay:
            np.matmul(vec1, vec2.T, out=self._preallocated_update)
            self.avg.update(self._preallocated_update)
        self.n += 1

    def get_corr(self, autocorr=False, est_method="plain", pos_def=False):
        """Returns the correlation matrix but will ensure
        positive semi-definiteness and hermitian-ness. If pos_def=True
        it will even ensure that the matrix is positive definite. 
        
        est_method can be 'oas', 'plain', possibly 'wl' if implemented
        """
        if not autocorr:
            self.corr_mat[...] = self.avg.state
            return self.corr_mat

        if est_method == "plain":
            self.corr_mat[...] = self.avg.state
        elif est_method == "oas":
            self.corr_mat[...] = cov_est_oas(self.avg.state, self.n, verbose=True)
        else:
            raise ValueError("Invalid est_method name")
        
        self.corr_mat = mat.ensure_hermitian(self.corr_mat)
        if pos_def:
            self.corr_mat = mat.ensure_pos_def_adhoc(self.corr_mat, verbose=True)
        else:
            self.corr_mat = mat.ensure_pos_semidef(self.corr_mat)
        return self.corr_mat
        

class Autocorrelation:
    """
    Autocorrelation is defined as r_m1m2(i) = E[s_m1(n) s_m2(n-i)]
    the returned shape is (num_channels, num_channels, max_lag)
    r[m1, m2, i] is positive entries, r_m1m2(i)
    r[m2, m1, i] is negative entries, r_m1m2(-i)
    """
    def __init__(self, forget_factor, max_lag, num_channels):
        """
        size : scalar integer, correlation matrix is size x size. 
        forget_factor : scalar between 0 and 1. 
            1 is straight averaging, increasing time window
            0 will make matrix only dependent on the last sample
        """
        self.forget_factor = forget_factor
        self.max_lag = max_lag
        self.num_channels = num_channels

        self.corr = fc.MovingAverage(forget_factor, (num_channels, num_channels, max_lag))
        self.corr_mat = np.zeros((self.num_channels*self.max_lag, self.num_channels*self.max_lag))

        self._buffer = np.zeros((self.num_channels, self.max_lag-1))

    def update(self, sig):
        num_samples = sig.shape[-1]
        if num_samples == 0:
            return
        padded_sig = np.concatenate((self._buffer, sig), axis=-1)
        self.corr.update(autocorr(padded_sig, self.max_lag), count_as_updates=num_samples)

        self._buffer[...] = padded_sig[:,sig.shape[-1]:]

    def get_corr_mat(self, max_lag=None, new_first=True, pos_def=False):
        """
        shorthands: L=max_lag, M=num_channels, r(i)=r_m1m2(i)

        R = E[s(n) s(n)^T] =    [R_11 ... R_1M  ]
                                [               ]
                                [               ]
                                [R_M1 ... R_MM  ]

        R_m1m2 = E[s_m1(n) s_m2(n)^T]

        == With new_first==True the vector is defined as
        s_m(n) = [s_m(n) s_m(n-1) ... s_m(n-max_lag+1)]^T

        R_m1m2 =    [r(0)   r(1) ... r(L-1) ]
                    [r(-1)                  ]
                    [                  r(1) ]
                    [r(-L+1) ... r(-1) r(0) ]

        == With new_first==False, the vector is defined as 
        s_m(n) = [s_m(n-max_lag+1) s_m(n-max_lag+2) ... s_m(n)]^T

        R_m1m2 =    [r(0) r(-1) ...   r(-L+1) ]
                    [r(1)                     ]
                    [                  r(-1)  ]
                    [r(L-1) ...   r(1) r(0)   ]
        """
        if max_lag is None:
            max_lag = self.max_lag
        if new_first:
            self.corr_mat = corr_matrix_from_autocorrelation(self.corr.state)
        else:
            self.corr_mat = mat.block_transpose(corr_matrix_from_autocorrelation(self.corr.state), max_lag)

        self.corr_mat = mat.ensure_hermitian(self.corr_mat)
        if pos_def:
            self.corr_mat = mat.ensure_pos_def_adhoc(self.corr_mat, verbose=True)
        else:
            self.corr_mat = mat.ensure_pos_semidef(self.corr_mat)
        return self.corr_mat






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
    corr_mat = mat.block_of_toeplitz(np.moveaxis(corr, 0,1), corr)
    return corr_mat
    

def corr_matrix(seq1, seq2, lag1, lag2):
    """Computes the cross-correlation matrix from two signals. See definition
        in Autocorrelation class or corr_matrix_from_autocorrelation.
    
        seq1 has shape (sumCh, numCh1, numSamples) or (numCh1, numSamples)
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

def is_autocorr_mat(ac_mat, verbose=False):
    assert ac_mat.ndim == 2
    square = ac_mat.shape[0] == ac_mat.shape[1]
    herm = mat.is_hermitian(ac_mat)
    psd = mat.is_pos_semidef(ac_mat)

    if verbose:
        print(f"Is square: {square}")
        print(f"Is hermitian: {herm}")
        print(f"Is positive semidefinite: {psd}")

    return all((square, herm, psd))

def is_autocorr_func(func, verbose=False):
    """
    An autocorrelation function should be of shape
    (num_channels, num_channels, max_lag)
    """
    raise NotImplementedError
    assert func.ndim == 3
    equal_channels = func.shape[0] == func.shape[1]
    symmetric = False

    if verbose:
        print(f"Has equal number of channels: {equal_channels}")
    
    return all((equal_channels,))

def _func_is_symmetric(func):
    raise NotImplementedError
    assert func.ndim == 3
    assert func.shape[0] == func.shape[1]
    num_ch = func.shape[0]
    for i in range(num_ch):
        for j in range(num_ch):
            np.allclose(func[i,j,:])



def cos_similary(vec1, vec2):
    """Computes <vec1, vec2> / (||vec1|| ||vec2||)
        which is cosine of the angle between the two vectors. 
        1 is paralell vectors, 0 is orthogonal, and -1 is opposite directions

        If the arrays have more than 1 axis, it will be flattened.
    """
    assert vec1.shape == vec2.shape
    vec1 = np.ravel(vec1)
    vec2 = np.ravel(vec2)
    norms = np.linalg.norm(vec1) *np.linalg.norm(vec2)
    if norms == 0:
        return np.nan
    ip = vec1.T @ vec2
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
    if norm1 * norm2 == 0:
        return np.nan
    return 1 - np.trace(mat1 @ mat2) / (norm1 * norm2)









def autocorrelation(sig, max_lag, interval):
    """
    I'm not sure I trust this one. Write some unit tests 
    against Autocorrelation class first. But the corr_matrix function
    works, so this shouldn't be needed. 

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
    # corr /= interval[1] - interval[0] #+ max_lag - 1
    return corr




# These should be working correctly, but are written only for testing purposes.
# Might be removed at any time and moved to test module. 

def autocorr(sig, max_lag, normalize=True):
    num_channels = sig.shape[0]
    padded_len = sig.shape[-1]
    num_samples = padded_len - max_lag + 1
    r = np.zeros((num_channels, num_channels, max_lag))
    for ch1 in range(num_channels):
        for ch2 in range(num_channels):
            for i in range(max_lag-1, padded_len):
                r[ch1, ch2, :] += sig[ch1,i] * np.flip(sig[ch2, i-max_lag+1:i+1])
    if normalize:
        r /= num_samples
    return r

def autocorr_ref_spsig(sig, max_lag):
    num_channels = sig.shape[0]
    num_samples = sig.shape[-1]
    sig = np.pad(sig, ((0,0), (max_lag-1, 0)))
    r = np.zeros((num_channels, num_channels, max_lag))
    for ch1 in range(num_channels):
        for ch2 in range(num_channels):
                r[ch1, ch2, :] = spsig.correlate(np.flip(sig[ch1,:]), 
                                            np.flip(sig[ch2,max_lag-1:]), "valid")
    r /= num_samples
    return r


def autocorr_ref(sig, max_lag):
    num_channels = sig.shape[0]
    num_samples = sig.shape[-1]
    padded_len = num_samples + max_lag - 1
    sig = np.pad(sig, ((0,0), (max_lag-1, 0)))
    r = np.zeros((num_channels, num_channels, max_lag))
    for ch1 in range(num_channels):
        for ch2 in range(num_channels):
            for i in range(max_lag-1, padded_len):
                r[ch1, ch2, :] += sig[ch1,i] * np.flip(sig[ch2, i-max_lag+1:i+1])
    r /= num_samples
    return r

def corr_mat_new_first_ref(sig, max_lag):
    num_channels = sig.shape[0]
    num_samples = sig.shape[-1]
    padded_len = num_samples + max_lag - 1
    sig = np.pad(sig, ((0,0), (max_lag-1, 0)))
    R = np.zeros((max_lag*num_channels, max_lag*num_channels))
    for i in range(max_lag-1, padded_len):
        vec = np.flip(sig[:,i-max_lag+1:i+1], axis=-1).reshape(-1,1)
        R+= vec @ vec.T
    R /= num_samples
    return R

def corr_mat_old_first_ref(sig, max_lag):
    num_channels = sig.shape[0]
    num_samples = sig.shape[-1]
    padded_len = num_samples + max_lag - 1
    sig = np.pad(sig, ((0,0), (max_lag-1, 0)))
    R = np.zeros((max_lag*num_channels, max_lag*num_channels))
    for i in range(max_lag-1, padded_len):
        vec = sig[:,i-max_lag+1:i+1].reshape(-1,1)
        R+= vec @ vec.T
    R /= num_samples
    return R