import numpy as np


def fftWithTranspose(timeSignal, n=None, addEmptyDim=False):
    freqSignal = np.fft.fft(timeSignal, n=n, axis=-1)

    newAxisOrder = np.concatenate(([timeSignal.ndim-1], np.arange(timeSignal.ndim-1)))
    freqSignal = np.transpose(freqSignal, newAxisOrder)
    if addEmptyDim:
        freqSignal = np.expand_dims(freqSignal,-1)
    return freqSignal

def ifftWithTranspose(freqSignal, removeEmptyDim=False):
    if removeEmptyDim:
        freqSignal = np.squeeze(freqSignal,axis=-1)
    timeSignal = np.fft.ifft(freqSignal, axis=0)
    newAxisOrder = np.concatenate((np.arange(1,freqSignal.ndim), [0]))
    timeSignal = np.transpose(timeSignal, newAxisOrder)
    return timeSignal

def correlateSumTT(timeFilter, timeSignal):
    """signal is two blocks long signal
        filter is one block FIR unpadded
        last dimension is time, the dimension to FFT over
        next to last dimension is summed over. Broadcasting applies"""
    assert(2*timeFilter.shape[-1] == timeSignal.shape[-1])
   # assert(timeFilter.shape[-2] == timeSignal.shape[-2])
    freqFilter = fftWithTranspose(np.concatenate((np.zeros_like(timeFilter), timeFilter), axis=-1), addEmptyDim=False)
    return correlateSumFT(freqFilter, timeSignal)

def correlateSumFT(freqFilter, timeSignal):
    """signal is two blocks long signal
        filter is FFT of one block FIR filter, padded with zeros BEFORE the ir
        last dimension of filter, next to last of signal is summed over"""
    freqSignal = fftWithTranspose(timeSignal, addEmptyDim=False)
    return correlateSumFF(freqFilter, freqSignal)

def correlateSumFF(freqFilter, freqSignal):
    """signal is FFT of two blocks worth of signal
        filter is FFT of one block FIR filter, padded with zeros BEFORE the ir
        first dimension is frequencies, the dimension to IFFT over
        last dimension is summed over"""
    assert(freqFilter.shape[0] == freqSignal.shape[0])
    assert(freqFilter.shape[0] % 2 == 0)
    outputLen = freqFilter.shape[0] // 2
    filteredSignal = ifftWithTranspose(np.sum(freqFilter * freqSignal.conj(),axis=-1))
    return np.real(filteredSignal[...,:outputLen])

def correlateEuclidianTT(timeFilter, timeSignal):
    """signal is two blocks long signal
        filter is one block FIR unpadded
        Convolves every channel of signal with every channel of the filter
        Outputs (filter.shape[:-1], signal.shape[:-1], numSamples)"""
    assert(2*timeFilter.shape[-1] == timeSignal.shape[-1])
    freqFilter = fftWithTranspose(np.concatenate((np.zeros_like(timeFilter), timeFilter), axis=-1), addEmptyDim=False)
    return correlateEuclidianFT(freqFilter, timeSignal)

def correlateEuclidianFT(freqFilter, timeSignal):
    """signal is two blocks long signal
        filter is FFT of one block FIR filter, padded with zeros BEFORE the ir"""
    freqSignal = fftWithTranspose(timeSignal, addEmptyDim=False)
    return correlateEuclidianFF(freqFilter, freqSignal)

def correlateEuclidianFF(freqFilter, freqSignal):
    """signal is two blocks long signal
        filter is FFT of one block FIR filter, padded with zeros BEFORE the ir
        last dimension of filter, next to last of signal is summed over"""
    assert(freqFilter.shape[0] % 2 == 0)
    outputLen = freqFilter.shape[0] // 2

    filteredSignal = freqFilter.reshape(freqFilter.shape + (1,)*(freqSignal.ndim-1)) * \
                    freqSignal.reshape(freqSignal.shape[0:1] + (1,)*(freqFilter.ndim-1) + freqSignal.shape[1:]).conj()
    filteredSignal = ifftWithTranspose(filteredSignal)
    return np.real(filteredSignal[...,:outputLen])

def convolveSum(freqFilter, timeSignal):
    """The freq domain filter should correspond to a time domain filter 
    padded with zeros after its impulse response
    The signal should be 2 blocks worth of signal. """
    assert(freqFilter.shape[-1] == timeSignal.shape[-2])
    assert (freqFilter.shape[0] == timeSignal.shape[-1])
    assert(freqFilter.shape[0] % 2 == 0)
    outputLen = freqFilter.shape[0] // 2

    # print("shapes")
    # print(freqFilter.shape)
    # print(timeSignal.shape)

    filteredSignal = freqFilter @ fftWithTranspose(timeSignal, addEmptyDim=True)
    filteredSignal = ifftWithTranspose(filteredSignal, removeEmptyDim=True)
    return np.real(filteredSignal[...,outputLen:])

def convolveEuclidianFF(freqFilter, freqSignal):
    """Convolves every channel of input with every channel of the filter
    Outputs (filter.shape, signal.shape, numSamples)
    Signals must be properly padded"""
    assert(freqFilter.shape[0] % 2 == 0)
    outputLen = freqFilter.shape[0] // 2

    filteredSignal = freqFilter.reshape(freqFilter.shape + (1,)*(freqSignal.ndim-1)) * \
                    freqSignal.reshape(freqSignal.shape[0:1] + (1,)*(freqFilter.ndim-1) + freqSignal.shape[1:])
    filteredSignal = ifftWithTranspose(filteredSignal)
    return np.real(filteredSignal[...,outputLen:])
    
def convolveEuclidianFT(freqFilter, timeSignal):
    """Convolves every channel of input with every channel of the filter
        Outputs (filter.shape[:-1], signal.shape[:-1], numSamples)"""
    assert (freqFilter.shape[0] == timeSignal.shape[-1])

    freqSignal = fftWithTranspose(timeSignal, addEmptyDim=False)
    return convolveEuclidianFF(freqFilter, freqSignal)
