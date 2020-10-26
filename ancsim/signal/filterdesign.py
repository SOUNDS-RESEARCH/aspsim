import numpy as np
import scipy.signal as signal
import ancsim.signal.freqdomainfiltering as fdf
import itertools as it
#=====================================================================

# def combineIRWithSum(ir1, ir2, axisToSum=-2, axisToConvolve=-1):
#     newShape = np.broadcast_arrays(ir1[...,0],ir2[...,0])[0].shape
#     #newShape = np.max(ir1.shape[:-2], ir2.shape[:-2])   #To account for broadcasting
#     np.zeros((np.concatenate(newShape, ir1.shape[-1]+ir2.shape[-2]-1)))


#======================================================================
def fracDelayLagrangeFilter(delay, maxOrder, maxFiltLen):
    """Generates fractional delay filter using lagrange interpolation
        if maxOrder would require a longer filter than maxFiltLen, 
        maxFiltLen will will override and set a lower interpolation order"""
    ir = np.zeros(maxFiltLen)
    fracdly, dly = np.modf(delay)
    order = int(np.min((maxOrder, 2*dly, 2*(maxFiltLen-1-dly))))

    delta = np.floor(order / 2) + fracdly 
    h = lagrangeInterpol(order, delta)
    
    diff = delay - delta
    startIdx = int(np.floor(diff))
    ir[startIdx:startIdx+order+1] = h
    return ir

def lagrangeInterpol(N, delta):
    irLen = int(N+1)
    h = np.zeros(irLen)
    
    for n in range(irLen):
        k = np.arange(irLen)
        k = k[np.arange(irLen)!=n]
        
        h[n] = np.prod((delta - k) / (n - k))
    return h

#=======================================================================

def getFrequencyValues(numFreq, samplerate):
    """Get the frequency values of all positive frequency bins in Hz
        numFreq is the number of frequency bins INCLUDING negative frequencies
        Ex: if numFreq == 10, this function will return 5 values"""
    if numFreq % 2 == 0:
        return ((samplerate / (numFreq)) * np.arange(numFreq//2 + 1))
    elif numFreq % 2 == 1:
        raise NotImplementedError
    else:
        raise ValueError

def insertNegativeFrequencies(freqSignal, even, axis=0):
    """To be used in conjunction with getFrequencyValues
        Inserts all negative frequency values under the 
        assumption of conjugate symmetry: WARNING: ASSUMES ACTUAL SYMMETRY FOR NOW, NOT CONJUGATE SYMMETRY
        Parameter even: boolean indicating if an even or odd number
        of bins is desired. This must correspond to numFreq value 
        set in getFrequencyValues"""
    if even:
        return np.concatenate((freqSignal, np.flip(freqSignal[1:-1,:,:], axis=axis)), axis=axis)
    else:
        raise NotImplementedError

def firFromFreqsWindow(freqFilter, filtLen):
    """Use this over the other window methods, 
        as they might be wrong. freqFilter is with both positive
        and negative frequencies.

        Makes FIR filter from frequency values. 
        Works only for odd impulse response lengths
        Uses hamming window"""
    assert(filtLen % 1 == 0)
    halfLen = int(filtLen / 2)

    #tdA = np.real(np.fft.ifft(A, axis=0))
    timeFilter = np.real(fdf.ifftWithTranspose(freqFilter))
    timeFilter = np.concatenate((timeFilter[...,-halfLen:], 
                                timeFilter[...,:halfLen+1]), axis=-1)
    timeFilter = timeFilter * signal.windows.hamming(filtLen).reshape(
        (1,)*(timeFilter.ndim-1) + timeFilter.shape[-1:])
    return timeFilter



def minTruncatedLength(ir, twosided=True, maxRelTruncError=1e-3):
    """Calculates the minimum length you can truncate a filter to.
        ir has shape (..., irLength)
        if twosided, the ir will be assumed centered in the middle. 
        The filter can be multidimensional, the minimum length will 
        be calculated independently for all impulse responses, and 
        the longest length chosen. The relative error is how much of the 
        power of the impulse response that is lost by truncating. """
    irLen = ir.shape[-1]
    irShape = ir.shape[:-1]

    totalEnergy = np.sum(ir**2, axis=-1)
    energyNeeded = (1-maxRelTruncError) * totalEnergy

    if twosided:
        centerIdx = irLen // 2
        casualSum = np.cumsum(ir[...,centerIdx:]**2, axis=-1)
        noncausalSum = np.cumsum(np.flip(ir[...,:centerIdx]**2, axis=-1), axis=-1)
        energySum = casualSum
        energySum[...,1:] += noncausalSum
    else:
        energySum = np.cumsum(ir**2,axis=-1)
    
    enoughEnergy = energySum > energyNeeded[...,None]
    truncIndices = np.zeros(irShape, dtype=int)
    for indices in it.product(*[range(dimSize) for dimSize in irShape]):
        truncIndices[indices] = np.min(np.nonzero(enoughEnergy[indices+(slice(None),)])[0])

    reqFilterLength = np.max(truncIndices)
    if twosided:
        reqFilterLength = 2*reqFilterLength+1
    return reqFilterLength





























#==============================================================================
#GENERATES TIME DOMAIN FILTER FROM FREQUENCY DOMAIN FILTERS
#SHOULD BE CHECKED OUT/TESTED BEFORE USED. I DONT FULLY TRUST THEM

def tdFilterFromFreq(freqSamples, irLen, posFreqOnly=True, method="window", window="hamming"):
    numFreqs = freqSamples.shape[-1]
    if posFreqOnly:
        if (numFreqs % 2 == 0):
            raise NotImplementedError
            freqSamples = np.concatenate((freqSamples, np.flip(freqSamples, axis=-1)), axis=-1)
        else:
            freqSamples = np.concatenate((freqSamples, np.flip(freqSamples[:,:,1:], axis=-1)), axis=-1)

    if method == "window":
        assert (irLen < freqSamples.shape[-1])
        tdFilter = tdFilterWindow(freqSamples, irLen, window)
    elif method == "minphase":
        assert (irLen < 2*freqSamples.shape[-1]-1)
        tdFilter = tdFilterMinphase(freqSamples, irLen, window)
        
    return tdFilter
    
def tdFilterWindow(freqSamples, irLen, window):
    ir = np.fft.ifft(freqSamples, axis=-1)
    ir = np.concatenate((ir[:,:,-irLen//2+1:], ir[:,:,0:irLen//2+1]), axis=-1)
    
    if (np.abs(np.imag(ir)).max() / np.abs(np.real(ir)).max() < 10**(-7)):
        ir = np.real(ir)
    
    ir *= signal.get_window(window,(irLen), fftbins=False)[None,None,:]
    return ir
    
def tdFilterMinphase(freqSamples, irLen, window):
    ir = np.fft.ifft(freqSamples**2, axis=-1)
    ir = np.concatenate((ir[:,:,-irLen+1:], ir[:,:,0:irLen]), axis=-1)
    ir = ir * (signal.get_window(window,(irLen*2 - 1), fftbins=False)[None,None,:]**2)
    
    if(np.abs(np.imag(ir)).max() / np.abs(np.real(ir)).max() < 10**(-7)):
        ir = np.real(ir)
        
    tdFilter = np.zeros((ir.shape[0], ir.shape[1], irLen))
    for i in range(ir.shape[0]):
        for j in range(ir.shape[1]):
            tdFilter[i,j,:] = signal.minimum_phase(ir[i,j,:])
    return tdFilter

            
#=================================================================================
def testFilterDesignFunc(samplerate):
    pos = setup.getPositionsCircular()
    
    numFreqSamples = 2**12 + 1
    freqs = (samplerate / (2*numFreqSamples)) * np.arange(numFreqSamples)
    freqsamples = ki.kernelInterpolationFR(pos["error"], freqs)
    freqsamples = np.transpose(freqsamples, (1,2,0))    
    
    filt1 = tdFilterFromFreq(freqsamples, 214, method="minphase", window="hamming")
    filt2 = tdFilterFromFreq(freqsamples, 215, method="window", window="hamming")
    
    nRows = 3
    nCols = 3
    fig, axes = plt.subplots(nRows, nCols, figsize = (8,8))
    for i in range(nRows):
        for j in range(nCols):  
            fig.tight_layout(pad=0.5)
            axes[i,j].plot(filt1[i,j,:])
            axes[i,j].plot(filt2[i,j,:])
            axes[i,j].legend(("minphase", "window"))
            axes[i,j].spines['right'].set_visible(False)
            axes[i,j].spines['top'].set_visible(False)
        
    fig, axes = plt.subplots(nRows, nCols, figsize = (8,8))   
    for i in range(nRows):
        for j in range(nCols):  
            fig.tight_layout(pad=0.5)
            axes[i,j].plot(util.mag2db(np.abs(np.fft.fft(filt1[i,j,:], 4096))))
            axes[i,j].plot(util.mag2db(np.abs(np.fft.fft(filt2[i,j,:], 4096))))
            axes[i,j].legend(("minphase", "window"))
            axes[i,j].spines['right'].set_visible(False)
            axes[i,j].spines['top'].set_visible(False)
    plt.show()  
            


if __name__ == "__main__":
    import setupfunctions as setup
    import kernelinterpolation as ki
    import matplotlib.pyplot as plt
    import utilityfunctions as util
    import settings as s
    testFilterDesignFunc()