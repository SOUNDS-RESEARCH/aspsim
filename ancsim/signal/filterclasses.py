import numpy as np
import itertools as it

import ancsim.signal.freqdomainfiltering as fdf
import scipy.signal as spsig
import numba as nb



def createFilter(ir=None, numIn=None, numOut=None, irLen=None, broadcastDim=None, sumOverInput=True, dynamic=False):
    if numIn is not None and numOut is not None and irLen is not None:
        assert ir is None
        ir = np.zeros((numIn, numOut, irLen))

    if ir is not None:
        #assert numIn is None and numOut is None and irLen is None
        if broadcastDim is not None:
            if dynamic:
                raise NotImplementedError
            if sumOverInput:
                raise NotImplementedError
            assert ir.ndim == 3 and isinstance(broadcastDim, int) # A fallback to non-numba MD-filter can be added instead of assert
            return FilterMD(broadcastDim, ir)
        
        if sumOverInput:
            if dynamic:
                return FilterSumDynamic(ir)
            return FilterSum(ir)
        return FilterIndividualInputs(ir)



spec_filtersum = [
    ("ir", nb.float64[:,:,:]),
    ("numIn", nb.int32),
    ("numOut", nb.int32),
    ("irLen", nb.int32),
    ("buffer", nb.float64[:,:]),
]
@nb.experimental.jitclass(spec_filtersum)
class FilterSum:
    """# The ir is a 3D array, where each entry in the first two dimensions is an impulse response
        Ex. a (3,2,5) filter has 6 (3x2) IRs each of length 5
        First dimension sets the number of inputs
        Second dimension sets the number of outputs
        (3,None) in, (2,None) out"""
    def __init__(self, ir):
        self.ir = ir
        self.numIn = ir.shape[0]
        self.numOut = ir.shape[1]
        self.irLen = ir.shape[2]
        self.buffer = np.zeros((self.numIn, self.irLen - 1))

    def process(self, dataToFilter):
        numSamples = dataToFilter.shape[-1]
        bufferedInput = np.concatenate((self.buffer, dataToFilter), axis=-1)

        filtered = np.zeros((self.numOut, numSamples))
        for outIdx in range(self.numOut):
            for i in range(numSamples):
                filtered[outIdx,i] += np.sum(self.ir[:,outIdx,:] * np.fliplr(bufferedInput[:,i:self.irLen+i]))

        self.buffer[:, :] = bufferedInput[:, bufferedInput.shape[-1] - self.irLen + 1 :]
        return filtered


spec_filtermd = [
    ("ir", nb.float64[:,:,:]),
    ("irDim1", nb.int32),
    ("irDim2", nb.int32),
    ("irLen", nb.int32),
    ("dataDims", nb.int32),
    ("buffer", nb.float64[:,:]),
]
@nb.experimental.jitclass(spec_filtermd)
class FilterMD:
    """Filter that filters each channel of the input signal
        through each channel of the impulse response.
        input signal is shape (dataDims, numSamples)
        impulse response is shape (dim1, dim2, irLen)
        output signal is  (dataDims, dim1, dim2, numSamples)"""
    def __init__(self, dataDims, ir):
        self.ir = ir
        self.irDim1 = ir.shape[0]
        self.irDim2 = ir.shape[1]
        self.irLen = ir.shape[-1]
        self.dataDims = dataDims

        #self.outputDims = self.irDims + self.dataDims
        #self.numDataDims = len(self.dataDims)
        #self.numIrDims = len(self.irDims)
        self.buffer = np.zeros((self.dataDims, self.irLen - 1))

    def process(self, dataToFilter):
        #inDims = dataToFilter.shape[0:-1]
        numSamples = dataToFilter.shape[-1]

        bufferedInput = np.concatenate((self.buffer, dataToFilter), axis=-1)
        filtered = np.zeros((self.irDim1, self.irDim2, self.dataDims, numSamples))

        for i in range(numSamples):
            filtered[:, :, :,i] = np.sum(np.expand_dims(self.ir,2) * \
                    np.expand_dims(np.expand_dims(np.fliplr(bufferedInput[:,i:self.irLen+i]),0),0),axis=-1)

        self.buffer[:, :] = bufferedInput[:, -self.irLen + 1:]
        return filtered


class FilterSumDynamic:
    """# Identical in use to FilterSum, but uses a buffered output instead of input
        for a more correct filtering when the IR changes while processing. """
    def __init__(self, ir):
        self.ir = ir
        self.numIn = ir.shape[0]
        self.numOut = ir.shape[1]
        self.irLen = ir.shape[2]
        self.buffer = np.zeros((self.numIn, self.irLen - 1))
        self.out_buffer = np.zeros((self.numOut, self.irLen - 1))

    def process(self, data_to_filter):
        num_samples = data_to_filter.shape[-1]
        filtered = np.zeros((self.numOut, num_samples+self.irLen-1))
        filtered[:,:self.irLen-1] = self.out_buffer
        #bufferedInput = np.concatenate((self.buffer, dataToFilter), axis=-1)
        #for out_idx in range(self.numOut):
        for i in range(num_samples):
            filtered[:,i:i+self.irLen] += np.sum(self.ir * data_to_filter[:,None,i:i+1], axis=0)

        self.out_buffer = filtered[:, num_samples:]
        return filtered[:,:num_samples]



class FilterIndividualInputs:
    """Acts as the FilterSum filter before the sum,
    with each input channel having an indiviudal set of IRs.
    IR should be (inputChannels, outputChannels, irLength)
    Input to process method is (inputChannels, numSamples)
    output of process method is (inputChannels, outputChannels, numSamples)"""

    def __init__(self, ir=None, irLen=None, numIn=None, numOut=None):
        if ir is not None:
            self.ir = ir
            self.numIn = ir.shape[0]
            self.numOut = ir.shape[1]
            self.irLen = ir.shape[2]
        elif (irLen is not None) and (numIn is not None) and (numOut is not None):
            self.ir = np.zeros((numIn, numOut, irLen))
            self.irLen = irLen
            self.numIn = numIn
            self.numOut = numOut
        else:
            raise Exception("Not enough constructor arguments")
        self.buffer = np.zeros((self.numIn, self.irLen - 1))

    def process(self, dataToFilter):
        numSamples = dataToFilter.shape[-1]
        bufferedInput = np.concatenate((self.buffer, dataToFilter), axis=-1)

        filtered = np.zeros((self.numIn, self.numOut, numSamples))
        for inIdx, outIdx in it.product(range(self.numIn), range(self.numOut)):
            filtered[inIdx, outIdx, :] = spsig.convolve(
                self.ir[inIdx, outIdx, :], bufferedInput[inIdx, :], "valid"
            )

        self.buffer[:, :] = bufferedInput[:, bufferedInput.shape[-1] - self.irLen + 1 :]
        return filtered

    def setIR(self, irNew):
        if irNew.shape != self.ir.shape:
            self.numIn = irNew.shape[0]
            self.numOut = irNew.shape[1]
            self.irLen = irNew.shape[2]
            self.buffer = np.zeros((self.numIn, self.irLen - 1))
        self.ir = irNew


class FilterMD_IntBuffer:
    """Multidimensional filter with internal buffer
    Will filter every dimension of the input with every impulse response
    (a0,a1,...,an,None) filter with dataDim (b0,...,bn) will give (a0,...,an,b0,...,bn,None) output
    (a,b,None) filter with (x,None) input will give (a,b,x,None) output"""

    def __init__(self, dataDims, ir=None, irLen=None, irDims=None):
        if ir is not None:
            self.ir = ir
            self.irLen = ir.shape[-1]
            self.irDims = ir.shape[0:-1]
        elif (irLen is not None) and (irDims is not None):
            self.ir = np.zeros(irDims + (irLen,))
            self.irLen = irLen
            self.irDims = irDims
        else:
            raise Exception("Not enough constructor arguments")

        if isinstance(dataDims, int):
            self.dataDims = (dataDims,)
        else:
            self.dataDims = dataDims

        self.outputDims = self.irDims + self.dataDims
        self.numDataDims = len(self.dataDims)
        self.numIrDims = len(self.irDims)
        self.buffer = np.zeros(self.dataDims + (self.irLen - 1,))

    def process(self, dataToFilter):
        inDims = dataToFilter.shape[0:-1]
        numSamples = dataToFilter.shape[-1]

        bufferedInput = np.concatenate((self.buffer, dataToFilter), axis=-1)
        filtered = np.zeros(self.irDims + self.dataDims + (numSamples,))

        for idxs in it.product(*[range(d) for d in self.outputDims]):
            # filtered[idxs+(slice(None),)] = np.convolve(self.ir[idxs[0:self.numIrDims]+(slice(None),)],
            #                                             bufferedInput[idxs[-self.numDataDims:]+(slice(None),)], "valid")
            filtered[idxs + (slice(None),)] = spsig.convolve(
                self.ir[idxs[0 : self.numIrDims] + (slice(None),)],
                bufferedInput[idxs[-self.numDataDims :] + (slice(None),)],
                "valid",
            )

        self.buffer[..., :] = bufferedInput[..., -self.irLen + 1 :]
        return filtered

    def setIR(self, irNew):
        if irNew.shape != self.ir.shape:
            self.irDims = irNew.shape[0:-1]
            self.irLen = irNew.shape[-1]
            self.numIrDims = len(self.irDims)
            self.buffer = np.zeros(self.dataDims + (self.irLen - 1,))
        self.ir = irNew




















class FilterMD_Freqdomain:
    """ir is the time domain impulse response, with shape (a0, a1,..., an, irLen)
    tf is frequency domain transfer function, with shape (2*irLen, a0, a1,..., an),

    input of filter function is shape (b0, b1, ..., bn, irLen)
    output of filter function is shape (a0,...,an,b0,...,bn,irLen)

    dataDims must be provided, which is a tuple like (b0, b1, ..., bn)
    filtDim is (a0, a1,..., an) and must then be complemented with irLen or numFreq
    If you give to many arguments, it will propritize tf -> ir -> numFreq -> irLen"""

    def __init__(
        self, dataDims, tf=None, ir=None, filtDim=None, irLen=None, numFreq=None
    ):
        # assert(tf or ir or (numIn and numOut and irLen) is not None)
        if tf is not None:
            assert tf.shape[0] % 2 == 0
            self.tf = tf
        elif ir is not None:
            self.tf = fdf.fftWithTranspose(
                np.concatenate((ir, np.zeros_like(ir)), axis=-1)
            )
        elif filtDim is not None:
            if numFreq is not None:
                self.tf = np.zeros((numFreq, *filtDim), dtype=np.complex128)
            elif irLen is not None:
                self.tf = np.zeros((2 * irLen, *filtDim), dtype=np.complex128)
        else:
            raise ValueError("Arguments missing for valid initialization")

        if isinstance(dataDims, int):
            self.dataDims = (dataDims,)
        else:
            self.dataDims = dataDims

        self.irLen = self.tf.shape[0] // 2
        self.numFreq = self.tf.shape[0]
        self.filtDim = self.tf.shape[1:]

        self.buffer = np.zeros((*self.dataDims, self.irLen))

    def process(self, samplesToProcess):
        assert samplesToProcess.shape == (*self.dataDims, self.irLen)
        outputSamples = fdf.convolveEuclidianFT(
            self.tf, np.concatenate((self.buffer, samplesToProcess), axis=-1)
        )

        self.buffer[...] = samplesToProcess
        return outputSamples

    def processFreq(self, freqsToProcess):
        """Can be used if the fft of the input signal is already available.
        Assumes padding is already applied correctly."""
        assert freqsToProcess.shape == (self.numFreq, *self.dataDims)
        outputSamples = fdf.convolveEuclidianFF(self.tf, freqsToProcess)

        self.buffer[...] = fdf.ifftWithTranspose(freqsToProcess, removeEmptyDim=False)[
            ..., self.irLen :
        ]
        return outputSamples

    def setFilter(self, tfNew):
        raise NotImplementedError
        if tfNew.shape != self.tf.shape:
            self.irLen = self.tf.shape[0] // 2
            self.numFreq = self.tf.shape[0]
            self.filtDim = self.tf.shape[1:]
            self.buffer = np.zeros((self.dataDims, self.irLen))
        self.ir = irNew


class FilterSum_Freqdomain:
    """- ir is the time domain impulse response, with shape (numIn, numOut, irLen)
    - tf is frequency domain transfer function, with shape (2*irLen, numOut, numIn),
    - it is also possible to only provide the dimensions of the filter, numIn and numOut,
        together with either number of frequencies or ir length, where 2*irLen==numFreq
    - dataDims is extra dimensions of the data to broadcast over. Input should then be
        with shape (*dataDims, numIn, numSamples), output will be (*dataDims, numOut, numSamples)

    If you give to many arguments, it will propritize tf -> ir -> numFreq -> irLen"""

    def __init__(
        self,
        tf=None,
        ir=None,
        numIn=None,
        numOut=None,
        irLen=None,
        numFreq=None,
        dataDims=None,
    ):
        if tf is not None:
            assert all(arg is None for arg in [ir, numIn, numOut, irLen, numFreq])
            assert tf.ndim == 3
            assert tf.shape[0] % 2 == 0
            self.tf = tf
        elif ir is not None:
            assert all(arg is None for arg in [tf, numIn, numOut])
            if irLen is not None:
                assert numFreq is None
                totLen = irLen*2
            elif numFreq is not None:
                assert irLen is None
                totLen = numFreq
            else:
                totLen = 2*ir.shape[-1]

            self.tf = np.transpose(
                np.fft.fft(np.concatenate((ir, np.zeros(ir.shape[:-1]+(totLen-ir.shape[-1],))), axis=-1), axis=-1),
                (2, 1, 0),
            )

            # self.tf = np.transpose(
            #     np.fft.fft(np.concatenate((ir, np.zeros_like(ir)), axis=-1), axis=-1),
            #     (2, 1, 0),
            # )
        elif numIn is not None and numOut is not None:
            assert all(arg is None for arg in [tf, ir])
            if numFreq is not None:
                self.tf = np.zeros((numFreq, numOut, numIn), dtype=np.complex128)
            elif irLen is not None:
                self.tf = np.zeros((2 * irLen, numOut, numIn), dtype=np.complex128)
        else:
            raise ValueError("Arguments missing for valid initialization")

        self.irLen = self.tf.shape[0] // 2
        self.numFreq = self.tf.shape[0]
        self.numOut = self.tf.shape[1]
        self.numIn = self.tf.shape[2]

        if dataDims is not None:
            if isinstance(dataDims, int):
                self.dataDims = (dataDims,)
            else:
                self.dataDims = dataDims
            self.lenDataDims = len(self.dataDims)
            self.buffer = np.zeros((*self.dataDims, self.numIn, self.irLen))
        else:
            self.buffer = np.zeros((self.numIn, self.irLen))
            self.lenDataDims = 0

    def process(self, samplesToProcess):
        assert samplesToProcess.shape == self.buffer.shape
        freqsToProcess = fdf.fftWithTranspose(
            np.concatenate((self.buffer, samplesToProcess), axis=-1), addEmptyDim=True
        )
        tfNewShape = self.tf.shape[0:1] + (1,) * self.lenDataDims + self.tf.shape[1:]
        outputSamples = fdf.ifftWithTranspose(
            self.tf.reshape(tfNewShape) @ freqsToProcess, removeEmptyDim=True
        )

        self.buffer[...] = samplesToProcess
        return np.real(outputSamples[..., self.irLen :])

    def processFreq(self, freqsToProcess):
        """Can be used if the fft of the input signal is already available.
        Assumes padding is already applied correctly."""
        assert freqsToProcess.shape == (self.numFreq, self.numIn, 1)
        tfNewShape = self.tf.shape[0:1] + (1,) * self.lenDataDims + self.tf.shape[1:]
        outputSamples = fdf.ifftWithTranspose(
            self.tf.reshape(tfNewShape) @ freqsToProcess, removeEmptyDim=True
        )

        self.buffer[...] = fdf.ifftWithTranspose(freqsToProcess, removeEmptyDim=True)[
            ..., self.irLen :
        ]
        return np.real(outputSamples[..., self.irLen :])

    def processWithoutSum(self, samplesToProcess):
        assert samplesToProcess.shape == self.buffer.shape
        freqsToProcess = fdf.fftWithTranspose(
            np.concatenate((self.buffer, samplesToProcess), axis=-1), addEmptyDim=True)
        tfNewShape = self.tf.shape[0:1] + (1,) * self.lenDataDims + self.tf.shape[1:]
        outputSamples = fdf.ifftWithTranspose(
            np.swapaxes(self.tf.reshape(tfNewShape),-1,-2) * freqsToProcess)

        self.buffer[...] = samplesToProcess
        return np.real(outputSamples[..., self.irLen :])

    def processEuclidian(self, samplesToProcess):
        """Will filter every channel of the input with every channel of the filter
            outputs shape (filt0)"""
        #assert dataDims is not None
        raise NotImplementedError

    def setFilter(self, tfNew):
        raise NotImplementedError
        if tfNew.shape != self.tf.shape:
            self.irLen = tfNew.shape[0] // 2
            self.numFreq = tfNew.shape[0]
            self.numOut = tfNew.shape[1]
            self.numIn = tfNew.shape[2]
            self.buffer = np.zeros_like(self.buffer)
        self.ir = irNew

    def setIR(self, irNew):
        assert irNew.shape == (self.numIn, self.numOut, self.irLen)
        self.tf = np.transpose(
                np.fft.fft(np.concatenate((irNew, np.zeros_like(irNew)), axis=-1), axis=-1),
                (2, 1, 0),
            )

