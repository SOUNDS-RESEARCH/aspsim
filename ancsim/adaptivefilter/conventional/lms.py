import numpy as np
from ancsim.signal.filterclasses import FilterSum_IntBuffer, SinglePoleLowPass
from ancsim.adaptivefilter.conventional.base import AdaptiveFilterBase, AdaptiveFilterFreqDomain

class LMS(AdaptiveFilterBase):
    """Dimension of filter is (input channels, output channels, IR length)"""
    def __init__(self, irLen, numIn, numOut, stepSize, filterType=None):
        super().__init__(irLen, numIn, numOut, filterType)
        self.mu = stepSize

    def update(self, ref, error):
        """Inputs should be of the shape (channels, numSamples)"""
        assert(ref.shape[-1] == error.shape[-1])
        numSamples = ref.shape[-1]
    
        for n in range(numSamples):
            self.insertInSignal(ref[:,n:n+1])
            self.filt.ir += self.mu * np.squeeze(error[None,:,None,n:n+1] * self.x[:,None,:,:], axis=-1)

class NLMS(AdaptiveFilterBase):
    """Sample by sample processing, although it accepts block inputs
    Dimension of filter is (input channels, output channels, IR length)"""
    def __init__(self, irLen, numIn, numOut, stepSize, regularization=1e-4, 
                    filterType=None, channelIndependentNorm=True):
        super().__init__(irLen, numIn, numOut, filterType)
        self.mu = stepSize
        self.beta = regularization
        if channelIndependentNorm:
            self.normFunc = self.channelIndepNorm
        else:
            self.normFunc = self.channelCommonNorm

    def channelIndepNorm(self):
        return 1 / (self.beta + np.transpose(self.x,(0,2,1)) @ self.x)
    
    def channelCommonNorm(self):
        return 1 / (self.beta + np.mean(np.transpose(self.x,(0,2,1)) @ self.x))

    def update(self, ref, error):
        """Inputs should be of the shape (channels, numSamples)"""
        assert(ref.shape[-1] == error.shape[-1])
        numSamples = ref.shape[-1]
    
        for n in range(numSamples):
            self.insertInSignal(ref[:,n:n+1])
            normalization = self.normFunc()
            self.filt.ir += self.mu * normalization * np.squeeze(error[None,:,None,n:n+1] * self.x[:,None,:,:], axis=-1)

class BlockNLMS(AdaptiveFilterBase):
    """Block based processing and normalization
    Dimension of filter is (input channels, output channels, IR length)
        Normalizes with a scalar, common for all channels"""
    def __init__(self, irLen, numIn, numOut, stepSize, regularization=1e-3, 
                    filterType=None):
        super().__init__(irLen, numIn, numOut, filterType)
        self.mu = stepSize
        self.beta = regularization

    # def channelIndepNorm(self):
    #     return 1 / (self.beta + np.transpose(self.x,(0,2,1)) @ self.x)
    
    # def channelCommonNorm(self):
    #     return 1 / (self.beta + np.mean(np.transpose(self.x,(0,2,1)) @ self.x))

    def update(self, ref, error):
        """Inputs should be of the shape (channels, numSamples)"""
        assert(ref.shape[-1] == error.shape[-1])
        numSamples = ref.shape[-1]
    
        grad = np.zeros_like(self.filt.ir)
        for n in range(numSamples):
            self.insertInSignal(ref[:,n:n+1])
            grad += np.squeeze(error[None,:,None,n:n+1] * self.x[:,None,:,:], axis=-1)
 
        norm = 1 / (np.sum(ref**2) + self.beta)
        self.filt.ir += self.mu * norm * grad

class NLMS_FREQ(AdaptiveFilterFreqDomain):
    def __init__(self, numFreq, numIn, numOut, stepSize, regularization=1e-4):
        super().__init__(numFreq, numIn, numOut)
        self.mu = stepSize
        self.beta = regularization

    def update(self, ref, error):
        """Inputs should be of the shape (numFreq, numChannels, 1)"""
        assert(ref.shape[1] == self.numIn)
        assert(error.shape[1] == self.numOut)
        assert(ref.shape[0] == error.shape[0])
        assert(ref.shape[-1] == error.shape[-1] == 1)

        normalization = 1 / (self.beta + np.transpose(ref.conj(),(0,2,1)) @ ref)
        self.ir += self.mu * normalization * error @ np.transpose(ref.conj(),(0,2,1))
        
        
class FastBlockNLMS(AdaptiveFilterFreqDomain):
    def __init__(self, blockSize, numIn, numOut, stepSize, regularization=1e-3, freqIndepNorm=False):
        super().__init__(2*blockSize, numIn, numOut)
        self.mu = stepSize
        self.beta = regularization
        self.blockSize = blockSize
        if freqIndepNorm:
            self.refPowerEstimate = SinglePoleLowPass(0.9, (2*blockSize,1,1))
            self.normFunc = self.freqIndependentNormalization
        else:
            self.normFunc = self.scalarNormalization

    def scalarNormalization(self, X):
        return 1 / (np.mean(np.transpose(X.conj(),(0,2,1)) @ X) + self.beta)
        
    def freqIndependentNormalization(self, X):
        self.refPowerEstimate.update(np.real(np.transpose(X.conj(),(0,2,1)) @ X))
        return 1 / (self.refPowerEstimate.state + self.beta)

    def update(self, ref, error):
        assert(ref.shape == (self.numIn, 2*self.blockSize))
        assert(error.shape == (self.numOut, self.blockSize))

        paddedError = np.concatenate((np.zeros_like(error), error), axis=-1)
        E = np.fft.fft(paddedError, axis=-1).T[:,:,None]
        X = np.fft.fft(ref, axis=-1).T[:,:,None]

        gradient =  E @ np.transpose(X.conj(),(0,2,1))
        tdgrad = np.fft.ifft(gradient, axis=0)
        tdgrad[self.blockSize:,:,:] = 0
        gradient = np.fft.fft(tdgrad, axis=0)

        norm = self.normFunc(X)
        self.ir += self.mu * norm * gradient

    def process(self, signal):
        assert (signal.shape == (self.numIn, 2*self.blockSize))
        X = np.fft.fft(signal, axis=-1).T[:,:,None]
        output = self.ir @ X
        output = np.squeeze(np.fft.ifft(output,axis=0),axis=-1).T
        output = output[:,self.blockSize:]

        if (np.mean(np.abs(np.imag(output))) > 0.00001):
            print("WARNING: Frequency domain adaptive filter truncates significant imaginary component")
        
        return np.real(output)


class FastBlockWeightedNLMS(FastBlockNLMS):
    def __init__(self, blockSize, numIn, numOut, stepSize, weightMatrix, regularization=1e-3, freqIndepNorm=False):
        super().__init__ (blockSize, numIn, numOut, stepSize, regularization, freqIndepNorm)
        self.weightMatrix = weightMatrix
        
    def update(self, ref, error):
        assert(ref.shape == (self.numIn, 2*self.blockSize))
        assert(error.shape == (self.numOut, self.blockSize))

        paddedError = np.concatenate((np.zeros_like(error), error), axis=-1)
        E = np.fft.fft(paddedError, axis=-1).T[:,:,None]
        X = np.fft.fft(ref, axis=-1).T[:,:,None]

        gradient =  self.weightMatrix @ E @ np.transpose(X.conj(),(0,2,1))
        tdgrad = np.fft.ifft(gradient, axis=0)
        tdgrad[self.blockSize:,:,:] = 0
        gradient = np.fft.fft(tdgrad, axis=0)

        norm = self.normFunc(X)
        self.ir += self.mu * norm * gradient