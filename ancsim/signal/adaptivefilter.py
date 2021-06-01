import numpy as np
from abc import ABC, abstractmethod

from ancsim.signal.filterclasses import FilterSum, FilterSum_Freqdomain, SinglePoleLowPass, MovingAverage
import ancsim.signal.freqdomainfiltering as fdf


class AdaptiveFilterBase(ABC):
    def __init__(self, irLen, numIn, numOut, filterType=None):
        filterDim = (numIn, numOut, irLen)
        self.numIn = numIn
        self.numOut = numOut
        self.irLen = irLen

        if filterType is not None:
            self.filt = filterType(ir=np.zeros(filterDim))
        else:
            self.filt = FilterSum(ir=np.zeros(filterDim))

        self.x = np.zeros((self.numIn, self.irLen, 1))  # Column vector

    def insertInSignal(self, ref):
        assert ref.shape[-1] == 1
        self.x = np.roll(self.x, 1, axis=1)
        self.x[:, 0:1, 0] = ref

    def prepare(self):
        pass

    @abstractmethod
    def update(self):
        pass

    def process(self, signalToProcess):
        return self.filt.process(signalToProcess)

    def setIR(self, newIR):
        if newIR.shape != self.filt.ir.shape:
            self.numIn = newIR.shape[0]
            self.numOut = newIR.shape[1]
            self.irLen = newIR.shape[2]
        self.filt.setIR(newIR)



class AdaptiveFilterFreq(ABC):
    def __init__(self, numFreq, numIn, numOut):
        assert numFreq % 2 == 0
        self.numIn = numIn
        self.numOut = numOut
        self.numFreq = numFreq

        self.filt = FilterSum_Freqdomain(numFreq=numFreq, numIn=numIn, numOut=numOut)

    @abstractmethod
    def update(self):
        pass

    def process(self, signalToProcess):
        return self.filt.process(signalToProcess)
    
    def processWithoutSum(self, signalToProcess):
        return self.filt.processWithoutSum(signalToProcess)


class LMS(AdaptiveFilterBase):
    """Dimension of filter is (input channels, output channels, IR length)"""

    def __init__(self, irLen, numIn, numOut, stepSize, filterType=None):
        super().__init__(irLen, numIn, numOut, filterType)
        self.mu = stepSize

    def update(self, ref, error):
        """Inputs should be of the shape (channels, numSamples)"""
        assert ref.shape[-1] == error.shape[-1]
        numSamples = ref.shape[-1]

        for n in range(numSamples):
            self.insertInSignal(ref[:, n : n + 1])
            self.filt.ir += self.mu * np.squeeze(
                error[None, :, None, n : n + 1] * self.x[:, None, :, :], axis=-1
            )


class NLMS(AdaptiveFilterBase):
    """Sample by sample processing, although it accepts block inputs
    Dimension of filter is (input channels, output channels, IR length)"""

    def __init__(
        self,
        irLen,
        numIn,
        numOut,
        stepSize,
        regularization=1e-4,
        filterType=None,
        channelIndependentNorm=True,
    ):
        super().__init__(irLen, numIn, numOut, filterType)
        self.mu = stepSize
        self.beta = regularization
        if channelIndependentNorm:
            self.normFunc = self.channelIndepNorm
        else:
            self.normFunc = self.channelCommonNorm

    def channelIndepNorm(self):
        return 1 / (self.beta + np.transpose(self.x, (0, 2, 1)) @ self.x)

    def channelCommonNorm(self):
        return 1 / (self.beta + np.mean(np.transpose(self.x, (0, 2, 1)) @ self.x))

    def update(self, ref, error):
        """Inputs should be of the shape (channels, numSamples)"""
        assert ref.shape[-1] == error.shape[-1]
        numSamples = ref.shape[-1]

        for n in range(numSamples):
            self.insertInSignal(ref[:, n : n + 1])
            normalization = self.normFunc()
            self.filt.ir += (
                self.mu
                * normalization
                * np.squeeze(
                    error[None, :, None, n : n + 1] * self.x[:, None, :, :], axis=-1
                )
            )


class BlockNLMS(AdaptiveFilterBase):
    """Block based processing and normalization
    Dimension of filter is (input channels, output channels, IR length)
        Normalizes with a scalar, common for all channels
        identical to FastBlockNLMS with scalar normalization"""

    def __init__(
        self, irLen, numIn, numOut, stepSize, regularization=1e-3, filterType=None
    ):
        super().__init__(irLen, numIn, numOut, filterType)
        self.mu = stepSize
        self.beta = regularization

    # def channelIndepNorm(self):
    #     return 1 / (self.beta + np.transpose(self.x,(0,2,1)) @ self.x)

    # def channelCommonNorm(self):
    #     return 1 / (self.beta + np.mean(np.transpose(self.x,(0,2,1)) @ self.x))

    def update(self, ref, error):
        """Inputs should be of the shape (channels, numSamples)"""
        assert ref.shape[-1] == error.shape[-1]
        numSamples = ref.shape[-1]

        grad = np.zeros_like(self.filt.ir)
        for n in range(numSamples):
            self.insertInSignal(ref[:, n : n + 1])
            grad += np.squeeze(
                error[None, :, None, n : n + 1] * self.x[:, None, :, :], axis=-1
            )

        norm = 1 / (np.sum(ref ** 2) + self.beta)
        #print(norm)
        print("Block normalization: ", norm)
        self.filt.ir += self.mu * norm * grad


class FastBlockNLMS(AdaptiveFilterFreq):
    """Identical to BlockNLMS when scalar normalization is used"""
    def __init__(
        self,
        blockSize,
        numIn,
        numOut,
        stepSize,
        regularization=1e-3,
        powerEstForgetFactor=0.99,
        normalization="scalar",
    ):
        super().__init__(2 * blockSize, numIn, numOut)
        self.mu = stepSize
        self.beta = regularization
        self.blockSize = blockSize

        if normalization == "scalar":
            self.normFunc = self.scalarNormalization
        elif normalization == "freqIndependent":
            self.refPowerEstimate = MovingAverage(powerEstForgetFactor, (2 * blockSize, 1, 1))
            self.normFunc = self.freqIndependentNormalization
        elif normalization == "channelIndependent":
            self.normFunc = self.channelIndependentNormalization
        else:
            raise ValueError
            
    def scalarNormalization(self, ref, freqRef):
        return 1 / (np.sum(ref[...,self.blockSize:]**2) + self.beta)

    def freqIndependentNormalization(self, ref, freqRef):
        self.refPowerEstimate.update(np.sum(np.abs(freqRef[:,:,None])**2, axis=(1,2), keepdims=True))
        return 1 / (self.refPowerEstimate.state + self.beta)
    
    def channelIndependentNormalization(self, ref, freqRef):
        return 1 / (np.mean(np.abs(X)**2,axis=0, keepdims=True) + self.beta)


    def update(self, ref, error):
        """ref is two blocks, the latter of which 
            corresponds to the single error block"""
        assert ref.shape == (self.numIn, 2 * self.blockSize)
        assert error.shape == (self.numOut, self.blockSize)

        X = fdf.fftWithTranspose(ref)
        tdGrad = fdf.correlateEuclidianTF(error, X)
        gradient = fdf.fftWithTranspose(np.concatenate((tdGrad, np.zeros_like(tdGrad)),axis=-1))
        norm = self.normFunc(ref, X)
        #print("Fast block normalization: ", norm)
        
        self.filt.tf += self.mu * norm * gradient

    # def process(self, signal):
    #     """"""
    #     assert signal.shape[-1] == self.blockSize
    #     concatBlock = np.concatenate((self.lastBlock, signal),axis=-1)
    #     self.lastBlock[:] = signal
    #     return fdf.convolveSum(self.ir, concatBlock)


class FastBlockWeightedNLMS(FastBlockNLMS):
    def __init__(
        self,
        blockSize,
        numIn,
        numOut,
        stepSize,
        weightMatrix,
        regularization=1e-3,
        freqIndepNorm=False,
    ):
        super().__init__(
            blockSize, numIn, numOut, stepSize, regularization, freqIndepNorm
        )
        self.weightMatrix = weightMatrix

    def update(self, ref, error):
        assert ref.shape == (self.numIn, 2 * self.blockSize)
        assert error.shape == (self.numOut, self.blockSize)

        paddedError = np.concatenate((np.zeros_like(error), error), axis=-1)
        E = np.fft.fft(paddedError, axis=-1).T[:, :, None]
        X = np.fft.fft(ref, axis=-1).T[:, :, None]

        gradient = self.weightMatrix @ E @ np.transpose(X.conj(), (0, 2, 1))
        tdgrad = np.fft.ifft(gradient, axis=0)
        tdgrad[self.blockSize :, :, :] = 0
        gradient = np.fft.fft(tdgrad, axis=0)

        norm = self.normFunc(X)
        self.ir += self.mu * norm * gradient



class RLS(AdaptiveFilterBase):
    """Dimension of filter is (input channels, output channels, IR length)"""

    def __init__(self, irLen, numIn, numOut, forgettingFactor=0.999, signalPowerEst=10):
        super().__init__(irLen, numIn, numOut)
        self.flatIrLen = self.irLen * self.numIn

        self.forget = forgettingFactor
        self.forgetInv = 1 / forgettingFactor
        self.signalPowerEst = signalPowerEst

        self.invRefCorr = np.zeros((1, self.flatIrLen, self.flatIrLen))
        self.invRefCorr[0, :, :] = np.eye(self.flatIrLen) * signalPowerEst
        self.crossCorr = np.zeros((self.numOut, self.flatIrLen, 1))

    def update(self, ref, desired):
        """Inputs should be of the shape (channels, numSamples)"""
        assert ref.shape[-1] == desired.shape[-1]
        numSamples = ref.shape[-1]

        for n in range(numSamples):
            self.insertInSignal(ref[:, n : n + 1])  # )

            X = self.x.reshape((1, -1, 1))
            tempMat = self.invRefCorr @ X
            g = tempMat @ np.transpose(tempMat, (0, 2, 1))
            denom = self.forget + np.transpose(X, (0, 2, 1)) @ tempMat
            self.invRefCorr = self.forgetInv * (self.invRefCorr - g / denom)

            self.crossCorr *= self.forget
            self.crossCorr += desired[:, n, None, None] * X
            newFilt = np.transpose(self.invRefCorr @ self.crossCorr, (0, 2, 1))
            self.filt.ir = np.transpose(
                newFilt.reshape((self.numOut, self.numIn, self.irLen)), (1, 0, 2)
            )


class RLS_singleRef(AdaptiveFilterBase):
    """Dimension of filter is (input channels, output channels, IR length)
    Only works for input channels == 1.
    If more is needed, stack the vectors into one instead"""

    def __init__(self, irLen, numIn, numOut, forgettingFactor=0.999, signalPowerEst=10):
        assert numIn == 1
        super().__init__(irLen, numIn, numOut)

        self.forget = forgettingFactor
        self.forgetInv = 1 / forgettingFactor
        self.signalPowerEst = signalPowerEst

        self.invRefCorr = np.zeros((1, self.irLen, self.irLen))
        self.invRefCorr[:, :, :] = np.eye(self.irLen) * signalPowerEst
        self.crossCorr = np.zeros((self.numOut, self.irLen, 1))

    def update(self, ref, desired):
        """Inputs should be of the shape (channels, numSamples)"""
        assert ref.shape[-1] == desired.shape[-1]
        numSamples = ref.shape[-1]

        for n in range(numSamples):
            self.insertInSignal(ref[:, n : n + 1])

            tempMat = self.invRefCorr @ self.x
            g = tempMat @ np.transpose(tempMat, (0, 2, 1))
            denom = self.forget + np.transpose(self.x, (0, 2, 1)) @ tempMat
            self.invRefCorr = self.forgetInv * (self.invRefCorr - g / denom)

            self.crossCorr *= self.forget
            self.crossCorr += desired[:, n, None, None] * self.x
            self.filt.ir = np.transpose(self.invRefCorr @ self.crossCorr, (2, 0, 1))


class RLS_1d_only(AdaptiveFilterBase):
    """Dimension of filter is (input channels, output channels, IR length)"""

    def __init__(self, filterDim, forgettingFactor=0.999, signalPowerEst=10):
        super().__init__(filterDim)
        assert filterDim[0] == filterDim[1] == 1
        self.forget = forgettingFactor
        self.forgetInv = 1 / forgettingFactor
        self.signalPowerEst = signalPowerEst

        self.invRefCorr = np.zeros((1, self.irLen, self.irLen))
        self.invRefCorr[:, :, :] = np.eye(self.irLen) * signalPowerEst
        self.crossCorr = np.zeros((1, self.irLen, 1))

    def update(self, ref, desired):
        assert ref.shape[-1] == desired.shape[-1]
        numSamples = ref.shape[-1]

        for n in range(numSamples):
            self.insertInSignal(ref[:, n : n + 1])

            tempMat = self.invRefCorr @ self.x
            g = tempMat @ np.transpose(tempMat, (0, 2, 1))
            denom = self.forget + np.transpose(self.x, (0, 2, 1)) @ tempMat
            self.invRefCorr = self.forgetInv * (self.invRefCorr - g / denom)

            self.crossCorr *= self.forget
            self.crossCorr += desired[:, n] * self.x
            self.filt.ir = np.transpose(self.invRefCorr @ self.crossCorr, (0, 2, 1))
