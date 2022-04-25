import numpy as np
from abc import ABC, abstractmethod

from ancsim.signal.filterclasses import FilterSum, FilterSum_Freqdomain, SinglePoleLowPass, MovingAverage
import ancsim.signal.freqdomainfiltering as fdf


class AdaptiveFilterBase(ABC):
    def __init__(self, ir_len, num_in, num_out, filter_type=None):
        filter_dim = (num_in, num_out, ir_len)
        self.num_in = num_in
        self.num_out = num_out
        self.ir_len = ir_len

        if filter_type is not None:
            self.filt = filter_type(ir=np.zeros(filter_dim))
        else:
            self.filt = FilterSum(ir=np.zeros(filter_dim))

        self.x = np.zeros((self.num_in, self.ir_len, 1))  # Column vector

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

    def set_ir(self, new_ir):
        if new_ir.shape != self.filt.ir.shape:
            self.num_in = new_ir.shape[0]
            self.num_out = new_ir.shape[1]
            self.ir_len = new_ir.shape[2]
        self.filt.set_ir(new_ir)



class AdaptiveFilterFreq(ABC):
    def __init__(self, numFreq, num_in, num_out):
        assert numFreq % 2 == 0
        self.num_in = num_in
        self.num_out = num_out
        self.numFreq = numFreq

        self.filt = FilterSum_Freqdomain(numFreq=numFreq, num_in=num_in, num_out=num_out)

    @abstractmethod
    def update(self):
        pass

    def process(self, signalToProcess):
        return self.filt.process(signalToProcess)
    
    def processWithoutSum(self, signalToProcess):
        return self.filt.processWithoutSum(signalToProcess)


# class LMS(AdaptiveFilterBase):
#     """Dimension of filter is (input channels, output channels, IR length)"""

#     def __init__(self, ir_len, num_in, num_out, stepSize, filter_type=None):
#         super().__init__(ir_len, num_in, num_out, filter_type)
#         self.mu = stepSize

#     def update(self, ref, error):
#         """Inputs should be of the shape (channels, numSamples)"""
#         assert ref.shape[-1] == error.shape[-1]
#         numSamples = ref.shape[-1]

#         for n in range(numSamples):
#             self.insertInSignal(ref[:, n : n + 1])
#             self.filt.ir += self.mu * np.squeeze(
#                 error[None, :, None, n : n + 1] * self.x[:, None, :, :], axis=-1
#             )


class NLMS(AdaptiveFilterBase):
    """Sample by sample processing, although it accepts block inputs
    Dimension of filter is (input channels, output channels, IR length)"""

    def __init__(
        self,
        ir_len,
        num_in,
        num_out,
        stepSize,
        regularization=1e-4,
        filter_type=None,
        channelIndependentNorm=True,
    ):
        super().__init__(ir_len, num_in, num_out, filter_type)
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
        self, ir_len, num_in, num_out, stepSize, regularization=1e-3, filter_type=None
    ):
        super().__init__(ir_len, num_in, num_out, filter_type)
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
        num_in,
        num_out,
        stepSize,
        regularization=1e-3,
        powerEstForgetFactor=0.99,
        normalization="scalar",
    ):
        super().__init__(2 * blockSize, num_in, num_out)
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
        assert ref.shape == (self.num_in, 2 * self.blockSize)
        assert error.shape == (self.num_out, self.blockSize)

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
        num_in,
        num_out,
        stepSize,
        weightMatrix,
        regularization=1e-3,
        freqIndepNorm=False,
    ):
        super().__init__(
            blockSize, num_in, num_out, stepSize, regularization, freqIndepNorm
        )
        self.weightMatrix = weightMatrix

    def update(self, ref, error):
        assert ref.shape == (self.num_in, 2 * self.blockSize)
        assert error.shape == (self.num_out, self.blockSize)

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

    def __init__(self, ir_len, num_in, num_out, forget_factor, signal_power_est):
        super().__init__(ir_len, num_in, num_out)
        self.flat_ir_len = self.ir_len * self.num_in

        self.forget_factor = forget_factor
        self.forget_inv = 1 - forget_factor
        self.signal_power_est = signal_power_est

        # self.inv_corr = np.zeros((1, self.flat_ir_len, self.flat_ir_len))
        # self.inv_corr[0, :, :] = np.eye(self.flat_ir_len) * signal_power_est
        self.inv_corr = np.zeros((self.flat_ir_len, self.flat_ir_len))
        self.inv_corr[:, :] = np.eye(self.flat_ir_len) * signal_power_est
        self.gain = np.zeros((self.num_in*self.ir_len, 1))
        #self.crossCorr = np.zeros((self.num_out, self.flat_ir_len, 1))

    def update_old(self, ref, desired):
        """Inputs should be of the shape (channels, numSamples)"""
        assert ref.shape[-1] == desired.shape[-1]
        numSamples = ref.shape[-1]

        for n in range(numSamples):
            self.insertInSignal(ref[:, n : n + 1])

            X = self.x.reshape((1, -1, 1))
            x_times_corr = self.inv_corr @ X
            g = x_times_corr @ np.transpose(x_times_corr, (0, 2, 1))
            denom = self.forget * np.transpose(X, (0, 2, 1)) @ x_times_corr
            self.inv_corr = self.forget_inv * (self.inv_corr - g / denom)

            self.crossCorr *= self.forget
            self.crossCorr += desired[:, n, None, None] * X
            newFilt = np.transpose(self.inv_corr @ self.crossCorr, (0, 2, 1))
            self.filt.ir = np.transpose(
                newFilt.reshape((self.num_out, self.num_in, self.ir_len)), (1, 0, 2)
            )
    def update(self, ref, desired):
        assert ref.shape[-1] == desired.shape[-1]
        num_samples = ref.shape[-1]
        for i in range(num_samples):
            mic_est = self.filt.process(ref[:,i:i+1])
            error = desired[:,i:i+1] - mic_est

            self.insertInSignal(ref[:, i:i+1])
            X = self.x.reshape((-1, 1))
            #X = np.flip(self.sig["ls"][:,i-self.rir_len+1:i+1],axis=-1).reshape(-1,1)
            x_times_corr = self.inv_corr @ X

            self.gain = x_times_corr / (self.forget_factor + X.T @ x_times_corr)
            self.inv_corr = (self.inv_corr - self.gain @ x_times_corr.T) / self.forget_factor
            self.filt.ir += self.gain.reshape(self.num_in, 1, self.ir_len) * error[None,:,:]



class RLS_singleRef(AdaptiveFilterBase):
    """Dimension of filter is (input channels, output channels, IR length)
    Only works for input channels == 1.
    If more is needed, stack the vectors into one instead"""

    def __init__(self, ir_len, num_in, num_out, forgetting_factor=0.999, signal_power_est=10):
        assert num_in == 1
        super().__init__(ir_len, num_in, num_out)

        self.forget = forgetting_factor
        self.forget_inv = 1 / forgetting_factor
        self.signal_power_est = signal_power_est

        self.invRefCorr = np.zeros((1, self.ir_len, self.ir_len))
        self.invRefCorr[:, :, :] = np.eye(self.ir_len) * signal_power_est
        self.crossCorr = np.zeros((self.num_out, self.ir_len, 1))

    def update(self, ref, desired):
        """Inputs should be of the shape (channels, numSamples)"""
        assert ref.shape[-1] == desired.shape[-1]
        numSamples = ref.shape[-1]

        for n in range(numSamples):
            self.insertInSignal(ref[:, n : n + 1])

            tempMat = self.invRefCorr @ self.x
            g = tempMat @ np.transpose(tempMat, (0, 2, 1))
            denom = self.forget + np.transpose(self.x, (0, 2, 1)) @ tempMat
            self.invRefCorr = self.forget_inv * (self.invRefCorr - g / denom)

            self.crossCorr *= self.forget
            self.crossCorr += desired[:, n, None, None] * self.x
            self.filt.ir = np.transpose(self.invRefCorr @ self.crossCorr, (2, 0, 1))


class RLS_1d_only(AdaptiveFilterBase):
    """Dimension of filter is (input channels, output channels, IR length)"""

    def __init__(self, filter_dim, forgetting_factor=0.999, signal_power_est=10):
        super().__init__(filter_dim)
        assert filter_dim[0] == filter_dim[1] == 1
        self.forget = forgetting_factor
        self.forget_inv = 1 / forgetting_factor
        self.signal_power_est = signal_power_est

        self.invRefCorr = np.zeros((1, self.ir_len, self.ir_len))
        self.invRefCorr[:, :, :] = np.eye(self.ir_len) * signal_power_est
        self.crossCorr = np.zeros((1, self.ir_len, 1))

    def update(self, ref, desired):
        assert ref.shape[-1] == desired.shape[-1]
        numSamples = ref.shape[-1]

        for n in range(numSamples):
            self.insertInSignal(ref[:, n : n + 1])

            tempMat = self.invRefCorr @ self.x
            g = tempMat @ np.transpose(tempMat, (0, 2, 1))
            denom = self.forget + np.transpose(self.x, (0, 2, 1)) @ tempMat
            self.invRefCorr = self.forget_inv * (self.invRefCorr - g / denom)

            self.crossCorr *= self.forget
            self.crossCorr += desired[:, n] * self.x
            self.filt.ir = np.transpose(self.invRefCorr @ self.crossCorr, (0, 2, 1))
