import numpy as np
from ancsim.signal.filterclasses import FilterSum
from ancsim.adaptivefilter.conventional.base import AdaptiveFilterBase


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
