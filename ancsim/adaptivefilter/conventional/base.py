import numpy as np
from ancsim.signal.filterclasses import FilterSum, FilterSum_Freqdomain
from abc import ABC, abstractmethod


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


class AdaptiveFilterFreqDomain(ABC):
    def __init__(self, numFreq, numIn, numOut):
        # filterDim = (numIn, numOut, irLen)
        self.numIn = numIn
        self.numOut = numOut
        self.numFreq = numFreq

        self.ir = np.zeros((numFreq, numOut, numIn), dtype=np.complex128)

    def prepare(self):
        pass

    @abstractmethod
    def update(self):
        pass

    def process(self, signalToProcess):
        return self.ir @ signalToProcess

    def setIR(self, newIR):
        if newIR.shape != self.ir.shape:
            self.numIn = newIR.shape[2]
            self.numOut = newIR.shape[1]
            self.numFreq = newIR.shape[0]
        self.ir = newIR
