import numpy as np
import scipy.linalg as linalg
import scipy.signal as spsig
from abc import abstractmethod

import ancsim.adaptivefilter.base as base
import ancsim.signal.filterclasses as fc
import ancsim.adaptivefilter.diagnostics as diag

class SoundzoneFIR(base.AudioProcessor):
    def __init__(self, sim_info, arrays, blockSize, src, ctrlFiltLen, corrAverageLen):
        super().__init__(sim_info, arrays, blockSize)
        self.name = "Abstract Soundzone Processor"
        self.src = src
        self.ctrlFiltLen = ctrlFiltLen
        self.corrAverageLen = corrAverageLen

        assert self.src.numChannels == 1
        
        self.pathsBright = self.arrays.paths["speaker"]["mic_bright"]
        self.pathsDark = self.arrays.paths["speaker"]["mic_dark"]
        self.numSpeaker = self.arrays["speaker"].num
        self.numMicBright = self.arrays["mic_bright"].num
        self.numMicDark = self.arrays["mic_dark"].num

        self.createNewBuffer("orig_sig", 1)
        self.createNewBuffer("error_bright", self.numMicBright)
        self.createNewBuffer("desired", self.numMicBright)

        self.diag.addNewDiagnostic("bright_zone_power", diag.SignalPower(self.sim_info, "mic_bright", blockSize=self.blockSize))
        self.diag.addNewDiagnostic("dark_zone_power", diag.SignalPower(self.sim_info, "mic_dark", blockSize=self.blockSize))
        self.diag.addNewDiagnostic("acoustic_contrast", diag.SignalPowerRatio(self.sim_info, "mic_bright", "mic_dark", blockSize=self.blockSize))
        self.diag.addNewDiagnostic("bright_zone_error", diag.SignalPowerRatio(self.sim_info, "error_bright", "desired", blockSize=self.blockSize))

        self.desiredFilter = fc.createFilter(ir=self.pathsBright, broadcastDim=1, sumOverInput=False)

    @abstractmethod
    def calcControlFilter(self):
        pass

    def process(self, numSamples):
        x = self.src.getSamples(self.blockSize)
        self.sig["orig_sig"][:,self.idx:self.idx+self.blockSize] = x
        self.sig["speaker"][:,self.idx:self.idx+self.blockSize] = self.controlFilter.process(x)

        self.sig["desired"][:,self.idx:self.idx+self.blockSize] = np.sum(np.squeeze(self.desiredFilter.process(x), axis=-2),axis=0)
            
        self.sig["error_bright"][:,self.idx-self.blockSize:self.idx] = self.sig["mic_bright"][:,self.idx-self.blockSize:self.idx] - \
                                                                        self.sig["desired"][:,self.idx-self.blockSize:self.idx]
            

class PressureMatching(SoundzoneFIR):
    def __init__(self, sim_info, arrays, blockSize, src, ctrlFiltLen, corrAverageLen):
        super().__init__(sim_info, arrays, blockSize, src, ctrlFiltLen, corrAverageLen)
        self.name = "Pressure Matching"
        self.controlFilter = fc.createFilter(ir=self.calcControlFilter())

    def calcControlFilter(self):
        pathLen = np.max((self.pathsBright.shape[-1],self.pathsDark.shape[-1]))
        totSamples = pathLen + self.ctrlFiltLen + self.corrAverageLen
        x = self.src.getSamples(totSamples)
        brightPathFilt = fc.createFilter(ir=self.pathsBright, broadcastDim=1, sumOverInput=False)
        xb = np.squeeze(brightPathFilt.process(x), axis=-2)
        xb = xb[...,pathLen:]
        darkPathFilt = fc.createFilter(ir=self.pathsDark, broadcastDim=1, sumOverInput=False)
        xd = np.squeeze(darkPathFilt.process(x), axis=-2)
        xd = xd[...,pathLen:]

        print("Rb")
        Rb = self.spatialCorrMatrix(xb)
        print("Rd")
        Rd = self.spatialCorrMatrix(xd)
        R = Rb + Rd + 1e-2*np.eye(self.numSpeaker*self.ctrlFiltLen)

        print("rb")
        d = np.sum(xb, axis=0)
        rb = self.spatialCorrVector(xb, d)
        print("solve")
        optFilt = linalg.solve(R, rb, assume_a="sym")
        optFilt = optFilt.reshape((1, self.numSpeaker,-1))
        return optFilt


    def spatialCorrVector3(self, filtSig, desired):
        numMics = filtSig.shape[-2]
        corrStart = filtSig.shape[-1] - 1
        corrEnd = corrStart + self.ctrlFiltLen
        r = np.zeros((self.numSpeaker, self.ctrlFiltLen))
        for m in range(numMics):
            for l in range(self.numSpeaker):
                r[l,:] += spsig.correlate(desired[m,:], filtSig[l,m,:])[corrStart:corrEnd]
                #filtSig should lag behind desired
        r = r / self.corrAverageLen
        return r.ravel()

    def spatialCorrMatrix3(self, filtSig):
        numMics = filtSig.shape[-2]
        corrStart = filtSig.shape[-1] - 1
        corrEnd = corrStart + self.ctrlFiltLen
        r = np.zeros((self.numSpeaker, self.numSpeaker, self.ctrlFiltLen))
        for m in range(numMics):
            for l1 in range(self.numSpeaker):
                for l2 in range(self.numSpeaker):
                    r[l1,l2,:] += spsig.correlate(filtSig[l1,m,:], filtSig[l2,m,:])[corrStart:corrEnd]
                    # index on last axis is how many samples l2 is lagging behind l1
        r = r / self.corrAverageLen
            
        R = np.zeros((self.numSpeaker*self.ctrlFiltLen, self.numSpeaker*self.ctrlFiltLen))
        for l1 in range(self.numSpeaker):
            for l2 in range(self.numSpeaker):
                R[l1*self.ctrlFiltLen:(l1+1)*self.ctrlFiltLen, 
                l2*self.ctrlFiltLen:(l2+1)*self.ctrlFiltLen] = \
                    linalg.toeplitz(r[l2,l1,:], r[l1,l2,:])
                    
        return R

    def spatialCorrMatrix2(self, filteredSignal):
        """Conceptually simple, but enormously slow"""
        numMics = filteredSignal.shape[-2]
        R = np.zeros((self.numSpeaker*self.ctrlFiltLen, self.numSpeaker*self.ctrlFiltLen))
        for n in range(self.ctrlFiltLen, self.corrAverageLen+self.ctrlFiltLen):
            print(n)
            for m in range(numMics):
                y = np.ravel(np.fliplr(filteredSignal[:,m,n-self.ctrlFiltLen:n]))
                R += np.expand_dims(y,1) * np.expand_dims(y,0)
        R = R / self.corrAverageLen
        return R
        

    def spatialCorrVector(self, sigMatrix, sigVec):
        rb = np.zeros((self.numSpeaker, self.ctrlFiltLen))
        for j in range(self.ctrlFiltLen):
            rb[:,j] = np.sum(sigVec[None,:,j:j+self.corrAverageLen] * sigMatrix[:,:,:self.corrAverageLen],
                                axis=(-2,-1))
        rb = rb / self.corrAverageLen
        rb = rb.reshape(-1)
        return rb

    def spatialCorrMatrix(self, filteredSignal):
        rxf = np.zeros((self.numSpeaker, self.numSpeaker, self.ctrlFiltLen))
        for j in range(self.ctrlFiltLen):
            rxf[:,:,j] = np.sum(filteredSignal[None,:,:,j:j+self.corrAverageLen] * filteredSignal[:,None,:,:self.corrAverageLen],
                                axis=(-2,-1)) / self.corrAverageLen
            #first axis lags behind second axis

        R = np.zeros((self.numSpeaker*self.ctrlFiltLen, self.numSpeaker*self.ctrlFiltLen))
        for l1 in range(self.numSpeaker):
            for l2 in range(self.numSpeaker):
                R[l1*self.ctrlFiltLen:(l1+1)*self.ctrlFiltLen, 
                   l2*self.ctrlFiltLen:(l2+1)*self.ctrlFiltLen] = linalg.toeplitz(rxf[l1,l2,:], rxf[l2,l1,:])
        return R
                


class VAST(SoundzoneFIR):
    def __init__(self, sim_info, arrays, blockSize, src, ctrlFiltLen, corrAverageLen, rank, mu):
        super().__init__(sim_info, arrays, blockSize, src, ctrlFiltLen, corrAverageLen)
        self.name = "VAST"
        self.rank = rank
        self.mu = mu

        self.matSize = self.numSpeaker*self.ctrlFiltLen
        assert 1 <= rank <= self.matSize
        self.controlFilter = fc.createFilter(ir=self.calcControlFilter())
    
    def calcControlFilter(self):
        pathLen = np.max((self.pathsBright.shape[-1],self.pathsDark.shape[-1]))
        totSamples = pathLen + self.ctrlFiltLen + self.corrAverageLen
        x = self.src.getSamples(totSamples)
        brightPathFilt = fc.createFilter(ir=self.pathsBright, broadcastDim=1, sumOverInput=False)
        xb = np.squeeze(brightPathFilt.process(x), axis=-2)
        xb = xb[...,pathLen:]
        darkPathFilt = fc.createFilter(ir=self.pathsDark, broadcastDim=1, sumOverInput=False)
        xd = np.squeeze(darkPathFilt.process(x), axis=-2)
        xd = xd[...,pathLen:]

        Rb = self.spatialCorrMatrix(xb)
        Rd = self.spatialCorrMatrix(xd)

        print(f"Rb is symmetric: {np.allclose(Rb, Rb.T)}")
        print(f"Rd is symmetric: {np.allclose(Rd, Rd.T)}")
        try:
            np.linalg.cholesky(Rb)
            print(f"Rb is positive definite")
        except np.linalg.LinAlgError:
            print(f"Rb is not positive definite")
        try:
            np.linalg.cholesky(Rd)
            print(f"Rd is positive definite")
        except np.linalg.LinAlgError:
            print(f"Rd is not positive definite")
        
        d = np.sum(xb, axis=0)
        rb = self.spatialCorrVector(xb, d)

        Rd += 1e-2*np.eye(self.matSize)
        eigVal, eigVec = linalg.eigh(Rb, Rd, type=1)
        eigVal = eigVal[-self.rank:]
        eigVec = eigVec[:,-self.rank:]
        #, subset_by_index=(self.matSize-self.rank,self.matSize-1)
        optFilt = eigVec @ np.diag(1 / (eigVal + self.mu*np.ones(self.rank))) @ eigVec.T @ rb
        
        #optFilt = linalg.solve(R, rb, assume_a="sym")
        optFilt = optFilt.reshape((1, self.numSpeaker,-1))
        return optFilt

    def spatialCorrVector(self, sigMatrix, sigVec):
        rb = np.zeros((self.numSpeaker, self.ctrlFiltLen))
        for j in range(self.ctrlFiltLen):
            rb[:,j] = np.sum(sigVec[None,:,j:j+self.corrAverageLen] * sigMatrix[:,:,:self.corrAverageLen],
                                axis=(-2,-1))
        rb = rb / self.corrAverageLen
        rb = rb.reshape(-1)
        return rb

    def spatialCorrMatrix(self, filteredSignal):
        rxf = np.zeros((self.numSpeaker, self.numSpeaker, self.ctrlFiltLen))
        for j in range(self.ctrlFiltLen):
            rxf[:,:,j] = np.sum(filteredSignal[None,:,:,j:j+self.corrAverageLen] * filteredSignal[:,None,:,:self.corrAverageLen],
                                axis=(-2,-1)) / self.corrAverageLen
            #first axis lags behind second axis

        R = np.zeros((self.numSpeaker*self.ctrlFiltLen, self.numSpeaker*self.ctrlFiltLen))
        for l1 in range(self.numSpeaker):
            for l2 in range(self.numSpeaker):
                R[l1*self.ctrlFiltLen:(l1+1)*self.ctrlFiltLen, 
                   l2*self.ctrlFiltLen:(l2+1)*self.ctrlFiltLen] = linalg.toeplitz(rxf[l1,l2,:], rxf[l2,l1,:])
        return R





class KIVAST(SoundzoneFIR):
    def __init__(self, sim_info, arrays, blockSize, src, ctrlFiltLen, corrAverageLen, rank, mu):
        super().__init__(sim_info, arrays, blockSize, src, ctrlFiltLen, corrAverageLen)
        self.name = "KI-VAST"
        self.rank = rank
        self.mu = mu

        self.matSize = self.numSpeaker*self.ctrlFiltLen
        assert 1 <= rank <= self.matSize
        self.controlFilter = fc.createFilter(ir=self.calcControlFilter())


