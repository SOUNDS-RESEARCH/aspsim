import numpy as np

import ancsim.settings as s
import ancsim.utilities as util
from ancsim.adaptivefilter.base import AdaptiveFilterFF, AdaptiveFilterFFComplex
from ancsim.signal.filterclasses import FilterMD_IntBuffer, FilterMD_Freqdomain, FilterSum_Freqdomain

import ancsim.signal.freqdomainfiltering as fdf



class FxLMS_FF(AdaptiveFilterFF):
    """Standard FxLMS implementation. Using sample-by-sample normalization
        When in doubt, use FxLMS_FF_Block instead, as it has the bonus
        of being equivalent to frequency domain implementations."""
    def __init__(self, config, mu, beta, speakerRIR):
        super().__init__(config, mu, beta, speakerRIR)
        self.name = "FxLMS Feedforward"

        self.buffers["xf"] = np.zeros((s.NUMREF, s.NUMSPEAKER, s.NUMERROR, s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.secPathXfFilt = FilterMD_IntBuffer((s.NUMREF,), self.secPathFilt.ir)

    def prepare(self):
        self.buffers["xf"][:,:,:,0:self.idx] = np.transpose(self.secPathXfFilt.process(
                                                self.x[:,0:self.idx]), (2,0,1,3))
        
   # @util.measure("MPC update")
    def updateFilter(self):
        self.buffers["xf"][:,:,:,self.updateIdx:self.idx] = np.transpose(self.secPathXfFilt.process(
                                                self.x[:,self.updateIdx:self.idx]), (2,0,1,3))
        grad = np.zeros_like(self.controlFilt.ir)
        for n in range(self.updateIdx, self.idx):
            Xf = np.flip(self.buffers["xf"][:,:,:,n-self.filtLen+1:n+1], axis=-1)
            grad += np.sum(Xf * self.e[None, None,:,n,None],axis=2) / (np.sum(Xf**2) + self.beta)

        self.controlFilt.ir -= grad * self.mu
        self.updateIdx = self.idx
        
        
class FxLMS_FF_Block(AdaptiveFilterFF):
    """Fully block based operation. Can accept changing block sizes
        Is equivalent to FastBlockFxLMS"""
    def __init__(self, config, mu, beta, speakerRIR):
        super().__init__(config, mu, beta, speakerRIR)
        self.name = "FxLMS Feedforward"

        self.buffers["xf"] = np.zeros((s.NUMREF, s.NUMSPEAKER, s.NUMERROR, s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.secPathXfFilt = FilterMD_IntBuffer((s.NUMREF,), self.secPathFilt.ir)

    def prepare(self):
        self.buffers["xf"][:,:,:,0:self.idx] = np.transpose(self.secPathXfFilt.process(
                                                self.x[:,0:self.idx]), (2,0,1,3))
        
    #@util.measure("MPC update")
    def updateFilter(self):
        self.buffers["xf"][:,:,:,self.updateIdx:self.idx] = np.transpose(self.secPathXfFilt.process(
                                                self.x[:,self.updateIdx:self.idx]), (2,0,1,3))

        grad = np.zeros_like(self.controlFilt.ir)
        for n in range(self.updateIdx, self.idx):
            Xf = np.flip(self.buffers["xf"][:,:,:,n-self.filtLen+1:n+1], axis=-1)
            grad += np.sum(Xf * self.e[None, None,:,n,None],axis=2) 
            
        norm = 1 / (np.sum(self.buffers["xf"][:,:,:,self.updateIdx:self.idx]**2) + self.beta)
        self.controlFilt.ir -= grad * self.mu * norm
        self.updateIdx = self.idx


class FastBlockFxLMS(AdaptiveFilterFFComplex):
    """IS EQUIVALENT TO FxLMS_FF_Block
        Requires fixed blocksize during the whole operation.
        Both secondary path and control filter lengths are equal to block size
        Much faster for most system sizes than FxLMS_FF_Block

        Using the exact same normalization as FxLMS_FF_Block, by calculating the power of
        filtered-x in the time domain."""
    def __init__(self, config, mu, beta, speakerRIR, blockSize):
        super().__init__(config, mu,beta, speakerRIR, blockSize)
        assert(speakerRIR["error"].shape[-1] <= blockSize)
        self.name = "Fast Block FxLMS"
        self.updated = True
        self.buffers["xf"] = np.zeros((s.NUMERROR, s.NUMSPEAKER, s.NUMREF,s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.controlFilt = FilterSum_Freqdomain(numIn=s.NUMREF, numOut=s.NUMSPEAKER, irLen=blockSize)
        self.secPathEstimate = FilterMD_Freqdomain(dataDims=s.NUMREF,tf=np.transpose(self.G,(0,2,1)))

    def forwardPassImplement(self, numSamples):
        assert(numSamples == self.blockSize)
        self.y[:,self.idx:self.idx+self.blockSize] = self.controlFilt.process(
                self.x[:,self.idx:self.idx+self.blockSize])
        self.updated = False

    def updateFilter(self):
        assert (self.idx - self.updateIdx == self.blockSize)
        assert(self.updated == False)

        self.buffers["xf"][:,:,:,self.updateIdx:self.idx] = self.secPathEstimate.process(
            self.x[:,self.updateIdx:self.idx]
        )
        
        tdGrad = fdf.correlateSumTT(self.e[None,None,:,self.updateIdx:self.idx], 
                    np.transpose(self.buffers["xf"][:,:,:,self.idx-(2*self.blockSize):self.idx],
                        (1,2,0,3)))

        grad = fdf.fftWithTranspose(np.concatenate(
            (tdGrad, np.zeros_like(tdGrad)), axis=-1))
        norm = 1 / (np.sum(self.buffers["xf"][:,:,:,self.updateIdx:self.idx]**2) + self.beta)
        # powerPerFreq = np.sum(np.abs(xfFreq)**2,axis=(1,2,3))
        # norm = 1 / (np.mean(powerPerFreq) + self.beta)

        self.controlFilt.tf -= self.mu * norm * grad

        self.updateIdx += self.blockSize
        self.updated = True


