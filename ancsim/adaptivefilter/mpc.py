import numpy as np
import scipy.signal.windows as windows
import time

import ancsim.settings as s
import ancsim.utilities as util
from ancsim.adaptivefilter.base import AdaptiveFilterFF, AdaptiveFilterFB, AdaptiveFilterFFComplex
from ancsim.signal.filterclasses import FilterMD_IntBuffer, FilterMD_Freqdomain, FilterSum_Freqdomain

import ancsim.signal.freqdomainfiltering as fdf



class FxLMS_FF(AdaptiveFilterFF):
    """Standard FxLMS implementation. Using sample-by-sample normalization
        When in doubt, use FxLMS_FF_Block instead, as it has the bonus
        of being equivalent to frequency domain implementations."""
    def __init__(self, mu, beta, speakerRIR):
        super().__init__(mu, beta, speakerRIR)
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
            Xf = np.flip(self.buffers["xf"][:,:,:,n-s.FILTLENGTH+1:n+1], axis=-1)
            grad += np.sum(Xf * self.e[None, None,:,n,None],axis=2) / (np.sum(Xf**2) + self.beta)

        self.controlFilt.ir -= grad * self.mu
        self.updateIdx = self.idx
        
        
class FxLMS_FF_Block(AdaptiveFilterFF):
    """Fully block based operation. Can accept changing block sizes
        Is equivalent to FastBlockFxLMS"""
    def __init__(self, mu, beta, speakerRIR):
        super().__init__(mu, beta, speakerRIR)
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
            Xf = np.flip(self.buffers["xf"][:,:,:,n-s.FILTLENGTH+1:n+1], axis=-1)
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
    def __init__(self, mu, beta, speakerRIR, blockSize):
        super().__init__(mu,beta, speakerRIR, blockSize)
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
            (tdGrad, np.zeros_like(tdGrad)), axis=-1), 
            addEmptyDim=False)
        norm = 1 / (np.sum(self.buffers["xf"][:,:,:,self.updateIdx:self.idx]**2) + self.beta)
        # powerPerFreq = np.sum(np.abs(xfFreq)**2,axis=(1,2,3))
        # norm = 1 / (np.mean(powerPerFreq) + self.beta)

        self.controlFilt.tf -= self.mu * norm * grad

        self.updateIdx += self.blockSize
        self.updated = True


class MPC_FB(AdaptiveFilterFB):
    def __init__(self, mu, beta, secPathError, secPathTarget, secPathEvals):
        super().__init__(mu,beta,secPathError, secPathTarget, secPathEvals)
        self.name = "MPC FB"
        
    def updateFilter(self):
        grad = np.zeros_like(self.H)
        for n in range(self.updateIdx, self.idx):
            Xf = np.flip(self.xf[:,:,:,n-s.FILTLENGTH+1:n+1], axis=-1)
            grad += np.sum(Xf * self.e[None, None,:,n,None],axis=2) / (np.sum(Xf**2) + self.beta)
        self.H -= grad * self.mu
        self.updateIdx = self.idx

    def forwardPassImplement(self, numSamples, noiseAtError, noiseAtRef, errorMicNoise):
        for i in range(numSamples):
            n = self.idx + i
            
            Y = np.flip(self.y[:,n-self.secPathError.shape[-1]:n], axis=-1)
            yf = np.sum(Y[:,np.newaxis,:]*self.secPathError, axis=(0,-1))
            self.e[:,n] = np.squeeze(noiseAtError[:,i] + yf + errorMicNoise[:,i])
            
            self.x[:,n] = self.e[:,n] - yf
            
            X = np.flip(self.x[:,n-s.FILTLENGTH+1:n+1], axis=-1)
            self.y[:,n] = np.sum(X[:,None,:]*self.H, axis=(0,-1)) 

        self.xf[:,:,:,self.idx:self.idx+numSamples] = np.transpose(self.secPathXfFilt.process(self.x[:,self.idx:self.idx+numSamples]), (2,0,1,3))


class ConstrainedFastBlockFxLMS(AdaptiveFilterFFComplex):
    """DEPRECATED - USING MISUNDERSTOOD METHOD
        USE FASTBLOCKFXLMS INSTEAD"""
    def __init__(self, mu, beta, speakerRIR, blockSize):
        super().__init__(mu,beta, speakerRIR, blockSize)
        self.name = "MPC FF Frequency Domain"
        self.updated = True
        
    #@util.measure("Kernel freq forward")
    def forwardPassImplement(self, numSamples):
        assert(numSamples == self.blockSize)
        xConcatBlock = self.x[:,self.idx-self.blockSize:self.idx+self.blockSize]
        self.X = np.fft.fft(xConcatBlock,axis=-1).T[:,:,None]
        self.Y = self.H @ self.X 
        
        y = np.fft.ifft(self.Y, axis=0)
        if (np.mean(np.abs(np.imag(y))) > 0.00001):
            print("WARNING: Frequency domain adaptive filter truncates significant imaginary component")

        y = np.squeeze(np.real(y),axis=-1).T
        y = y[:,self.blockSize:]
        self.y[:,self.idx:self.idx+self.blockSize] = y
        self.updated = False

    #@util.measure("old FD")
    def updateFilter(self):
        assert (self.idx - self.updateIdx == self.blockSize)
        assert(self.updated == False)
        self.E = np.fft.fft(np.concatenate((np.zeros((s.NUMERROR,self.blockSize)),
                                            self.e[:,self.updateIdx:self.idx]),axis=-1), 
                                            axis=-1).T[:,:,None]

        grad = self.G.conj() @ self.E @ np.transpose(self.X.conj(),(0,2,1))

        tdGrad = np.fft.ifft(grad, axis=0)
        tdGrad[self.blockSize:,:] = 0
        grad = np.fft.fft(tdGrad, axis=0)
        
        powerPerFreq = np.sum(np.abs(self.G)**2,axis=(1,2)) * np.sum(np.abs(self.X)**2,axis=(1,2)) 
        norm = 1 / (np.mean(powerPerFreq) + self.beta)
        #print("Old FD Norm ", norm)
       # norm = 1 / ((np.sum(np.abs(self.G)**2) * np.sum(np.abs(self.X)**2) / (2*self.blockSize)**2) + self.beta)
        self.H -= self.mu * norm * grad

        self.updateIdx += self.blockSize
        self.updated = True

# class FastBlockFxLMS(AdaptiveFilterFFComplex):
#     def __init__(self, mu, beta, speakerRIR, blockSize):
#         super().__init__(mu,beta, speakerRIR, blockSize)
#         self.name = "MPC FF Frequency Domain meticuluous"
#         self.updated = True
#         self.secPathEstimate = np.transpose(self.G,(0,2,1))
#         self.G = np.transpose(self.G,(0,2,1))

#     #@util.measure("Kernel freq forward")
#     def forwardPassImplement(self, numSamples):
#         assert(numSamples == self.blockSize)
#         xConcatBlock = self.x[:,self.idx-self.blockSize:self.idx+self.blockSize]
#         self.X = np.fft.fft(xConcatBlock,axis=-1).T[:,:,None]
#         self.Y = self.H @ self.X 
        
#         y = np.fft.ifft(self.Y, axis=0)
#         if (np.mean(np.abs(np.imag(y))) > 0.00001):
#             print("WARNING: Frequency domain adaptive filter truncates significant imaginary component")

#         y = np.squeeze(np.real(y),axis=-1).T
#         y = y[:,self.blockSize:]
#         self.y[:,self.idx:self.idx+self.blockSize] = y
#         self.updated = False

#     #@util.measure("new FD")
#     def updateFilter(self):
#         assert (self.idx - self.updateIdx == self.blockSize)
#         assert(self.updated == False)

#         self.X = np.fft.fft(self.x[:,self.idx-(2*self.blockSize):self.idx],axis=-1).T[:,:,None]
#         xf = self.secPathEstimate[:,None,:,:] * self.X[:,:,None,:] 

#         xf = np.fft.ifft(xf, axis=0)
#         xf[:self.blockSize,:,:,:] = 0
#         xf = np.fft.fft(xf,axis=0)
        
#         self.E = np.fft.fft(np.concatenate((np.zeros((s.NUMERROR,self.blockSize)),
#                                             self.e[:,self.updateIdx:self.idx]),axis=-1), 
#                                             axis=-1).T[:,:,None]

#         grad = np.squeeze(np.transpose(xf.conj(),(0,3,1,2)) @ self.E[:,None,:,:], axis=-1)

#         tdGrad = np.fft.ifft(grad, axis=0)
#         tdGrad[self.blockSize:,:,:] = 0
#         grad = np.fft.fft(tdGrad, axis=0)

#         powerPerFreq = np.sum(np.abs(xf)**2,axis=(1,2,3))
#         norm = 1 / (np.mean(powerPerFreq) + self.beta)
#         self.H -= self.mu * norm * grad

#         self.updateIdx += self.blockSize
#         self.updated = True



class UnconstrainedFastBlockFxLMS(ConstrainedFastBlockFxLMS):
    """NOT CORRECTLY IMPLEMENTED. USING MISUNDERSTOOD METHOD"""
    def __init__(self, mu, beta, speakerRIR, blockSize):
        super().__init__(mu,beta, speakerRIR, blockSize)

    def updateFilter(self):
        assert (self.idx - self.updateIdx == self.blockSize)
        assert(self.updated == False)

        self.E = np.fft.fft(np.concatenate((np.zeros((s.NUMERROR,self.blockSize)),
                                    self.e[:,self.updateIdx:self.idx]),axis=-1), 
                                    axis=-1).T[:,:,None]
        grad = self.G.conj() @ self.E @ np.transpose(self.X.conj(),(0,2,1))

        #norm = 1 / (np.sum(np.abs(self.G)**2) * np.sum(np.abs(self.X)**2) + self.beta)
        norm = 1 / ((np.sum(np.abs(self.G)**2) * np.sum(np.abs(self.X)**2) / (2*self.blockSize)**2) + self.beta)
        self.H -= self.mu * norm * grad

        self.updateIdx += self.blockSize
        self.updated = True
