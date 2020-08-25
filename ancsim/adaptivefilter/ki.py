import numpy as np

from ancsim.adaptivefilter.base import AdaptiveFilterFF
from ancsim.adaptivefilter.mpc import ConstrainedFastBlockFxLMS, UnconstrainedFastBlockFxLMS, FastBlockFxLMS, FxLMS_FF_Block
from ancsim.signal.filterclasses import FilterSum_IntBuffer, FilterSum_Freqdomain,  FilterMD_Freqdomain, Filter_IntBuffer, FilterMD_IntBuffer
import ancsim.signal.freqdomainfiltering as fdf
import ancsim.settings as s
import ancsim.utilities as util
import ancsim.soundfield.kernelinterpolation as ki

def reduceIRLength(ir, tolerance, maxSamplesToReduce=np.inf):
    firstNonZeroIndex = 0
    if tolerance > 0:
        maxVal = np.abs(ir).max()
        for i in range(ir.shape[-1]):
            if np.abs(ir[...,i]).max() / maxVal > tolerance:
                firstNonZeroIndex = i
                break

    samplesToReduce = np.min((maxSamplesToReduce, firstNonZeroIndex))
    reducedIR = ir[...,samplesToReduce:]
    return reducedIR, samplesToReduce


class KIFxLMS(AdaptiveFilterFF):
    """KIFxLMS for IWAENC/ICASSP paper. 
        Coming from kernel derivation 10, or equivalently 12
        
        normalization options is 'xf', 'xfApprox', and 'kixf'"""
    def __init__(self, mu, beta, speakerRIR, errorPos, kiFiltLen, kernelReg, mcPointGen, ipVolume, 
                    mcNumPoints=100, normalization="xf", delayRemovalTolerance=0):
        super().__init__(mu, beta, speakerRIR)
        self.name = "KIFxLMS"

        #GENERATE KERNEL INTERPOLATION FILTER
        kiFilt = ki.getAKernelTimeDomain3d(errorPos, kiFiltLen, kernelReg, mcPointGen, ipVolume, mcNumPoints)

        reducedSecPath, numSamplesRemoved = reduceIRLength(self.secPathFilt.ir, 
                tolerance=delayRemovalTolerance, maxSamplesToReduce=kiFilt.shape[-1]//2)
        print("removed ", numSamplesRemoved," samples")

        self.kiDelay = kiFilt.shape[-1]//2 - numSamplesRemoved
        combinedFilt = np.zeros((*reducedSecPath.shape[0:-1],
                                        reducedSecPath.shape[-1]+kiFilt.shape[-1]-1))
        for i in range(s.NUMERROR):
            for j in range(s.NUMERROR):
                for k in range(reducedSecPath.shape[0]):
                    combinedFilt[k,j,:] += np.convolve(reducedSecPath[k,i,:], kiFilt[i,j,:], "full")

        self.kixfFilt = FilterMD_IntBuffer(dataDims=s.NUMREF, ir=combinedFilt)
        self.buffers["kixf"] = np.zeros((s.NUMREF, s.NUMSPEAKER, s.NUMERROR, s.SIMCHUNKSIZE+s.SIMBUFFER))
        
        #CREATE NORMALIZATION FILTERS:
        if normalization == "xf":
            self.secPathEstimate = FilterMD_IntBuffer(dataDims=s.NUMREF, ir=speakerRIR["error"])
            self.buffers["xf"] = np.zeros((s.NUMSPEAKER, s.NUMERROR, s.NUMREF, s.SIMCHUNKSIZE+s.SIMBUFFER))
            self.normFunc = self.xfNormalization

        elif normalization == "xfApprox":
            self.normFunc = self.xfApproxNormalization
            self.buffers["xfnorm"] = np.zeros((1, s.SIMBUFFER+s.SIMCHUNKSIZE))
            self.secPathNormFilt = Filter_IntBuffer(np.sum(self.secPathFilt.ir**2, axis=(0,1)))
        elif normalization == "kixf":
            self.normFunc = self.kixfNormalization
        else:
            raise ValueError ("Invalid normalization option")
        

        self.metadata["kernelInterpolationDelay"] = int(self.kiDelay)
        self.metadata["samplesRemovedFromSecPath"] = int(numSamplesRemoved)
        self.metadata["delayRemovalTolerance"] = delayRemovalTolerance
        self.metadata["kiFiltLength"] = kiFiltLen
        self.metadata["mcPointGen"] = mcPointGen.__name__
        self.metadata["volume of interpolated region"] = ipVolume
        self.metadata["mcNumPoints"] = mcNumPoints
        
    def kixfNormalization(self):
        """xf in this implementation is x filtered through the combined filter, so it is really kixf"""
        return 1 / (np.sum(self.buffers["kixf"][:,:,:,self.updateIdx-self.kiDelay:self.idx-self.kiDelay]**2) + self.beta)
    
    def xfApproxNormalization(self):
        self.buffers["xfnorm"][:,self.updateIdx:self.idx] = self.secPathNormFilt.process(
            np.sum(self.x[:,self.updateIdx:self.idx]**2,axis=0,keepdims=True)
        )
        
        return 1 / (np.sum(self.buffers["xfnorm"][:,self.updateIdx:self.idx]) + self.beta)
        
    def xfNormalization(self):
        self.buffers["xf"][...,self.updateIdx:self.idx] = \
            self.secPathEstimate.process(self.x[:,self.updateIdx:self.idx])
        
        return 1 / (np.sum(self.buffers["xf"][:,:,:,self.updateIdx:self.idx]**2) + self.beta)
        
    def prepare(self):
        self.buffers["kixf"][:,:,:,0:self.idx] = np.transpose(self.kixfFilt.process(
                                                        self.x[:,0:self.idx]), (2,0,1,3))
        self.buffers["xf"][...,0:self.idx] = self.secPathEstimate.process(self.x[:,0:self.idx])

    def updateFilter(self):
        self.buffers["kixf"][:,:,:,self.updateIdx:self.idx] = np.transpose(self.kixfFilt.process(
                                                        self.x[:,self.updateIdx:self.idx]), (2,0,1,3))
        
        grad = np.zeros_like(self.controlFilt.ir)
        for n in range(self.updateIdx, self.idx):
            Xf = np.flip(self.buffers["kixf"][:,:,:,n-s.FILTLENGTH+1:n+1], axis=-1)
            grad += np.sum(Xf* self.e[None, None,:,n-self.kiDelay,None],axis=2)

        norm = self.normFunc()
        #norm = 1 / (np.sum(self.buffers["xf"][:,:,:,self.updateIdx:self.idx]**2) + self.beta)
        # powerPerSample = np.sum(self.buffers["xf"][:,:,:,self.updateIdx:self.idx]**2, axis=(0,1,2))
        # norm = 1 / (np.mean(powerPerSample) + self.beta)

        self.controlFilt.ir -= grad * self.mu * norm
        self.updateIdx = self.idx



class FastBlockKIFxLMS(FastBlockFxLMS):
    """kernelFilt is the impulse resposnse of kernel interpolation filter A in the time domain
        The filter is attained by monte carlo integration of the frequency domain filter from Ito-sans paper
        and then IFFT-> truncate -> window
        
        It is the frequency domain version of tdkernelderivation 10, or equivalently, 12"""
    def __init__(self, mu, beta, speakerRIR, blockSize, errorPos, kiFiltLen, kernelReg, mcPointGen, ipVolume,mcNumPoints=100, delayRemovalTolerance=0):
        super().__init__(mu,beta, speakerRIR, blockSize)
        self.name = "Fast Block KIFxLMS"
        assert(kiFiltLen % 1 == 0)
        kiFilt = ki.getAKernelTimeDomain3d(errorPos, kiFiltLen, kernelReg, mcPointGen, ipVolume, mcNumPoints)
        
        assert(kiFilt.shape[-1] <= blockSize)
        
        reducedSecPath, numSamplesRemoved = reduceIRLength(np.transpose(self.secPathFilt.ir, (1,0,2)), 
                tolerance=delayRemovalTolerance, maxSamplesToReduce=kiFilt.shape[-1]//2)
        self.secPathEstimate = FilterMD_Freqdomain(dataDims=s.NUMREF,ir=np.concatenate(
                (reducedSecPath, np.zeros((reducedSecPath.shape[:-1] + (numSamplesRemoved,)))), axis=-1))

        self.kiDelay = kiFilt.shape[-1]//2 - numSamplesRemoved
        self.buffers["kixf"] = np.zeros((s.NUMSPEAKER, s.NUMREF,s.NUMERROR,s.SIMCHUNKSIZE+s.SIMBUFFER))

        self.kiFilt = FilterSum_Freqdomain(tf=fdf.fftWithTranspose(kiFilt,n=2*blockSize), 
                                            dataDims=(s.NUMSPEAKER, s.NUMREF))

        self.normFunc = self.xfNormalization
        
        self.metadata["kernelInterpolationDelay"] = int(self.kiDelay)
        self.metadata["samplesRemovedFromSecPath"] = int(numSamplesRemoved)
        self.metadata["delayRemovalTolerance"] = delayRemovalTolerance
        self.metadata["kiFiltLength"] = int(kiFilt.shape[-1])
        self.metadata["mcPointGen"] = mcPointGen.__name__
        self.metadata["mcNumPoints"] = mcNumPoints
        self.metadata["ipVolume"] = ipVolume
        self.metadata["normalization"] = self.normFunc.__name__

    def kixfNormalization(self):
        return 1 / (np.sum(self.buffers["kixf"][:,:,:,self.updateIdx-self.kiDelay:self.idx-self.kiDelay]**2) + self.beta)
    
    def xfNormalization(self):
        return 1 / (np.sum(self.buffers["xf"][:,:,:,self.updateIdx:self.idx]**2) + self.beta)
    
    def updateFilter(self):
        assert (self.idx - self.updateIdx == self.blockSize)
        assert(self.updated == False)

        self.buffers["xf"][:,:,:,self.updateIdx:self.idx] = self.secPathEstimate.process(
            self.x[:,self.updateIdx:self.idx]
        )

        self.buffers["kixf"][:,:,:,self.updateIdx-self.kiDelay:self.idx-self.kiDelay] = \
            self.kiFilt.process(np.transpose(self.buffers["xf"][:,:,:,self.updateIdx:self.idx],(1,2,0,3)))
        
        tdGrad = fdf.correlateSumTT(self.e[None,None,:,self.updateIdx-self.kiDelay:self.idx-self.kiDelay], 
                    self.buffers["kixf"][:,:,:,self.idx-(2*self.blockSize)-self.kiDelay:self.idx-self.kiDelay])

        grad = fdf.fftWithTranspose(np.concatenate(
            (tdGrad, np.zeros_like(tdGrad)), axis=-1))
        norm = self.xfNormalization()
        # powerPerFreq = np.sum(np.abs(xfFreq)**2,axis=(1,2,3))
        # norm = 1 / (np.mean(powerPerFreq) + self.beta)

        self.controlFilt.tf -= self.mu * norm * grad

        self.updateIdx += self.blockSize
        self.updated = True


















class KIFxLMS_old(FxLMS_FF_Block):
    """KIFxLMS for IWAENC/ICASSP paper. 
        Coming from kernel derivation 10, or equivalently 12"""
    def __init__(self, mu, beta, speakerRIR, kiFilt, delayRemovalTolerance=0):
        super().__init__(mu, beta, speakerRIR)
        self.name = "KIFxLMS"

        reducedSecPath, numSamplesRemoved = reduceIRLength(self.secPathFilt.ir, 
                tolerance=delayRemovalTolerance, maxSamplesToReduce=kiFilt.shape[-1]//2)
        print("removed ", numSamplesRemoved," samples")

        self.kiDelay = kiFilt.shape[-1]//2 - numSamplesRemoved
        combinedFilt = np.zeros((*reducedSecPath.shape[0:-1],
                                        reducedSecPath.shape[-1]+kiFilt.shape[-1]-1))
        for i in range(s.NUMERROR):
            for j in range(s.NUMERROR):
                for k in range(reducedSecPath.shape[0]):
                    combinedFilt[k,j,:] += np.convolve(reducedSecPath[k,i,:], kiFilt[i,j,:], "full")

        self.secPathXfFilt.setIR(combinedFilt)

        self.metadata["kernelInterpolationDelay"] = int(self.kiDelay)
        self.metadata["samplesRemovedFromSecPath"] = int(numSamplesRemoved)
        self.metadata["delayRemovalTolerance"] = delayRemovalTolerance
        self.metadata["kiFiltLength"] = int(kiFilt.shape[-1])
        
    # def generateKiFilt(self):
    #     kiFilt = np.transpose(ki.getAKernelTimeDomain3d(sim.pos.error, config["KERNFILTLEN"], config), (1,2,0))

    def prepare(self):
        self.buffers["xf"][:,:,:,0:self.idx] = np.transpose(self.secPathXfFilt.process(
                                                        self.x[:,0:self.idx]), (2,0,1,3))

    def updateFilter(self):
        self.buffers["xf"][:,:,:,self.updateIdx:self.idx] = np.transpose(self.secPathXfFilt.process(
                                                        self.x[:,self.updateIdx:self.idx]), (2,0,1,3))
        
        grad = np.zeros_like(self.controlFilt.ir)
        for n in range(self.updateIdx, self.idx):
            Xf = np.flip(self.buffers["xf"][:,:,:,n-s.FILTLENGTH+1:n+1], axis=-1)
            grad += np.sum(Xf* self.e[None, None,:,n-self.kiDelay,None],axis=2)

        norm = 1 / (np.sum(self.buffers["xf"][:,:,:,self.updateIdx:self.idx]**2) + self.beta)
        # powerPerSample = np.sum(self.buffers["xf"][:,:,:,self.updateIdx:self.idx]**2, axis=(0,1,2))
        # norm = 1 / (np.mean(powerPerSample) + self.beta)

        self.controlFilt.ir -= grad * self.mu * norm
        self.updateIdx = self.idx


class KernelIPFreqCon(ConstrainedFastBlockFxLMS):
    """Frequency domain KIFxLMS used for drafts of IWAENC paper. Not using fully linear convolutions"""
    def __init__(self, mu, beta, speakerRIR, blockSize, kernFilt):
        super().__init__(mu,beta, speakerRIR, blockSize)
        self.A = kernFilt
        self.name = "Kernel IP freq constrained"
        self.combinedFilt = self.G.conj() @ self.A
        #self.normFactor = np.sqrt(np.sum(np.abs(self.G.conj() @ self.A @ np.transpose(self.G,(0,2,1))**2)))
        self.normFactor = np.sqrt(np.sum(np.abs(self.G.conj() @ self.A @ np.transpose(self.G,(0,2,1)))**2))

    #@util.measure("Update Kfreq")
    def updateFilter(self):
        assert (self.idx - self.updateIdx == self.blockSize)
        assert(self.updated == False)
        self.E = np.fft.fft(np.concatenate((np.zeros((s.NUMERROR,self.blockSize)),
                                            self.e[:,self.updateIdx:self.idx]),axis=-1), 
                                            axis=-1).T[:,:,None]
        grad = self.combinedFilt @ self.E @ np.transpose(self.X.conj(),(0,2,1))

        tdGrad = np.fft.ifft(grad, axis=0)
        tdGrad[self.blockSize:,:] = 0
        grad = np.fft.fft(tdGrad, axis=0)
        
        #norm = 1 / ((np.sum(self.G.conj() @ self.A @ np.transpose(self.G,(0,2,1))) * np.sum(np.abs(self.X)**2) / (2*self.blockSize)**2) + self.beta)
        norm = 1 / ((self.normFactor *  \
                    np.sum(np.abs(self.X)**2) / (2*self.blockSize)**2) + self.beta)
        
        self.H -= self.mu * norm * grad

        self.updateIdx += self.blockSize
        self.update = True
        
        
class KernelIPFreqItoUncon(UnconstrainedFastBlockFxLMS):
    """NOT WElL TESTED, USING FLAWED UnconstrainedFastBlockFxLMS"""
    def __init__(self, mu, beta, secPathError,  secPathTarget, secPathEvals,  blockSize, kernFilt):
        super().__init__(mu,beta,secPathError, secPathTarget, secPathEvals, blockSize)
        self.A = kernFilt
        self.name = "Kernel IP freq unconstrained"

    def updateFilter(self):
        assert (self.idx - self.updateIdx == self.blockSize)
        assert(self.updated == False)
        self.E = np.fft.fft(np.concatenate((np.zeros((s.NUMERROR,self.blockSize)),
                                            self.e[:,self.updateIdx:self.idx]),axis=-1), 
                                            axis=-1).T[:,:,None]
        grad = self.G.conj() @ self.A @ self.E @ np.transpose(self.X.conj(),(0,2,1))
        
        # filtNorm = np.sqrt(np.sum(np.abs(self.G.conj() @ self.A @ np.transpose(self.G,(0,2,1))**2), axis=(1,2), keepdims=True))
        # refNorm = np.sum(np.abs(self.X)**2, axis=(1,2), keepdims=True) 
        # norm = 1 / (filtNorm* refNorm + self.beta)
        norm = 1 / ((np.sqrt(np.sum(np.abs(self.G.conj() @ self.A @ np.transpose(self.G,(0,2,1))**2))) * \
                    np.sum(np.abs(self.X)**2) / (2*self.blockSize)**2) + self.beta)
        
        #norm = 1 / (np.sum(np.abs(self.G)**2, axis=(1,2),keepdims=True) * self.refPowerEstimate + self.beta)
        
        self.H -= self.mu * 0.2 * norm * grad

        self.updateIdx += self.blockSize
        self.update = True