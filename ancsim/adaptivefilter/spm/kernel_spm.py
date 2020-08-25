import numpy as np

from ancsim.adaptivefilter.mpc import ConstrainedFastBlockFxLMS, FastBlockFxLMS
from ancsim.adaptivefilter.ki import FastBlockKIFxLMS
from ancsim.adaptivefilter.spm.auxnoise import FreqAuxNoiseFxLMS2, FastBlockFxLMSSPMEriksson
from ancsim.signal.sources import GoldSequenceSource, WhiteNoiseSource
from ancsim.adaptivefilter.conventional.lms import FastBlockNLMS, FastBlockWeightedNLMS
from ancsim.soundfield.kernelinterpolation import soundfieldInterpolation, soundfieldInterpolationFIR, getAKernelTimeDomain3d
import ancsim.adaptivefilter.diagnostics as dia
import ancsim.signal.freqdomainfiltering as fdf
from ancsim.signal.filterclasses import FilterSum_Freqdomain, FilterMD_Freqdomain, \
    FilterMD_IntBuffer, FilterSum_IntBuffer

import ancsim.settings as s


class FastBlockKIFxLMSSPMEriksson(FastBlockFxLMSSPMEriksson):
    """"""
    def __init__(self, mu, muSPM,speakerFilters, blockSize, kiFilt):
        super().__init__(mu,muSPM, speakerFilters, blockSize)
        self.name = "Kernel IP Freq ANC with Eriksson SPM"
        assert(kiFilt.shape[-1] <= blockSize)
        assert(kiFilt.shape[-1] % 1 == 0)

        self.kiDelay = kiFilt.shape[-1]//2
        self.buffers["kixf"] = np.zeros((s.NUMSPEAKER, s.NUMREF,s.NUMERROR,s.SIMCHUNKSIZE+s.SIMBUFFER))

        self.kiFilt = FilterSum_Freqdomain(tf=fdf.fftWithTranspose(kiFilt,n=2*blockSize,addEmptyDim=False), 
                                            dataDims=(s.NUMSPEAKER, s.NUMREF))

        self.metadata["kernelInterpolationDelay"] = int(self.kiDelay)
        self.metadata["kiFiltLength"] = int(kiFilt.shape[-1])

    def forwardPassImplement(self, numSamples): 
        super().forwardPassImplement(numSamples)

    def updateFilter(self):
        assert (self.idx - self.updateIdx == self.blockSize)
        assert(self.updated == False)
        super().updateSPM()
        
        self.buffers["xf"][:,:,:,self.updateIdx:self.idx] = self.secPathEstimateMD.process(
            self.x[:,self.updateIdx:self.idx]
        )

        self.buffers["kixf"][:,:,:,self.updateIdx-self.kiDelay:self.idx-self.kiDelay] = \
            self.kiFilt.process(np.transpose(self.buffers["xf"][:,:,:,self.updateIdx:self.idx],(1,2,0,3)))
        
        tdGrad = fdf.correlateSumTT(self.buffers["f"][None,None,:,self.updateIdx-self.kiDelay:self.idx-self.kiDelay], 
                    self.buffers["kixf"][:,:,:,self.idx-(2*self.blockSize)-self.kiDelay:self.idx-self.kiDelay])

        grad = fdf.fftWithTranspose(np.concatenate(
            (tdGrad, np.zeros_like(tdGrad)), axis=-1), 
            addEmptyDim=False)
        norm = 1 / (np.sum(self.buffers["kixf"][:,:,:,self.updateIdx-self.kiDelay:self.idx-self.kiDelay]**2) + self.beta)

        self.controlFilt.tf -= self.mu * norm * grad

        self.updateIdx += self.blockSize
        self.updated = True


class KernelSPM6(FastBlockFxLMSSPMEriksson):
    """Very similar to FastBlockKIFxLMSSPMEriksson. Applies the kernel interpolation
        filter to the spm update as well. Given by a straigtforward differentiation of
        the integral cost function int_omega|f(r)|^2dr where f = e - Ghat y_spm"""
    def __init__(self, mu, muSPM,speakerFilters, blockSize, errorPos, kiFiltLen, kernelReg, mcPointGen, ipVolume, 
                    mcNumPoints=100):
        super().__init__(mu,muSPM, speakerFilters, blockSize)
        self.name = "Fast Block FxLMS Eriksson SPM with weighted spm error"
        assert(kiFiltLen <= blockSize)
        assert(kiFiltLen % 1 == 0)
        kiFiltFir = getAKernelTimeDomain3d(errorPos, kiFiltLen, kernelReg, mcPointGen, ipVolume, mcNumPoints)
        self.kiFilt = FilterSum_Freqdomain(tf=fdf.fftWithTranspose(kiFiltFir,n=2*blockSize,addEmptyDim=False), 
                                            dataDims=(s.NUMSPEAKER, s.NUMREF))
        self.kiFiltSecPathEst = FilterSum_Freqdomain(tf=self.kiFilt.tf)


        self.kiDelay = kiFiltFir.shape[-1]//2
        self.buffers["kixf"] = np.zeros((s.NUMSPEAKER, s.NUMREF,s.NUMERROR,s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.buffers["kif"] = np.zeros((s.NUMERROR, s.SIMCHUNKSIZE+s.SIMBUFFER))

        # self.kiFilt = FilterSum_Freqdomain(tf=fdf.fftWithTranspose(kiFilt,n=2*blockSize,addEmptyDim=False), 
        #                                     dataDims=(s.NUMSPEAKER, s.NUMREF))

        self.metadata["kernelInterpolationDelay"] = int(self.kiDelay)
        self.metadata["kiFiltLength"] = int(kiFiltLen)
        self.metadata["mcPointGen"] = mcPointGen.__name__
        self.metadata["mcNumPoints"] = mcNumPoints
        self.metadata["ipVolume"] = ipVolume
        #self.metadata["normalization"] = self.normFunc.__name__

    def forwardPassImplement(self, numSamples): 
        super().forwardPassImplement(numSamples)

    def updateSPM(self):
        vf = self.secPathEstimate.process(self.buffers["v"][:,self.updateIdx:self.idx])

        self.buffers["f"][:,self.updateIdx:self.idx] = self.e[:,self.updateIdx:self.idx] - vf

        self.buffers["kif"][:,self.updateIdx-self.kiDelay:self.idx-self.kiDelay] = \
            self.kiFiltSecPathEst.process(self.buffers["f"][:,self.updateIdx:self.idx])

        grad = fdf.correlateEuclidianTT(self.buffers["kif"][:,self.updateIdx-self.kiDelay:self.idx-self.kiDelay], 
                                self.buffers["v"][:,self.idx-self.kiDelay-(2*self.blockSize):self.idx-self.kiDelay])
        
        grad = fdf.fftWithTranspose(np.concatenate(
                (grad, np.zeros_like(grad)),axis=-1),addEmptyDim=False)
        norm = 1 / (np.sum(self.buffers["v"][:,self.updateIdx-self.kiDelay:self.idx-self.kiDelay]**2) + 1e-3)

        self.secPathEstimate.tf += self.muSPM * norm * grad
        self.secPathEstimateMD.tf[...] = self.secPathEstimate.tf

        self.diag.saveBlockData("secpath", self.idx, self.secPathEstimate.tf)

    def updateFilter(self):
        assert (self.idx - self.updateIdx == self.blockSize)
        assert(self.updated == False)
        self.updateSPM()
        
        self.buffers["xf"][:,:,:,self.updateIdx:self.idx] = self.secPathEstimateMD.process(
            self.x[:,self.updateIdx:self.idx]
        )

        self.buffers["kixf"][:,:,:,self.updateIdx-self.kiDelay:self.idx-self.kiDelay] = \
            self.kiFilt.process(np.transpose(self.buffers["xf"][:,:,:,self.updateIdx:self.idx],(1,2,0,3)))
        
        tdGrad = fdf.correlateSumTT(self.buffers["f"][None,None,:,self.updateIdx-self.kiDelay:self.idx-self.kiDelay], 
                    self.buffers["kixf"][:,:,:,self.idx-(2*self.blockSize)-self.kiDelay:self.idx-self.kiDelay])

        grad = fdf.fftWithTranspose(np.concatenate(
            (tdGrad, np.zeros_like(tdGrad)), axis=-1), 
            addEmptyDim=False)
        norm = 1 / (np.sum(self.buffers["kixf"][:,:,:,self.updateIdx-self.kiDelay:self.idx-self.kiDelay]**2) + self.beta)

        self.controlFilt.tf -= self.mu * norm * grad

        self.updateIdx += self.blockSize
        self.updated = True


class FDKIFxLMSEriksson(ConstrainedFastBlockFxLMS):
    """DEPRECATED - DOES NOT HAVRE LINEAR CONVOLUTIONS"""
    def __init__(self, mu, beta, speakerFilters, blockSize, muSPM, kernFilt):
        super().__init__(mu,beta, speakerFilters, blockSize)
        self.name = "Kernel IP Freq ANC with Eriksson SPM"
        self.A = kernFilt
        
        self.auxNoiseSource = WhiteNoiseSource(power=3, numChannels=s.NUMSPEAKER)
        self.secPathEstimate = FastBlockNLMS(blockSize=blockSize, numIn=s.NUMSPEAKER, numOut=s.NUMERROR, 
                                            stepSize=muSPM, freqIndepNorm=False)

        self.buffers["v"] = np.zeros((s.NUMSPEAKER, s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.buffers["f"] = np.zeros((s.NUMERROR, s.SIMCHUNKSIZE+s.SIMBUFFER))

        self.G = np.transpose(self.G, (0,2,1))
        self.diag.addNewDiagnostic("secpath", dia.ConstantEstimateNMSE(self.G, plotFrequency=s.PLOTFREQUENCY))
        #self.combinedFilt = self.G.conj() @ self.A
        #self.normFactor = np.sqrt(np.sum(np.abs(self.G.conj() @ self.A @ np.transpose(self.G,(0,2,1))**2)))

        self.metadata["auxNoiseSource"] = self.auxNoiseSource.__class__.__name__
        self.metadata["auxNoisePower"] = self.auxNoiseSource.power
        self.metadata["muSPM"] = muSPM

    def prepare(self):
        self.buffers["f"][:,:s.SIMBUFFER] = self.e[:,:s.SIMBUFFER]

    def forwardPassImplement(self, numSamples): 
        super().forwardPassImplement(numSamples)

        v = self.auxNoiseSource.getSamples(numSamples)
        self.buffers["v"][:,self.idx:self.idx+numSamples] = v
        self.y[:,self.idx:self.idx+numSamples] += v

    #@util.measure("Update Kfreq")
    def updateFilter(self):
        assert (self.idx - self.updateIdx == self.blockSize)
        assert(self.updated == False)
        self.updateSPM()
        self.F = np.fft.fft(np.concatenate((np.zeros((s.NUMERROR,self.blockSize)),
                                            self.buffers["f"][:,self.updateIdx:self.idx]),axis=-1), 
                                            axis=-1).T[:,:,None]
        grad = np.transpose(self.secPathEstimate.ir.conj(),(0,2,1)) @ self.A @ self.F @ np.transpose(self.X.conj(),(0,2,1))

        tdGrad = np.fft.ifft(grad, axis=0)
        tdGrad[self.blockSize:,:] = 0
        grad = np.fft.fft(tdGrad, axis=0)
     
        normFactor = np.sqrt(np.sum(np.abs(np.transpose(self.secPathEstimate.ir.conj(),(0,2,1)) @ self.A @ self.secPathEstimate.ir)**2,axis=(1,2)))
        #norm = 1 / ((normFactor *  \
        #            np.sum(np.abs(self.X)**2) / (2*self.blockSize)**2) + self.beta)

        powerPerFreq = normFactor * np.sum(np.abs(self.X)**2,axis=(1,2))
        norm = 1 / (np.mean(powerPerFreq) + self.beta)
        
        self.H -= self.mu * norm * grad

        self.updateIdx += self.blockSize
        self.update = True

    def updateSPM(self):
        vf = self.secPathEstimate.process(np.concatenate((np.zeros((s.NUMSPEAKER,self.blockSize)),
                                                        self.buffers["v"][:,self.updateIdx:self.idx]), axis=-1))

        self.buffers["f"][:,self.updateIdx:self.idx] = self.e[:,self.updateIdx:self.idx] - vf
        self.secPathEstimate.update(self.buffers["v"][:,self.updateIdx-self.blockSize:self.idx], 
                                    self.buffers["f"][:,self.updateIdx:self.idx])
        self.diag.saveBlockData("secpath", self.idx, self.secPathEstimate.ir)


class FDKIFxLMSErikssonInterpolatedF(FDKIFxLMSEriksson):
    def __init__(self, mu, beta, speakerFilters, blockSize, muSPM, kernFilt):
        super().__init__(mu, beta, speakerFilters, blockSize, muSPM, kernFilt)
        self.name = "Kernel IP Freq ANC with Eriksson SPM, Weighted f"

        self.secPathEstimate = FastBlockWeightedNLMS(blockSize=blockSize, numIn=s.NUMSPEAKER, 
                                                    numOut=s.NUMERROR, weightMatrix=kernFilt,
                                                    stepSize=muSPM, freqIndepNorm=False)
        self.diag.addNewDiagnostic("weightedsecpath", dia.ConstantEstimateNMSE(self.A @ self.G, plotFrequency=s.PLOTFREQUENCY))

    def updateSPM(self):
        super().updateSPM()
        self.diag.saveBlockData("weightedsecpath", self.idx, self.secPathEstimate.ir)



class KernelSPM2(ConstrainedFastBlockFxLMS):
    """DOES NOT HAVE LINEAR CONVOLUTIONS"""
    def __init__(self, mu, muSPM, speakerRIR, blockSize, errorPos):
        super().__init__(mu,1e-3, speakerRIR, blockSize)
        self.name = "Freq ANC Kernel SPM 2"
        
        kernelReg = 1e-3
        self.kernelMat = soundfieldInterpolation(errorPos, errorPos, 2*self.blockSize, kernelReg)
        
        self.auxNoiseSource = WhiteNoiseSource(power=3, numChannels=s.NUMSPEAKER)
        self.secPathEstimate = FastBlockWeightedNLMS(blockSize=blockSize, numIn=s.NUMSPEAKER, numOut=s.NUMERROR, 
                                                    stepSize=muSPM, weightMatrix=self.kernelMat, freqIndepNorm=False)
        
    
        self.buffers["v"] = np.zeros((s.NUMSPEAKER, s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.buffers["f"] = np.zeros((s.NUMERROR, s.SIMCHUNKSIZE+s.SIMBUFFER))

        self.G = np.transpose(self.G, (0,2,1))
        self.diag.addNewDiagnostic("secpath", dia.ConstantEstimateNMSE(self.G, plotFrequency=s.PLOTFREQUENCY))
        #self.combinedFilt = self.G.conj() @ self.A
        #self.normFactor = np.sqrt(np.sum(np.abs(self.G.conj() @ self.A @ np.transpose(self.G,(0,2,1))**2)))

        self.metadata["auxNoiseSource"] = self.auxNoiseSource.__class__.__name__
        self.metadata["auxNoisePower"] = self.auxNoiseSource.power
        self.metadata["muSPM"] = muSPM
        self.metadata["kernelReg"] = kernelReg

    def prepare(self):
        self.buffers["f"][:,:s.SIMBUFFER] = self.e[:,:s.SIMBUFFER]

    def forwardPassImplement(self, numSamples): 
        super().forwardPassImplement(numSamples)

        v = self.auxNoiseSource.getSamples(numSamples)
        self.buffers["v"][:,self.idx:self.idx+numSamples] = v
        self.y[:,self.idx:self.idx+numSamples] += v

    def updateFilter(self):
        assert (self.idx - self.updateIdx == self.blockSize)
        assert(self.updated == False)
        self.updateSPM()
        self.F = np.fft.fft(np.concatenate((np.zeros((s.NUMERROR,self.blockSize)),
                                            self.buffers["f"][:,self.updateIdx:self.idx]),axis=-1), 
                                            axis=-1).T[:,:,None]
        grad = np.transpose(self.secPathEstimate.ir.conj(),(0,2,1)) @ self.F @ np.transpose(self.X.conj(),(0,2,1))

        tdGrad = np.fft.ifft(grad, axis=0)
        tdGrad[self.blockSize:,:] = 0
        grad = np.fft.fft(tdGrad, axis=0)
        
        powerPerFreq = np.sum(np.abs(self.secPathEstimate.ir)**2,axis=(1,2)) * np.sum(np.abs(self.X)**2,axis=(1,2))
        norm = 1 / (np.mean(powerPerFreq) + self.beta)
        self.H -= self.mu * norm * grad

        self.updateIdx += self.blockSize
        self.updated = True

    def updateSPM(self):
        V = np.fft.fft(np.concatenate((np.zeros((s.NUMSPEAKER,self.blockSize)),
                        self.buffers["v"][:,self.updateIdx:self.idx]), axis=-1), axis=-1).T[:,:,None]
        Vf = self.kernelMat @ self.secPathEstimate.ir @ V
        vf = np.squeeze(np.fft.ifft(Vf,axis=0),axis=-1).T
        vf = np.real(vf[:,self.blockSize:])
        #vf = self.secPathEstimate.process(np.concatenate((np.zeros((s.NUMSPEAKER,self.blockSize)),
        #                                                self.buffers["v"][:,self.updateIdx:self.idx]), axis=-1))

        self.buffers["f"][:,self.updateIdx:self.idx] = self.e[:,self.updateIdx:self.idx] - vf
        self.secPathEstimate.update(self.buffers["v"][:,self.updateIdx-self.blockSize:self.idx], 
                                    self.buffers["f"][:,self.updateIdx:self.idx])
        self.diag.saveBlockData("secpath", self.idx, self.secPathEstimate.ir)




# class FxLMSKernelSPM1(ConstrainedFastBlockFxLMS):
#     def __init__(self, mu, beta, speakerFilters, muSPM, secPathEstimateLen):
#         super().__init__(mu, beta, speakerFilters)
#         self.name = "FxLMS SPM Eriksson"
#         self.muSPM = muSPM
#         self.spmLen = secPathEstimateLen
#         self.auxNoiseSource = GoldSequenceSource(11, numChannels=s.NUMSPEAKER)

#         self.kernelParams = soundfieldInterpolation(errorPos, errorPos, 2*blockSize, 1e-2)

#         self.secPathEstimate = FilterMD_IntBuffer(dataDims=s.NUMREF, irLen=secPathEstimateLen, irDims=(s.NUMSPEAKER, s.NUMERROR))  
#         self.secPathEstimateSum = FilterSum_IntBuffer(irLen=secPathEstimateLen, numIn=s.NUMSPEAKER, numOut=s.NUMERROR)

#         initialSecPathEstimate = np.random.normal(scale=0.0001, size=self.secPathEstimate.ir.shape)
#         commonLength = np.min((secPathEstimateLen,speakerFilters["error"].shape[-1]))
#         initialSecPathEstimate[:,:,0:commonLength] = 0.5 * speakerFilters["error"][:,:,0:commonLength]
        
#         self.secPathEstimate.setIR(initialSecPathEstimate)
#         self.secPathEstimateSum.setIR(initialSecPathEstimate)

#         self.buffers["xf"] = np.zeros((s.NUMREF, s.NUMSPEAKER, s.NUMERROR, s.SIMCHUNKSIZE+s.SIMBUFFER))
#         self.buffers["v"] = np.zeros((s.NUMSPEAKER, s.SIMCHUNKSIZE+s.SIMBUFFER))
#         self.buffers["vEst"] = np.zeros((s.NUMERROR, s.SIMCHUNKSIZE+s.SIMBUFFER))

#         self.diag.addNewDiagnostic("secpath", dia.ConstantEstimateNMSE(speakerFilters["error"], plotFrequency=s.PLOTFREQUENCY))

#     def updateSPM(self):
#         self.buffers["vEst"][:,self.updateIdx:self.idx] = self.secPathEstimateSum.process(
#                                                             self.buffers["v"][:,self.updateIdx:self.idx])

#         f = self.e[:,self.updateIdx:self.idx] - self.buffers["vEst"][:,self.updateIdx:self.idx]

#         grad = np.zeros_like(self.secPathEstimateSum.ir)
#         numSamples = self.idx-self.updateIdx
#         for i in range(numSamples):
#             V = np.flip(self.buffers["v"][:,None,self.updateIdx+i-self.spmLen+1:self.updateIdx+i+1], axis=-1)
#             grad += V * f[None,:,i:i+1] / (np.sum(V**2) + 1e-4)

#         grad *= self.muSPM
#         self.secPathEstimate.ir += grad
#         self.secPathEstimateSum.ir[:,:,:] = self.secPathEstimate.ir[:,:,:]

#         self.diag.saveBlockData("secpath", self.updateIdx, self.secPathEstimate.ir)

#     def updateFilter(self):
#         self.buffers["xf"][:,:,:,self.updateIdx:self.idx] = np.transpose(self.secPathEstimate.process(
#                                                         self.x[:,self.updateIdx:self.idx]), (2,0,1,3))
#         grad = np.zeros_like(self.controlFilt.ir)
#         for n in range(self.updateIdx, self.idx):
#            Xf = np.flip(self.buffers["xf"][:,:,:,n-s.FILTLENGTH+1:n+1], axis=-1)
#            grad += np.sum(Xf * self.e[None, None,:,n,None],axis=2) / (np.sum(Xf**2) + self.beta)

#         self.controlFilt.ir -= grad * self.mu
#         #super().updateFilter()
#         self.updateSPM()
#         self.updateIdx = self.idx

        
#     def forwardPassImplement(self, numSamples): 
#         y = self.controlFilt.process(self.x[:,self.idx:self.idx+numSamples])

#         v = self.auxNoiseSource.getSamples(numSamples)
#         self.buffers["v"][:,self.idx:self.idx+numSamples] = v

#         self.y[:,self.idx:self.idx+numSamples] = v + y


class KernelSPM3(FastBlockFxLMSSPMEriksson):
    """UPDATED VERSION FOR NEW FAST BLOCK FXLMS
    
    Attempts to update each sec path filter using data from all microphones.
        It uses the normal Freq domain Eriksson cost function, together with
        ||e - G_ip y_spm||^2 where G_ip is the sec path interpolated to microphone m
        from all positions except m (to avoid the inteprolation filter reducing to identity)."""
    def __init__(self, mu, muSPM, eta, speakerRIR, blockSize, errorPos, kiFiltLen):
        super().__init__(mu, muSPM, speakerRIR, blockSize)
        self.name = "Freq ANC Kernel SPM 3"
        assert(kiFiltLen % 2 == 1)
        assert(kiFiltLen <= blockSize)
        self.eta = eta
        kernelReg = 1e-3
        self.kernelMat = self.constructInterpolation(errorPos, kernelReg, kiFiltLen)
        self.kiDelay = kiFiltLen // 2

        self.nextBlockAuxNoise = self.auxNoiseSource.getSamples(blockSize)
        #self.nextBlockFilteredAuxNoise = np.zeros((s.NUMERROR, blockSize))

        self.buffers["v"] = np.zeros((s.NUMSPEAKER, s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.buffers["vf"] = np.zeros((s.NUMERROR, s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.buffers["f"] = np.zeros((s.NUMERROR, s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.buffers["fip"] = np.zeros((s.NUMERROR, s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.buffers["vIP"] = np.zeros((s.NUMSPEAKER, s.NUMERROR, s.NUMERROR, s.SIMCHUNKSIZE+s.SIMBUFFER))

        self.metadata["kernelReg"] = kernelReg
        self.metadata["kernelCostWeight"] = self.eta

    def constructInterpolation(self, errorPos, kernelRegParam, kiFiltLen):
        """Output is shape(numFreq, position interpolated to, positions interpolated from),
            which is (2*blocksize, M, M-1)"""
        kiParams = np.zeros((2*self.blockSize, s.NUMERROR, s.NUMERROR), dtype=np.complex128)
        kiFirAll = np.zeros((16,16,155))
        for m in range(s.NUMERROR):
            kiFIR = soundfieldInterpolationFIR(errorPos[m:m+1,:], 
                                            errorPos[np.arange(s.NUMERROR) != m,:], 
                                            kiFiltLen, kernelRegParam)

            kiFreq = fdf.fftWithTranspose(np.concatenate((kiFIR, 
                np.zeros((kiFIR.shape[:-1]+(2*self.blockSize-kiFiltLen,)))), 
                axis=-1), addEmptyDim=False)
            kiParams[:,m:m+1,np.arange(s.NUMERROR)!=m] = kiFreq

            kiFirAll[m:m+1,np.arange(s.NUMERROR)!=m,:] = kiFIR
        return kiParams

    def getVariableNumberOfFutureSamples():
        v = np.zeros((s.NUMSPEAKER, numSamples))
        numOldSamples = np.min((numSamples,self.futureAuxNoise.shape[-1]))
        v[:,:numOldSamples] = self.futureAuxNoise[:,:numOldSamples]
        numNewSamples = numSamples - self.futureAuxNoise.shape[-1]
        if numNewSamples > 0:
            v[:,numOldSamples:] = self.auxNoiseSource.getSamples(numNewSamples)
            
        newToReplace = self.futureAuxNoise.shape[-1] - numOldSamples
        self.futureAuxNoise[:,:newToReplace] = self.futureAuxNoise[:,numOldSamples:]
        self.futureAuxNoise[:,newToReplace:] = self.auxNoiseSource.getSamples(newToReplace)

    def forwardPassImplement(self, numSamples): 
        assert(numSamples == self.blockSize)
        self.y[:,self.idx:self.idx+self.blockSize] = self.controlFilt.process(
                self.x[:,self.idx:self.idx+self.blockSize])
        
        v = self.nextBlockAuxNoise
        self.nextBlockAuxNoise[...] = self.auxNoiseSource.getSamples(numSamples)

        self.buffers["v"][:,self.idx:self.idx+numSamples] = v
        self.y[:,self.idx:self.idx+numSamples] += v 
        self.updated = False

    def interpolateFIR(kiFilt, fir):
        """kiFilt is (outDim, inDim, irLen)
            fir is (extraDim, inDim, irLen2)
            out is (extraDim, outDim, irLen2)"""
        assert(kiFilt.shape[-1]%2 == 1)
        out = np.zeros((fir.shape[:-1] + (fir.shape[-1]+kiFilt.shape[-1]-1,)))
        for i in range(kiFilt.shape[0]): #outDim
            for j in range(kiFilt.shape[1]): #inDim
                for k in range(fir.shape[0]): #extraDim
                    out[k,i,:] += np.convolve(kiFirAll[i,j,:], self.secPathFilt.ir[k,j,:], "full")
        return out[...,irLen//2:-irLen//2+1]

    def getIPGrad(self):
        #Calculate error signal
        nextBlockVf = fdf.convolveSum(self.secPathEstimate.tf, np.concatenate(
            (self.buffers["v"][:,self.updateIdx:self.idx], 
            self.nextBlockAuxNoise),axis=-1))
        vfWithFutureValues = np.concatenate(
            (self.buffers["vf"][:,self.idx-(2*self.blockSize)+self.kiDelay:self.idx], 
            nextBlockVf[:,:self.kiDelay]), axis=-1)
        vWithFutureValues = np.concatenate(
            (self.buffers["v"][:,self.idx-(2*self.blockSize)+self.kiDelay:self.idx], 
            self.nextBlockAuxNoise[:,:self.kiDelay]), axis=-1)

        vfIP = fdf.convolveSum(self.kernelMat, vfWithFutureValues)

        self.buffers["fip"][:,self.updateIdx:self.idx] = self.e[:,self.updateIdx:self.idx] - vfIP

        #Calculate the interpolated reference
        self.buffers["vIP"][...,self.updateIdx:self.idx] = \
            np.transpose(fdf.convolveEuclidianFT(
                self.kernelMat, vWithFutureValues),(2,1,0,3))

        #Calculate the correlation, ie. gradient
        grad = fdf.correlateSumTT(self.buffers["fip"][None,None,:,self.updateIdx:self.idx],
                self.buffers["vIP"][...,self.idx-(2*self.blockSize):self.idx])
        grad = np.transpose(grad, (1,0,2))
        grad = fdf.fftWithTranspose(np.concatenate(
                (grad, np.zeros_like(grad)),axis=-1), addEmptyDim=False)

        norm = 1 / (np.sum(self.buffers["vIP"][...,self.updateIdx:self.idx]**2) + 1e-3)
        return grad * norm

    def getNormalGrad(self):
        self.buffers["vf"][:,self.updateIdx:self.idx] = self.secPathEstimate.process(
            self.buffers["v"][:,self.updateIdx:self.idx])

        self.buffers["f"][:,self.updateIdx:self.idx] = self.e[:,self.updateIdx:self.idx] - \
            self.buffers["vf"][:,self.updateIdx:self.idx]

        grad = fdf.correlateEuclidianTT(self.buffers["f"][:,self.updateIdx:self.idx], 
                                self.buffers["v"][:,self.idx-(2*self.blockSize):self.idx])
        
        grad = fdf.fftWithTranspose(np.concatenate(
                (grad, np.zeros_like(grad)),axis=-1),addEmptyDim=False)
        norm = 1 / (np.sum(self.buffers["v"][:,self.updateIdx:self.idx]**2) + 1e-3)
        return grad * norm

    def updateSPM(self):
        grad = self.getNormalGrad()
        grad += self.eta * self.getIPGrad()

        self.secPathEstimate.tf += self.muSPM * grad
        self.secPathEstimateMD.tf[...] = self.secPathEstimate.tf

        self.diag.saveBlockData("secpath", self.idx, self.secPathEstimate.tf)





class KernelSPM4(FastBlockFxLMSSPMEriksson):
    """
    Attempts to update each sec path filter using data from all microphones.
    Uses the kernel including all microphones. Therefore only uses the 
    interpolated cost function"""
    def __init__(self, mu, muSPM, speakerRIR, blockSize, errorPos, kiFiltLen):
        super().__init__(mu, muSPM, speakerRIR, blockSize)
        self.name = "Freq ANC Kernel SPM 4"
        assert(kiFiltLen % 2 == 1)
        assert(kiFiltLen <= blockSize)
        kernelReg = 1e-3
        self.kernelMat = self.constructInterpolation(errorPos, kernelReg, kiFiltLen)
        self.kiDelay = kiFiltLen // 2

        self.nextBlockAuxNoise = self.auxNoiseSource.getSamples(blockSize)

        self.buffers["v"] = np.zeros((s.NUMSPEAKER, s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.buffers["vf"] = np.zeros((s.NUMERROR, s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.buffers["f"] = np.zeros((s.NUMERROR, s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.buffers["fip"] = np.zeros((s.NUMERROR, s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.buffers["vIP"] = np.zeros((s.NUMSPEAKER, s.NUMERROR, s.NUMERROR, s.SIMCHUNKSIZE+s.SIMBUFFER))

        self.metadata["kernelReg"] = kernelReg
        #self.metadata["kernelCostWeight"] = self.eta

    def constructInterpolation(self, errorPos, kernelRegParam, kiFiltLen):
        """Output is shape(numFreq, position interpolated to, positions interpolated from),
            which is (2*blocksize, M, M)"""
        kiParams = np.zeros((2*self.blockSize, s.NUMERROR, s.NUMERROR), dtype=np.complex128)
        
        kiFIR = soundfieldInterpolationFIR(errorPos, errorPos, 
                                        kiFiltLen, kernelRegParam)

        kiFreq = fdf.fftWithTranspose(np.concatenate((kiFIR, 
            np.zeros((kiFIR.shape[:-1]+(2*self.blockSize-kiFiltLen,)))), 
            axis=-1), addEmptyDim=False)
        return kiFreq

    def forwardPassImplement(self, numSamples): 
        assert(numSamples == self.blockSize)
        self.y[:,self.idx:self.idx+self.blockSize] = self.controlFilt.process(
                self.x[:,self.idx:self.idx+self.blockSize])
        
        v = self.nextBlockAuxNoise
        self.nextBlockAuxNoise[...] = self.auxNoiseSource.getSamples(numSamples)

        self.buffers["v"][:,self.idx:self.idx+numSamples] = v
        self.y[:,self.idx:self.idx+numSamples] += v 
        self.updated = False

    def getIPGrad(self):
        #Calculate error signal
        nextBlockVf = fdf.convolveSum(self.secPathEstimate.tf, np.concatenate(
            (self.buffers["v"][:,self.updateIdx:self.idx], 
            self.nextBlockAuxNoise),axis=-1))
        vfWithFutureValues = np.concatenate(
            (self.buffers["vf"][:,self.idx-(2*self.blockSize)+self.kiDelay:self.idx], 
            nextBlockVf[:,:self.kiDelay]), axis=-1)
        vWithFutureValues = np.concatenate(
            (self.buffers["v"][:,self.idx-(2*self.blockSize)+self.kiDelay:self.idx], 
            self.nextBlockAuxNoise[:,:self.kiDelay]), axis=-1)

        vfIP = fdf.convolveSum(self.kernelMat, vfWithFutureValues)

        self.buffers["fip"][:,self.updateIdx:self.idx] = self.e[:,self.updateIdx:self.idx] - vfIP

        #Calculate the interpolated reference
        self.buffers["vIP"][...,self.updateIdx:self.idx] = \
            np.transpose(fdf.convolveEuclidianFT(
                self.kernelMat, vWithFutureValues),(2,1,0,3))

        #Calculate the correlation, ie. gradient
        grad = fdf.correlateSumTT(self.buffers["fip"][None,None,:,self.updateIdx:self.idx],
                self.buffers["vIP"][...,self.idx-(2*self.blockSize):self.idx])
        grad = np.transpose(grad, (1,0,2))
        grad = fdf.fftWithTranspose(np.concatenate(
                (grad, np.zeros_like(grad)),axis=-1), addEmptyDim=False)

        norm = 1 / (np.sum(self.buffers["vIP"][...,self.updateIdx:self.idx]**2) + 1e-3)
        return grad * norm

    def updateSPM(self):
        self.secPathEstimate.tf += self.muSPM * self.getIPGrad()
        self.secPathEstimateMD.tf[...] = self.secPathEstimate.tf

        self.diag.saveBlockData("secpath", self.idx, self.secPathEstimate.tf)




class KernelSPM5(FastBlockFxLMSSPMEriksson):
    """Closely related to version 3. Uses verified interpolation method in time domain
        """
    def __init__(self, mu, muSPM, eta, speakerRIR, blockSize, errorPos, kiFiltLen):
        super().__init__(mu, muSPM, speakerRIR, blockSize)
        self.name = "Freq ANC Kernel SPM 5"
        assert(kiFiltLen % 2 == 1)
        assert(kiFiltLen <= blockSize)
        self.eta = eta
        kernelReg = 1e-3
        self.kiFilt = FilterMD_IntBuffer(dataDims=s.NUMSPEAKER, ir=self.constructInterpolation(errorPos, kernelReg, kiFiltLen))

        self.secPathEstimate = FilterSum_IntBuffer(numIn=s.NUMSPEAKER, numOut=s.NUMERROR, irLen=blockSize)
        self.secPathEstimateIP = FilterSum_IntBuffer(numIn=s.NUMSPEAKER, numOut=s.NUMERROR, irLen=blockSize)
        self.secPathEstimateMD = FilterMD_IntBuffer(dataDims=s.NUMREF, ir=self.secPathEstimate.ir)
        self.kiDelay = kiFiltLen // 2

        self.nextBlockAuxNoise = self.auxNoiseSource.getSamples(blockSize)
        #self.nextBlockFilteredAuxNoise = np.zeros((s.NUMERROR, blockSize))

        self.diag.addNewDiagnostic("secpath", dia.ConstantEstimateNMSE(self.secPathFilt.ir, plotFrequency=s.PLOTFREQUENCY))

        self.buffers["v"] = np.zeros((s.NUMSPEAKER, s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.buffers["vf"] = np.zeros((s.NUMERROR, s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.buffers["f"] = np.zeros((s.NUMERROR, s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.buffers["fip"] = np.zeros((s.NUMERROR, s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.buffers["vIP"] = np.zeros((s.NUMSPEAKER, s.NUMERROR, s.NUMERROR, s.SIMCHUNKSIZE+s.SIMBUFFER))

        self.metadata["kernelReg"] = kernelReg
        self.metadata["kernelCostWeight"] = self.eta



    def updateFilter(self):
        assert (self.idx - self.updateIdx == self.blockSize)
        assert(self.updated == False)
        self.updateSPM()

        self.buffers["xf"][:,:,:,self.updateIdx:self.idx] = np.transpose(
            self.secPathEstimateMD.process(self.x[:,self.updateIdx:self.idx]),
            (1,0,2,3))
        
        tdGrad = fdf.correlateSumTT(self.buffers["f"][None,None,:,self.updateIdx:self.idx], 
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

    def constructInterpolation(self, errorPos, kernelRegParam, kiFiltLen):
        """Output is shape(numFreq, position interpolated to, positions interpolated from),
            which is (2*blocksize, M, M-1)"""
        kiParams = np.zeros((2*self.blockSize, s.NUMERROR, s.NUMERROR), dtype=np.complex128)
        kiFirAll = np.zeros((s.NUMERROR,s.NUMERROR,kiFiltLen))
        for m in range(s.NUMERROR):
            kiFIR = soundfieldInterpolationFIR(errorPos[m:m+1,:], 
                                            errorPos[np.arange(s.NUMERROR) != m,:], 
                                            kiFiltLen, kernelRegParam)

            kiFreq = fdf.fftWithTranspose(np.concatenate((kiFIR, 
                np.zeros((kiFIR.shape[:-1]+(2*self.blockSize-kiFiltLen,)))), 
                axis=-1), addEmptyDim=False)
            kiParams[:,m:m+1,np.arange(s.NUMERROR)!=m] = kiFreq

            kiFirAll[m:m+1,np.arange(s.NUMERROR)!=m,:] = kiFIR
        return kiFirAll

    def forwardPassImplement(self, numSamples): 
        assert(numSamples == self.blockSize)
        self.y[:,self.idx:self.idx+self.blockSize] = self.controlFilt.process(
                self.x[:,self.idx:self.idx+self.blockSize])
        
        v = self.nextBlockAuxNoise
        self.nextBlockAuxNoise[...] = self.auxNoiseSource.getSamples(numSamples)

        self.buffers["v"][:,self.idx:self.idx+numSamples] = v
        self.y[:,self.idx:self.idx+numSamples] += v 
        self.updated = False

    def interpolateFIR(self, kiFilt, fir):
        """kiFilt is (outDim, inDim, irLen)
            fir is (extraDim, inDim, irLen2)
            out is (extraDim, outDim, irLen2)"""
        assert(kiFilt.shape[-1]%2 == 1)
        out = np.zeros((fir.shape[:-1] + (fir.shape[-1]+kiFilt.shape[-1]-1,)))
        for i in range(kiFilt.shape[0]): #outDim
            for j in range(kiFilt.shape[1]): #inDim
                for k in range(fir.shape[0]): #extraDim
                    out[k,i,:] += np.convolve(kiFilt[i,j,:], fir[k,j,:], "full")
        return out[...,kiFilt.shape[-1]//2:-kiFilt.shape[-1]//2+1]

    def getIPGrad(self):
        #Calculate error signal
        self.secPathEstimateIP.ir = self.interpolateFIR(self.kiFilt.ir, self.secPathEstimate.ir)
        vfIP = self.secPathEstimateIP.process(self.buffers["v"][...,self.updateIdx:self.idx])

        self.buffers["fip"][:,self.updateIdx:self.idx] = self.e[:,self.updateIdx:self.idx] - vfIP

        vWithFutureValues = np.concatenate(
            (self.buffers["v"][:,self.idx-(self.blockSize)+self.kiDelay:self.idx], 
            self.nextBlockAuxNoise[:,:self.kiDelay]), axis=-1)

        self.buffers["vIP"][...,self.updateIdx:self.idx] = np.transpose(
            self.kiFilt.process(vWithFutureValues),(2,1,0,3))

        #Calculate the correlation, ie. gradient
        grad = fdf.correlateSumTT(self.buffers["fip"][None,None,:,self.updateIdx:self.idx],
                self.buffers["vIP"][...,self.idx-(2*self.blockSize):self.idx])

        norm = 1 / (np.sum(self.buffers["vIP"][...,self.updateIdx:self.idx]**2) + 1e-3)
        return grad * norm

    def getNormalGrad(self):
        self.buffers["vf"][:,self.updateIdx:self.idx] = self.secPathEstimate.process(
            self.buffers["v"][:,self.updateIdx:self.idx])

        self.buffers["f"][:,self.updateIdx:self.idx] = self.e[:,self.updateIdx:self.idx] - \
            self.buffers["vf"][:,self.updateIdx:self.idx]

        grad = fdf.correlateEuclidianTT(self.buffers["f"][:,self.updateIdx:self.idx], 
                                self.buffers["v"][:,self.idx-(2*self.blockSize):self.idx])
        
        # grad = fdf.fftWithTranspose(np.concatenate(
        #         (grad, np.zeros_like(grad)),axis=-1),addEmptyDim=False)
        norm = 1 / (np.sum(self.buffers["v"][:,self.updateIdx:self.idx]**2) + 1e-3)
        grad = np.transpose(grad, (1,0,2))
        return grad * norm

    def updateSPM(self):
        grad = self.getNormalGrad()
        grad += self.eta * self.getIPGrad()

        self.secPathEstimate.ir += self.muSPM * grad
        self.secPathEstimateMD.ir[...] = self.secPathEstimate.ir

        self.diag.saveBlockData("secpath", self.idx, self.secPathEstimate.ir)






class KernelSPM3_old(FreqAuxNoiseFxLMS2):
    """Attempts to update each sec path filter using data from all microphones.
        It uses the normal Freq domain Eriksson cost function, together with
        ||e - G_ip y_spm||^2 where G_ip is the sec path interpolated to microphone m
        from all positions except m (to avoid the inteprolation filter reducing to identity)."""
    def __init__(self, mu, muSPM, eta, speakerRIR, blockSize, errorPos):
        super().__init__(mu, muSPM, speakerRIR, blockSize,)
        self.name = "Freq ANC Kernel SPM 3"
        self.eta = eta
        kernelReg = 1e-3
        self.kernelMat = self.constructInterpolation(errorPos, kernelReg)
        
        self.buffers["v"] = np.zeros((s.NUMSPEAKER, s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.buffers["f"] = np.zeros((s.NUMERROR, s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.buffers["fip"] = np.zeros((s.NUMERROR, s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.buffers["VIP"] = np.zeros((s.NUMSPEAKER, s.NUMERROR, s.NUMERROR, s.SIMCHUNKSIZE+s.SIMBUFFER))

        self.metadata["kernelReg"] = kernelReg
        self.metadata["kernelCostWeight"] = self.eta

    def constructInterpolation(self, errorPos, kernelRegParam):
        """Output is shape(numFreq, position interpolated to, positions interpolated from),
            which is (2*blocksize, M, M-1)"""
        kiParams = np.zeros((2*self.blockSize, s.NUMERROR, s.NUMERROR), dtype=np.complex128)
        for m in range(s.NUMERROR):
            kiParams[:,m:m+1,np.arange(s.NUMERROR)!=m] = soundfieldInterpolation(errorPos[m:m+1,:], 
                                            errorPos[np.arange(s.NUMERROR) != m,:], 
                                            2*self.blockSize, kernelRegParam)
            # for m2 in range(s.NUMERROR):
            #     if m2 < m:
            #         kiParams[:,m:m+1,m2] = kip[:,:,m2]
            #     elif m2 > m:
            #         kiParams[:,m:m+1,m2]
        return kiParams

    def filterAuxNoise(self, secPath, auxNoise):
        auxNoise = np.fft.fft(auxNoise, axis=-1).T[:,:,None]
        output = secPath @ auxNoise
        output = np.squeeze(np.fft.ifft(output,axis=0),axis=-1).T
        output = output[:,self.blockSize:]
        return np.real(output)

    def interpolateAuxNoise(self):
        V = np.fft.fft(self.buffers["v"][:,self.idx-(2*self.blockSize):self.idx],axis=-1).T[:,:,None]
        
        VIP = self.kernelMat[:,None,:,:] * V[:,:,:,None]
        VIP = np.transpose(np.fft.ifft(VIP,axis=0), (1,2,3,0))
        self.buffers["VIP"][:,:,:,self.updateIdx:self.idx] = np.real(VIP[:,:,:,self.blockSize:])

        VIP = np.transpose(np.fft.fft(self.buffers["VIP"][:,:,:,self.idx-(2*self.blockSize):self.idx], axis=-1), (3,0,1,2))
        return VIP

    def getIPGrad(self):
        Gip = self.kernelMat @ self.secPathEstimate.ir
        vfIP = self.filterAuxNoise(Gip, self.buffers["v"][:,self.idx-(2*self.blockSize):self.idx])
        self.buffers["fip"][:,self.updateIdx:self.idx] = self.e[:,self.updateIdx:self.idx] - vfIP

        VIP = self.interpolateAuxNoise()
        Fip = np.concatenate((np.zeros_like(self.buffers["fip"][:,self.updateIdx:self.idx]), 
                                            self.buffers["fip"][:,self.updateIdx:self.idx]), axis=-1)
        Fip = np.fft.fft(Fip, axis=-1).T[:,:,None]

        error2 = np.zeros((2*self.blockSize, s.NUMERROR, s.NUMSPEAKER),dtype=np.complex128)
        for m in range(s.NUMERROR):
            error2[:,m,:] = np.sum(Fip[:,None,:,0] * VIP[:,:,:,m].conj(), axis=2)
            #error2[:,m,:]
        
        powerPerFreq = np.sum(np.abs(VIP)**2,axis=(1,2,3))
        VIPnorm = 1 / (np.mean(powerPerFreq) + self.beta)
        ipGrad = self.eta * error2 * VIPnorm
        return ipGrad

    def getNormalGrad_new(self):
        vf = self.secPathEstimate.process(np.concatenate((np.zeros((s.NUMSPEAKER,self.blockSize)),
                                                        self.buffers["v"][:,self.updateIdx:self.idx]), axis=-1))

        self.buffers["f"][:,self.updateIdx:self.idx] = self.e[:,self.updateIdx:self.idx] - vf
        #self.secPathEstimate.update(self.buffers["v"][:,self.updateIdx-self.blockSize:self.idx], 
        #                            self.buffers["f"][:,self.updateIdx:self.idx])

        #update
        paddedError = np.concatenate((np.zeros_like(self.buffers["f"][:,self.updateIdx:self.idx]),
                                        self.buffers["f"][:,self.updateIdx:self.idx]), axis=-1)
        F = np.fft.fft(paddedError, axis=-1).T[:,:,None]
        V = np.fft.fft(self.buffers["v"][:,self.idx-(2*self.blockSize):self.idx], axis=-1).T[:,:,None]

        gradient =  F @ np.transpose(V.conj(),(0,2,1))
        # tdgrad = np.fft.ifft(gradient, axis=0)
        # tdgrad[self.blockSize:,:,:] = 0
        # gradient = np.fft.fft(tdgrad, axis=0)

        norm = 1 / (np.mean(np.transpose(V.conj(),(0,2,1)) @ V) + self.beta)
        return norm * gradient


    def getNormalGrad(self):
        #vf = self.filterAuxNoise(self.secPathEstimate.ir, self.buffers["v"][:,self.idx-(2*self.blockSize):self.idx])
        vf = self.secPathEstimate.process(self.buffers["v"][:,self.idx-(2*self.blockSize):self.idx])
        self.buffers["f"][:,self.updateIdx:self.idx] = self.e[:,self.updateIdx:self.idx] - vf
        
        F = np.concatenate((np.zeros_like(self.buffers["f"][:,self.updateIdx:self.idx]), 
                                            self.buffers["f"][:,self.updateIdx:self.idx]), axis=-1)
        F = np.fft.fft(F, axis=-1).T[:,:,None]

        V = np.fft.fft(self.buffers["v"][:,self.idx-(2*self.blockSize):self.idx], axis=-1).T[:,:,None]
        error1 = F @ np.transpose(V.conj(), (0,2,1))

        powerPerFreq = np.sum(np.abs(V)**2,axis=(1,2))
        Vnorm = 1 / (np.mean(powerPerFreq) + self.beta)

        normalGrad = error1*Vnorm
        return normalGrad

    def updateSPM(self):
        grad = self.getNormalGrad() + self.eta * self.getIPGrad()
        grad = np.fft.ifft(grad,axis=0)
        grad[self.blockSize:,:,:] = 0
        grad = np.fft.fft(grad, axis=0)

        self.secPathEstimate.ir += self.muSPM * grad

        self.diag.saveBlockData("secpath", self.idx, self.secPathEstimate.ir)
