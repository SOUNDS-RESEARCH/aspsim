import numpy as np
import scipy.signal.windows as win

from ancsim.adaptivefilter.base import AdaptiveFilterFF
from ancsim.adaptivefilter.mpc import ConstrainedFastBlockFxLMS, FastBlockFxLMS
from ancsim.signal.filterclasses import FilterSum_IntBuffer, FilterMD_IntBuffer, \
    SinglePoleLowPass, FilterIndividualInputs, FilterSum_Freqdomain, FilterMD_Freqdomain, \
        MovingAverage
from ancsim.signal.sources import WhiteNoiseSource, GoldSequenceSource
from ancsim.adaptivefilter.util import getWhiteNoiseAtSNR
from ancsim.adaptivefilter.conventional.lms import NLMS, NLMS_FREQ, FastBlockNLMS, BlockNLMS
import ancsim.signal.freqdomainfiltering as fdf
import ancsim.adaptivefilter.diagnostics as dia
import ancsim.settings as s
import ancsim.utilities as util



class FxLMSSPMErikssonBlock(AdaptiveFilterFF):
    """Using block based normalization for both ANC and SPM
        Corresponds to FreqAuxNoiseFxLMS"""
    def __init__(self, mu, beta, speakerRIR, muSPM, secPathEstimateLen, auxNoisePower=3):
        super().__init__(mu, beta, speakerRIR)
        self.name = "FxLMS SPM Eriksson Block"
        self.muSPM = muSPM
        self.spmLen = secPathEstimateLen

        self.auxNoiseSource = WhiteNoiseSource(power=auxNoisePower, numChannels=s.NUMSPEAKER)
        #self.auxNoiseSource = GoldSequenceSource(11, power=10, numChannels=s.NUMSPEAKER)
        self.secPathEstimateMD = FilterMD_IntBuffer(dataDims=s.NUMREF, irLen=secPathEstimateLen, irDims=(s.NUMSPEAKER, s.NUMERROR))  

        self.secPathEstimate = BlockNLMS(irLen=secPathEstimateLen, numIn=s.NUMSPEAKER, numOut=s.NUMERROR, 
                                    stepSize=muSPM)

        self.buffers["xf"] = np.zeros((s.NUMREF, s.NUMSPEAKER, s.NUMERROR, s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.buffers["v"] = np.zeros((s.NUMSPEAKER, s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.buffers["vEst"] = np.zeros((s.NUMERROR, s.SIMCHUNKSIZE+s.SIMBUFFER))

        self.diag.addNewDiagnostic("secpath", dia.ConstantEstimateNMSE(speakerRIR["error"], plotFrequency=s.PLOTFREQUENCY))

        self.metadata["auxNoiseSource"] = self.auxNoiseSource.__class__.__name__
        self.metadata["auxNoisePower"] = self.auxNoiseSource.power
        self.metadata["muSPM"] = muSPM
        self.metadata["secPathEstimateLen"] = secPathEstimateLen
        self.metadata["auxNoisePower"] = auxNoisePower
        

    def updateSPM(self):
        self.buffers["vEst"][:,self.updateIdx:self.idx] = self.secPathEstimate.process(
                                            self.buffers["v"][:,self.updateIdx:self.idx])

        f = self.e[:,self.updateIdx:self.idx] - self.buffers["vEst"][:,self.updateIdx:self.idx]
        self.secPathEstimate.update(self.buffers["v"][:,self.updateIdx:self.idx], f)

        self.secPathEstimateMD.ir[...] = self.secPathEstimate.filt.ir

        self.diag.saveBlockData("secpath", self.updateIdx, self.secPathEstimate.filt.ir)

    def updateFilter(self):
        self.updateSPM()
        self.buffers["xf"][:,:,:,self.updateIdx:self.idx] = np.transpose(self.secPathEstimateMD.process(
                                                        self.x[:,self.updateIdx:self.idx]), (2,0,1,3))
        grad = np.zeros_like(self.controlFilt.ir)
        for n in range(self.updateIdx, self.idx):
           Xf = np.flip(self.buffers["xf"][:,:,:,n-s.FILTLENGTH+1:n+1], axis=-1)
           grad += np.sum(Xf * self.e[None, None,:,n,None],axis=2)# / (np.sum(Xf**2) + self.beta)

        norm = 1 / (np.sum(self.buffers["xf"][:,:,:,self.updateIdx:self.idx]**2) + self.beta)
        self.controlFilt.ir -= grad * self.mu * norm
        
        self.updateIdx = self.idx

    def forwardPassImplement(self, numSamples): 
        y = self.controlFilt.process(self.x[:,self.idx:self.idx+numSamples])
        v = self.auxNoiseSource.getSamples(numSamples)
        self.buffers["v"][:,self.idx:self.idx+numSamples] = v
        self.y[:,self.idx:self.idx+numSamples] = v + y



class FastBlockFxLMSSPMEriksson(FastBlockFxLMS):
    """Frequency domain equivalent algorithm to FxLMSSPMEriksson"""
    def __init__(self, mu, muSPM, speakerRIR, blockSize):
        self.beta = 1e-3
        super().__init__(mu, self.beta, speakerRIR, blockSize)
        self.name = "SPM Aux Noise Freq Domain"
        self.muSPM = muSPM
        self.diag.addNewDiagnostic("secpath", dia.ConstantEstimateNMSE(self.secPathEstimate.tf, plotFrequency=s.PLOTFREQUENCY))
        #self.auxNoiseSource = GoldSequenceSource(11, power=10, numChannels=s.NUMSPEAKER)
        self.auxNoiseSource = WhiteNoiseSource(power=3, numChannels=s.NUMSPEAKER)

        self.secPathEstimate = FilterSum_Freqdomain(numIn=s.NUMSPEAKER,numOut=s.NUMERROR,irLen=blockSize)
        self.secPathEstimateMD = FilterMD_Freqdomain(dataDims=s.NUMREF, tf=self.secPathEstimate.tf)

        self.buffers["v"] = np.zeros((s.NUMSPEAKER, s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.buffers["f"] = np.zeros((s.NUMERROR, s.SIMCHUNKSIZE+s.SIMBUFFER))
        #self.buffers["vf"] = np.zeros((s.NUMERROR, ))

        self.metadata["auxNoiseSource"] = self.auxNoiseSource.__class__.__name__
        self.metadata["auxNoisePower"] = self.auxNoiseSource.power
        self.metadata["muSPM"] = muSPM

    def forwardPassImplement(self, numSamples): 
        super().forwardPassImplement(numSamples)

        v = self.auxNoiseSource.getSamples(numSamples)
        self.buffers["v"][:,self.idx:self.idx+numSamples] = v
        self.y[:,self.idx:self.idx+numSamples] += v

    def updateSPM(self):
        vf = self.secPathEstimate.process(self.buffers["v"][:,self.updateIdx:self.idx])

        self.buffers["f"][:,self.updateIdx:self.idx] = self.e[:,self.updateIdx:self.idx] - vf

        grad = fdf.correlateEuclidianTT(self.buffers["f"][:,self.updateIdx:self.idx], 
                                self.buffers["v"][:,self.idx-(2*self.blockSize):self.idx])
        
        grad = fdf.fftWithTranspose(np.concatenate(
                (grad, np.zeros_like(grad)),axis=-1))
        norm = 1 / (np.sum(self.buffers["v"][:,self.updateIdx:self.idx]**2) + 1e-3)

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
        
        tdGrad = fdf.correlateSumTT(self.buffers["f"][None,None,:,self.updateIdx:self.idx], 
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



class FxLMSSPMTuningless(AdaptiveFilterFF):
    """As we know the akf of the auxiliary noise, we should be able to get the 
        wiener solution without inverting the estimated akf matrix, which is usually the problem.
        This is similar to the 'a tuning-less approach in secondary path 
        modelling in active noise control systems'"""
    def __init__(self, mu, invForgetFactor, speakerRIR, secPathEstimateLen):
        self.beta = 1e-3
        super().__init__(mu, self.beta, speakerRIR)
        self.name = "SPM Aux Noise Wiener"
        self.secPathEstLen = secPathEstimateLen
        self.auxNoisePower = 3
        self.auxNoiseSource = WhiteNoiseSource(power=self.auxNoisePower, numChannels=s.NUMSPEAKER)

        self.secPathEstimate = FilterSum_IntBuffer(irLen=secPathEstimateLen, numIn=s.NUMSPEAKER, numOut=s.NUMERROR)
        self.secPathEstimateMD = FilterMD_IntBuffer(dataDims=s.NUMREF, irLen=secPathEstimateLen, irDims=(s.NUMSPEAKER, s.NUMERROR))
        #self.crossCorr = SinglePoleLowPass(0.997, (s.NUMSPEAKER, s.NUMERROR, secPathEstimateLen))
        self.crossCorr = MovingAverage(1-invForgetFactor, (s.NUMSPEAKER, s.NUMERROR, secPathEstimateLen))

        self.buffers["v"] = np.zeros((s.NUMSPEAKER, s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.buffers["xf"] = np.zeros((s.NUMREF, s.NUMSPEAKER, s.NUMERROR, s.SIMCHUNKSIZE+s.SIMBUFFER))

        self.diag.addNewDiagnostic("secpath", dia.ConstantEstimateNMSE(self.secPathFilt.ir, plotFrequency=s.PLOTFREQUENCY))
  
        self.metadata["auxNoiseSource"] = self.auxNoiseSource.__class__.__name__
        self.metadata["auxNoisePower"] = self.auxNoiseSource.power
        self.metadata["crossCorrForgetFactor"] = self.crossCorr.forgetFactor
        self.metadata["secPathEstimateLen"] = secPathEstimateLen

    def forwardPassImplement(self, numSamples): 
        super().forwardPassImplement(numSamples)
        v = self.auxNoiseSource.getSamples(numSamples)
        self.buffers["v"][:,self.idx:self.idx+numSamples] = v
        self.y[:,self.idx:self.idx+numSamples] += v

    
    def updateSPM(self):
        cc = np.zeros_like(self.secPathEstimate.ir)
        for i in range(self.updateIdx, self.idx):
            cc += self.e[None,:,i:i+1] * np.flip(self.buffers["v"][:,None,i-self.secPathEstLen+1:i+1], axis=-1)
        cc /= (self.idx - self.updateIdx)

        self.crossCorr.update(cc)
        self.secPathEstimate.ir = self.crossCorr.state / self.auxNoisePower
        self.secPathEstimateMD.ir[:,:,:] = self.secPathEstimate.ir[:,:,:]

        self.diag.saveBlockData("secpath", self.idx, self.secPathEstimate.ir)
 
    def updateFilter(self):
        self.buffers["xf"][:,:,:,self.updateIdx:self.idx] = np.transpose(self.secPathEstimateMD.process(
                                                        self.x[:,self.updateIdx:self.idx]), (2,0,1,3))
        grad = np.zeros_like(self.controlFilt.ir)
        for n in range(self.updateIdx, self.idx):
           Xf = np.flip(self.buffers["xf"][:,:,:,n-s.FILTLENGTH+1:n+1], axis=-1)
           grad += np.sum(Xf * self.e[None, None,:,n,None],axis=2) / (np.sum(Xf**2) + self.beta)

        self.controlFilt.ir -= grad * self.mu

        self.updateSPM()
        self.updateIdx = self.idx
        self.updated = True



class FastBlockFxLMSSPMTuningless(FastBlockFxLMS):
    """As we know the akf of the auxiliary noise, we should be able to get the 
        wiener solution without inverting the estimated akf matrix, which is usually the problem"""
    def __init__(self, mu, invForgetFactor, speakerRIR, blockSize):
        self.beta = 1e-3
        super().__init__(mu, self.beta, speakerRIR, blockSize)
        self.name = "SPM Aux Noise Freq Domain Wiener"
        self.auxNoisePower = 3
        self.auxNoiseSource = WhiteNoiseSource(power=self.auxNoisePower, numChannels=s.NUMSPEAKER)

        self.G = np.transpose(self.G, (0,2,1))
        self.secPathEstimate = FilterSum_Freqdomain(numIn=s.NUMSPEAKER, 
                            numOut=s.NUMERROR, irLen=blockSize)
        self.secPathEstimateMD = FilterMD_Freqdomain(dataDims=s.NUMREF, tf=np.zeros_like(self.secPathEstimate.tf))
        self.crossCorr = MovingAverage(1-invForgetFactor, (2*blockSize, s.NUMERROR, s.NUMSPEAKER),dtype=np.complex128)
        
        self.buffers["v"] = np.zeros((s.NUMSPEAKER, s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.buffers["f"] = np.zeros((s.NUMERROR, s.SIMBUFFER+s.SIMCHUNKSIZE))

        self.diag.addNewDiagnostic("secpath", dia.ConstantEstimateNMSE(self.G, plotFrequency=s.PLOTFREQUENCY))
        
        self.metadata["auxNoiseSource"] = self.auxNoiseSource.__class__.__name__
        self.metadata["auxNoisePower"] = self.auxNoiseSource.power
        self.metadata["crossCorrForgetFactor"] = self.crossCorr.forgetFactor

    def forwardPassImplement(self, numSamples): 
        super().forwardPassImplement(numSamples)
        v = self.auxNoiseSource.getSamples(numSamples)
        self.buffers["v"][:,self.idx:self.idx+numSamples] = v
        self.y[:,self.idx:self.idx+numSamples] += v

    
    def updateSPM(self):
        crossCorr = fdf.correlateEuclidianTT(self.e[:,self.updateIdx:self.idx], 
            self.buffers["v"][:,self.idx-2*self.blockSize:self.idx])
        crossCorr = fdf.fftWithTranspose(np.concatenate(
            (crossCorr,np.zeros_like(crossCorr)),axis=-1)) 

        # eBlock = np.concatenate((np.zeros_like(self.e[:,self.updateIdx:self.idx]),self.e[:,self.updateIdx:self.idx]),axis=-1)
        # E = np.fft.fft(eBlock, axis=-1).T[:,:,None]
        # V = np.fft.fft(self.buffers["v"][:,self.idx-2*self.blockSize:self.idx],axis=-1).T[:,:,None]
        # crossCorr = E @ np.transpose(V.conj(),(0,2,1)) 
        # crossCorr = np.fft.ifft(crossCorr, axis=0)
        # crossCorr[self.blockSize:,:,:] = 0
        # crossCorr = np.fft.fft(crossCorr, axis=0)
        self.crossCorr.update(crossCorr / self.blockSize)
        self.secPathEstimate.tf = self.crossCorr.state / self.auxNoisePower
        self.secPathEstimateMD.tf[...] = self.secPathEstimate.tf

        self.diag.saveBlockData("secpath", self.idx, self.secPathEstimateMD.tf)

    def updateFilter(self):
        assert (self.idx - self.updateIdx == self.blockSize)
        assert(self.updated == False)
        self.updateSPM()

        self.buffers["xf"][:,:,:,self.updateIdx:self.idx] = self.secPathEstimateMD.process(
            self.x[:,self.updateIdx:self.idx]
        )

        vf = self.secPathEstimate.process(self.buffers["v"][:,self.updateIdx:self.idx])
        self.buffers["f"][:,self.updateIdx:self.idx] = self.e[:,self.updateIdx:self.idx] - vf

        tdGrad = fdf.correlateSumTT(self.buffers["f"][None,None,:,self.updateIdx:self.idx], 
                    np.transpose(self.buffers["xf"][:,:,:,self.idx-(2*self.blockSize):self.idx],
                        (1,2,0,3)))

        grad = fdf.fftWithTranspose(np.concatenate(
            (tdGrad, np.zeros_like(tdGrad)), axis=-1))
        norm = 1 / (np.sum(self.buffers["xf"][:,:,:,self.updateIdx:self.idx]**2) + self.beta)
        self.controlFilt.tf -= self.mu * norm * grad
        self.updateIdx += self.blockSize
        self.updated = True


    def updateFilter__(self):
        assert (self.idx - self.updateIdx == self.blockSize)
        assert(self.updated == False)
        self.updateSPM()
        
        V = np.fft.fft(self.buffers["v"][:,self.idx-(2*self.blockSize):self.idx],axis=-1).T[:,:,None]
        Vf = self.secPathEstimate @ V
        vf = np.squeeze(np.fft.ifft(Vf,axis=0),axis=-1).T
        f = self.e[:,self.updateIdx:self.idx] - vf[:,self.blockSize:]
        
        self.F = np.fft.fft(np.concatenate((np.zeros((s.NUMERROR,self.blockSize)),f),axis=-1), 
                            axis=-1).T[:,:,None]

        grad = np.transpose(self.secPathEstimate.conj(),(0,2,1)) @ self.F @ np.transpose(self.X.conj(),(0,2,1))

        tdGrad = np.fft.ifft(grad, axis=0)
        tdGrad[self.blockSize:,:,:] = 0
        grad = np.fft.fft(tdGrad, axis=0)

        powerPerFreq = np.sum(np.abs(self.secPathEstimate)**2,axis=(1,2)) * np.sum(np.abs(self.X)**2,axis=(1,2))
        norm = 1 / (np.mean(powerPerFreq) + self.beta)
        self.H -= self.mu * norm * grad

        self.updateIdx += self.blockSize
        self.updated = True

    # def updateFilter_lazy(self):
    #     assert (self.idx - self.updateIdx == self.blockSize)
    #     assert(self.updated == False)

    #     self.updateSPM()
    #     self.E = np.fft.fft(self.e[:,self.updateIdx-self.blockSize:self.idx], axis=-1).T[:,:,None]
    #     self.V = np.fft.fft(self.buffers["v"][:,self.updateIdx-self.blockSize:self.idx], axis=-1).T[:,:,None]
    #     self.F = self.E - self.secPathEstimate @ self.V
    #     grad = np.transpose(self.secPathEstimate.conj(),(0,2,1)) @ self.F @ np.transpose(self.X.conj(),(0,2,1))

    #     # tdGrad = np.fft.ifft(grad, axis=0)
    #     # tdGrad[self.blockSize:,:] = 0
    #     # grad = np.fft.fft(tdGrad, axis=0)

    #     powerPerFreq = np.sum(np.abs(self.secPathEstimate)**2,axis=(1,2)) * np.sum(np.abs(self.X)**2,axis=(1,2))
    #     norm = 1 / (np.mean(powerPerFreq) + self.beta)
    #     #norm = 1 / (powerPerFreq[:,None,None] + self.beta)
    #     self.H -= self.mu * norm * grad

    #     self.updateIdx += self.blockSize
    #     self.updated = True






class FxLMSSPMEriksson(AdaptiveFilterFF):
    """Using fully sample by sample normalization
        Will then not correspond exactly to frequency domain implementations"""
    def __init__(self, mu, beta, speakerRIR, muSPM, secPathEstimateLen, auxNoisePower=3):
        super().__init__(mu, beta, speakerRIR)
        self.name = "FxLMS SPM Eriksson modern"
        self.muSPM = muSPM
        self.spmLen = secPathEstimateLen

        self.auxNoiseSource = WhiteNoiseSource(power=auxNoisePower, numChannels=s.NUMSPEAKER)
        #self.auxNoiseSource = GoldSequenceSource(11, power=10, numChannels=s.NUMSPEAKER)
        self.secPathEstimateMD = FilterMD_IntBuffer(dataDims=s.NUMREF, irLen=secPathEstimateLen, irDims=(s.NUMSPEAKER, s.NUMERROR))  

        self.secPathEstimate = NLMS(irLen=secPathEstimateLen, numIn=s.NUMSPEAKER, numOut=s.NUMERROR, 
                                    stepSize=muSPM, channelIndependentNorm=False)

        self.buffers["xf"] = np.zeros((s.NUMREF, s.NUMSPEAKER, s.NUMERROR, s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.buffers["v"] = np.zeros((s.NUMSPEAKER, s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.buffers["vEst"] = np.zeros((s.NUMERROR, s.SIMCHUNKSIZE+s.SIMBUFFER))

        self.diag.addNewDiagnostic("secpath", dia.ConstantEstimateNMSE(speakerRIR["error"], plotFrequency=s.PLOTFREQUENCY))

        self.metadata["auxNoiseSource"] = self.auxNoiseSource.__class__.__name__
        self.metadata["auxNoisePower"] = self.auxNoiseSource.power
        self.metadata["muSPM"] = muSPM
        self.metadata["secPathEstimateLen"] = secPathEstimateLen
        self.metadata["auxNoisePower"] = auxNoisePower
        

    def updateSPM(self):
        self.buffers["vEst"][:,self.updateIdx:self.idx] = self.secPathEstimate.process(
                                                            self.buffers["v"][:,self.updateIdx:self.idx])

        f = self.e[:,self.updateIdx:self.idx] - self.buffers["vEst"][:,self.updateIdx:self.idx]
        self.secPathEstimate.update(self.buffers["v"][:,self.updateIdx:self.idx], f)

        self.secPathEstimateMD.ir[:,:,:] = self.secPathEstimate.filt.ir[:,:,:]

        self.diag.saveBlockData("secpath", self.updateIdx, self.secPathEstimate.filt.ir)

    def updateFilter(self):
        self.buffers["xf"][:,:,:,self.updateIdx:self.idx] = np.transpose(self.secPathEstimateMD.process(
                                                        self.x[:,self.updateIdx:self.idx]), (2,0,1,3))
        grad = np.zeros_like(self.controlFilt.ir)
        for n in range(self.updateIdx, self.idx):
           Xf = np.flip(self.buffers["xf"][:,:,:,n-s.FILTLENGTH+1:n+1], axis=-1)
           grad += np.sum(Xf * self.e[None, None,:,n,None],axis=2) / (np.sum(Xf**2) + self.beta)

        self.controlFilt.ir -= grad * self.mu
        self.updateSPM()
        self.updateIdx = self.idx

        
    def forwardPassImplement(self, numSamples): 
        y = self.controlFilt.process(self.x[:,self.idx:self.idx+numSamples])
        v = self.auxNoiseSource.getSamples(numSamples)
        self.buffers["v"][:,self.idx:self.idx+numSamples] = v
        self.y[:,self.idx:self.idx+numSamples] = v + y

class FxLMSSPMEriksson_Gold(FxLMSSPMEriksson):
    def __init__(self, mu, beta, speakerRIR, muSPM, secPathEstimateLen, auxNoisePower=3):   
        super().__init__(mu, beta, speakerRIR, muSPM, secPathEstimateLen, auxNoisePower=auxNoisePower)
        self.name = "FxLMS SPM Eriksson Gold Code"
        self.auxNoiseSource = GoldSequenceSource(11, power=auxNoisePower, numChannels=s.NUMSPEAKER)





class FreqAuxNoiseFxLMS(ConstrainedFastBlockFxLMS):
    """USING FLAWED METHOD FROM CONSTRAINEDFASTBLOCKFXLMS. NOT TRULY LINEAR CONVOLUTIONS"""
    def __init__(self, mu, muSPM, speakerFilters, blockSize, lowFreqLim=0, highFreqLim=s.SAMPLERATE/2):
        self.beta = 1e-3
        super().__init__(mu, self.beta, speakerFilters, blockSize)
        self.name = "SPM Aux Noise Freq Domain - linear conv"
        self.muSPM = muSPM

        #self.auxNoiseSource = GoldSequenceSource(11, power=10, numChannels=s.NUMSPEAKER)
        self.auxNoiseSource = WhiteNoiseSource(power=3, numChannels=s.NUMSPEAKER)

        self.secPathEstimate = FastBlockNLMS(blockSize=blockSize, numIn=s.NUMSPEAKER, numOut=s.NUMERROR, 
                                            stepSize=muSPM, freqIndepNorm=False)
        self.G = np.transpose(self.G, (0,2,1))
        #self.G = np.transpose(np.fft.fft(np.concatenate(
        #    (np.zeros_like(speakerFilters["error"]),speakerFilters["error"]),axis=-1),axis=-1),(2,1,0))
        #self.secPathEstimate = np.zeros((2*blockSize, s.NUMERROR, s.NUMSPEAKER), dtype=np.complex128)
        #self.secPathEstimate.setIR(0.5*self.G)# + np.random.normal(scale=0.0001, size=self.G.shape))

        self.buffers["v"] = np.zeros((s.NUMSPEAKER, s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.buffers["f"] = np.zeros((s.NUMERROR, s.SIMCHUNKSIZE+s.SIMBUFFER))

        self.diag.addNewDiagnostic("secpath", dia.ConstantEstimateNMSE(self.G, plotFrequency=s.PLOTFREQUENCY))
        self.diag.addNewDiagnostic("secpathselectedfrequencies", dia.ConstantEstimateNMSESelectedFrequencies(
            self.G, lowFreq=lowFreqLim, highFreq=highFreqLim, sampleRate=s.SAMPLERATE, plotFrequency=s.PLOTFREQUENCY))
        self.diag.addNewDiagnostic("secpathphase", dia.ConstantEstimatePhaseDifference(self.G, plotFrequency=s.PLOTFREQUENCY))

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

    def updateSPM(self):
        vf = self.secPathEstimate.process(np.concatenate((np.zeros((s.NUMSPEAKER,self.blockSize)),
                                                        self.buffers["v"][:,self.updateIdx:self.idx]), axis=-1))

        self.buffers["f"][:,self.updateIdx:self.idx] = self.e[:,self.updateIdx:self.idx] - vf
        self.secPathEstimate.update(self.buffers["v"][:,self.updateIdx-self.blockSize:self.idx], 
                                    self.buffers["f"][:,self.updateIdx:self.idx])


        self.diag.saveBlockData("secpath", self.idx, self.secPathEstimate.ir)
        self.diag.saveBlockData("secpathselectedfrequencies", self.idx, self.secPathEstimate.ir)
        self.diag.saveBlockData("secpathphase", self.idx, self.secPathEstimate.ir)

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


class FreqAuxNoiseFxLMS2(FastBlockFxLMS):
    """FLAWED"""
    def __init__(self, mu, muSPM, speakerFilters, blockSize):
        self.beta = 1e-3
        super().__init__(mu, self.beta, speakerFilters, blockSize)
        self.name = "SPM Aux Noise Freq Domain - linear conv"
        self.muSPM = muSPM
        self.diag.addNewDiagnostic("secpath", dia.ConstantEstimateNMSE(self.secPathEstimate, plotFrequency=s.PLOTFREQUENCY))
        #self.auxNoiseSource = GoldSequenceSource(11, power=10, numChannels=s.NUMSPEAKER)
        self.auxNoiseSource = WhiteNoiseSource(power=3, numChannels=s.NUMSPEAKER)

        self.secPathEstimate = FastBlockNLMS(blockSize=blockSize, numIn=s.NUMSPEAKER, numOut=s.NUMERROR, 
                                            stepSize=muSPM, freqIndepNorm=False)
        #self.G = np.transpose(np.fft.fft(np.concatenate(
        #    (np.zeros_like(speakerFilters["error"]),speakerFilters["error"]),axis=-1),axis=-1),(2,1,0))
        #self.secPathEstimate = np.zeros((2*blockSize, s.NUMERROR, s.NUMSPEAKER), dtype=np.complex128)
        #self.secPathEstimate.setIR(0.5*self.G)# + np.random.normal(scale=0.0001, size=self.G.shape))

        self.buffers["v"] = np.zeros((s.NUMSPEAKER, s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.buffers["f"] = np.zeros((s.NUMERROR, s.SIMCHUNKSIZE+s.SIMBUFFER))

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

    def updateSPM(self):
        vf = self.secPathEstimate.process(np.concatenate((np.zeros((s.NUMSPEAKER,self.blockSize)),
                                                        self.buffers["v"][:,self.updateIdx:self.idx]), axis=-1))

        self.buffers["f"][:,self.updateIdx:self.idx] = self.e[:,self.updateIdx:self.idx] - vf
        self.secPathEstimate.update(self.buffers["v"][:,self.updateIdx-self.blockSize:self.idx], 
                                    self.buffers["f"][:,self.updateIdx:self.idx])

        self.diag.saveBlockData("secpath", self.idx, self.secPathEstimate.ir)

    def updateFilter(self):
        assert (self.idx - self.updateIdx == self.blockSize)
        assert(self.updated == False)
        self.updateSPM()

        self.X = np.fft.fft(self.x[:,self.idx-(2*self.blockSize):self.idx],axis=-1).T[:,:,None]
        xf = self.secPathEstimate.ir[:,None,:,:] * self.X[:,:,None,:] 

        xf = np.fft.ifft(xf, axis=0)
        xf[:self.blockSize,:,:,:] = 0
        xf = np.fft.fft(xf,axis=0)
        
        self.F = np.fft.fft(np.concatenate((np.zeros((s.NUMERROR,self.blockSize)),
                                            self.buffers["f"][:,self.updateIdx:self.idx]),axis=-1), 
                                            axis=-1).T[:,:,None]

        grad = np.squeeze(np.transpose(xf.conj(),(0,3,1,2)) @ self.F[:,None,:,:], axis=-1)

        tdGrad = np.fft.ifft(grad, axis=0)
        tdGrad[self.blockSize:,:,:] = 0
        grad = np.fft.fft(tdGrad, axis=0)

        powerPerFreq = np.sum(np.abs(xf)**2,axis=(1,2,3))
        norm = 1 / (np.mean(powerPerFreq) + self.beta)
        #print("New FD Norm", norm)
        self.H -= self.mu * norm * grad

        self.updateIdx += self.blockSize
        self.updated = True


class FxLMS_SPM_Yuxue(AdaptiveFilterFF):
    """Implementation of proposed method from:
        Online secondary path modeling method
        with auxiliary noise power scheduling
        strategy for multi-channel adaptive
        active noise control system"""
    def __init__(self, mu, beta, speakerFilters, minSecPathStepSize, secPathEstimateLen):
        super().__init__(mu, beta, speakerFilters)
        self.name = "FxLMS SPM Yuxue"
        self.muSecMin = minSecPathStepSize
        self.spmLen = secPathEstimateLen

        self.auxNoiseSource = GoldSequenceSource(11, power=1, numChannels=s.NUMSPEAKER)

        self.secPathEstimateMD = FilterMD_IntBuffer(dataDims=s.NUMREF, 
                                                    irLen=secPathEstimateLen, 
                                                    irDims=(s.NUMSPEAKER, s.NUMERROR))  
        self.secPathEstimate = NLMS(numIn=s.NUMSPEAKER, 
                                    numOut=s.NUMERROR, 
                                    irLen=secPathEstimateLen, 
                                    stepSize=self.muSecMin, 
                                    filterType=FilterIndividualInputs)

        #initialSecPathEstimate = np.random.normal(scale=0.0001, size=self.secPathEstimate.filt.ir.shape)
        #commonLength = np.min((secPathEstimateLen,speakerFilters["error"].shape[-1]))
        #initialSecPathEstimate[:,:,0:commonLength] = 0.5 * speakerFilters["error"][:,:,0:commonLength]
        
        #self.secPathEstimateMD.setIR(initialSecPathEstimate)
        #self.secPathEstimate.setIR(initialSecPathEstimate)

        self.buffers["xf"] = np.zeros((s.NUMREF, s.NUMSPEAKER, s.NUMERROR, s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.buffers["v"] = np.zeros((s.NUMSPEAKER, s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.buffers["vEst"] = np.zeros((s.NUMSPEAKER, s.NUMERROR, s.SIMCHUNKSIZE+s.SIMBUFFER))

        self.buffers["eps"] = np.zeros((s.NUMSPEAKER, s.NUMERROR, s.SIMCHUNKSIZE+s.SIMBUFFER))

        self.Pf = SinglePoleLowPass(0.9, (s.NUMERROR,))
        self.Peps = SinglePoleLowPass(0.9, (s.NUMSPEAKER, s.NUMERROR))
        self.Pv = SinglePoleLowPass(0.9, (s.NUMSPEAKER,))

        self.diag.addNewDiagnostic("secpath", dia.ConstantEstimateNMSE(speakerFilters["error"], plotFrequency=s.PLOTFREQUENCY))
        self.diag.addNewDiagnostic("mu_spm", dia.RecordVector(s.NUMSPEAKER*s.NUMERROR ,plotFrequency=s.PLOTFREQUENCY))
        self.diag.addNewDiagnostic("aux_noise_power", dia.RecordVector(s.NUMSPEAKER ,plotFrequency=s.PLOTFREQUENCY))

        self.metadata["auxNoiseSource"] = self.auxNoiseSource.__class__.__name__
        self.metadata["minSecPathStepSize"] = minSecPathStepSize
        self.metadata["secPathEstimateLen"] = secPathEstimateLen
        self.metadata["forgetFactorPf"] = self.Pf.forgetFactor
        self.metadata["forgetFactorPeps"] = self.Peps.forgetFactor
        self.metadata["forgetFactorPv"] = self.Pv.forgetFactor


    def calcEpsilon(self):
        self.buffers["eps"][:,:,self.updateIdx:self.idx] = self.e[None,:,self.updateIdx:self.idx]
        for j in range(self.buffers["eps"].shape[0]):
            for i in range(self.buffers["eps"].shape[0]):
                if j != i:
                    self.buffers["eps"][j,:,self.updateIdx:self.idx] -= self.buffers["vEst"][i,:,self.updateIdx:self.idx]

    def updateSPM(self):
        self.buffers["vEst"][:,:,self.updateIdx:self.idx] = self.secPathEstimate.process(
                                                            self.buffers["v"][:,self.updateIdx:self.idx])

        f = self.e[:,self.updateIdx:self.idx] - np.sum(self.buffers["vEst"][:,:,self.updateIdx:self.idx],axis=0)
        self.calcEpsilon()

        self.Pf.update(np.mean(f**2,axis=-1))
        self.Pv.update(np.mean(self.buffers["v"][:,self.updateIdx:self.idx]**2,axis=-1))
        self.Peps.update(np.mean(self.buffers["eps"][:,:,self.updateIdx:self.idx]**2,axis=-1))

        newAuxNoisePower = np.sum(self.Pf.state) / np.sum(self.Peps.state,axis=-1)
        self.auxNoiseSource.setPower(newAuxNoisePower)
        self.secPathEstimate.mu = self.muSecMin * self.Pv.state[:,None,None] / self.Peps.state[:,:,None]

        self.secPathEstimate.update(self.buffers["v"][:,self.updateIdx:self.idx], f)
        self.secPathEstimateMD.ir[:,:,:] = self.secPathEstimate.filt.ir[:,:,:]

        self.diag.saveBlockData("secpath", self.updateIdx, self.secPathEstimate.filt.ir)
        self.diag.saveBlockData("mu_spm", self.updateIdx, self.secPathEstimate.mu.ravel())
        self.diag.saveBlockData("aux_noise_power", self.updateIdx, newAuxNoisePower)
        
    def updateFilter(self):
        self.buffers["xf"][:,:,:,self.updateIdx:self.idx] = np.transpose(self.secPathEstimateMD.process(
                                                        self.x[:,self.updateIdx:self.idx]), (2,0,1,3))
        grad = np.zeros_like(self.controlFilt.ir)
        for n in range(self.updateIdx, self.idx):
           Xf = np.flip(self.buffers["xf"][:,:,:,n-s.FILTLENGTH+1:n+1], axis=-1)
           grad += np.sum(Xf * self.e[None, None,:,n,None],axis=2) / (np.sum(Xf**2) + self.beta)

        self.controlFilt.ir -= grad * self.mu
        self.updateSPM()
        self.updateIdx = self.idx

        
    def forwardPassImplement(self, numSamples): 
        y = self.controlFilt.process(self.x[:,self.idx:self.idx+numSamples])
        v = self.auxNoiseSource.getSamples(numSamples)
        self.buffers["v"][:,self.idx:self.idx+numSamples] = v

        self.y[:,self.idx:self.idx+numSamples] = v + y












class FxLMSSPMZhang(AdaptiveFilterFF):
    """IS NOT FULLY FUNCTIONAL"""
    def __init__(self, mu, beta, speakerFilters, muSPM, secPathEstimateLen):
        super().__init__(mu, beta, speakerFilters)
        self.name = "FxLMS SPM Zhang"
        self.muSPM = muSPM
        self.spmLen = secPathEstimateLen
        self.auxNoiseLevel = -5
        
        #initialSecPathEstimate = np.random.normal(scale=0.01, size=speakerFilters["error"].shape) + 0.8 * speakerFilters["error"]

        self.secPathEstimateMD = FilterMD_IntBuffer(dataDims=s.NUMREF, irLen=secPathEstimateLen, irDims=(s.NUMSPEAKER, s.NUMERROR))  
        #self.secPathEstimateMD.ir = initialSecPathEstimate
        self.secPathEstimate = FilterSum_IntBuffer(irLen=secPathEstimateLen, numIn=s.NUMSPEAKER, numOut=s.NUMERROR)
        #self.secPathEstimate.ir = initialSecPathEstimate

        self.distFiltLen = secPathEstimateLen
        self.disturbanceFilter = FilterSum_IntBuffer(irLen=self.distFiltLen, numIn=s.NUMREF, numOut=s.NUMERROR)

        self.buffers["xf"] = np.zeros((s.NUMREF, s.NUMSPEAKER, s.NUMERROR, s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.buffers["v"] = np.zeros((s.NUMSPEAKER, s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.buffers["u"] = np.zeros((s.NUMERROR, s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.buffers["vEst"] = np.zeros((s.NUMERROR, s.SIMCHUNKSIZE+s.SIMBUFFER))

    def prepare(self):
        pass

    def updateDisturbanceFilter(self, g):
        grad = np.zeros_like(self.disturbanceFilter.ir)
        numSamples = self.idx-self.updateIdx
        for i in range(numSamples):
            grad += g[None,:,i:i+1] * np.flip(self.x[:,None,self.updateIdx+i-self.distFiltLen:self.updateIdx+i],axis=-1)

        self.disturbanceFilter.ir += self.muSPM * grad * 0



    def updateSPM(self):
        self.buffers["vEst"][:,self.updateIdx:self.idx] = self.secPathEstimate.process(
                                                            self.buffers["v"][:,self.updateIdx:self.idx])

        g = self.e[:,self.updateIdx:self.idx] - self.buffers["vEst"][:,self.updateIdx:self.idx]
        self.updateDisturbanceFilter(g)

        self.buffers["u"][:,self.updateIdx:self.idx] = self.disturbanceFilter.process(self.x[:,self.updateIdx:self.idx])
        f = g - self.buffers["u"][:,self.updateIdx:self.idx]

        grad = np.zeros_like(self.secPathEstimate.ir)
        numSamples = self.idx-self.updateIdx
        for i in range(numSamples):
            grad += np.flip(self.buffers["v"][:,None,self.updateIdx+i-self.spmLen+1:self.updateIdx+i+1], axis=-1) * f[None,:,i:i+1]
            
        self.secPathEstimateMD.ir += self.muSPM * grad
        self.secPathEstimate.ir[:,:,:] = self.secPathEstimateMD.ir[:,:,:]


    def updateFilter(self):
        self.buffers["xf"][:,:,:,self.updateIdx:self.idx] = np.transpose(self.secPathEstimateMD.process(
                                                        self.x[:,self.updateIdx:self.idx]), (2,0,1,3))
        grad = np.zeros_like(self.controlFilt.ir)
        for n in range(self.updateIdx, self.idx):
           Xf = np.flip(self.buffers["xf"][:,:,:,n-s.FILTLENGTH+1:n+1], axis=-1)
           grad += np.sum(Xf * self.e[None, None,:,n,None],axis=2) / (np.sum(Xf**2) + self.beta)

        self.controlFilt.ir -= grad * self.mu
        self.updateSPM()
        self.updateIdx = self.idx

        
    def forwardPassImplement(self, numSamples): 
        y = self.controlFilt.process(self.x[:,self.idx:self.idx+numSamples])

        v = getWhiteNoiseAtSNR(y, y.shape, self.auxNoiseLevel)
        self.buffers["v"][:,self.idx:self.idx+numSamples] = v
        
        self.y[:,self.idx:self.idx+numSamples] = v + y
