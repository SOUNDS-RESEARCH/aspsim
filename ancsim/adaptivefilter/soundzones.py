import numpy as np
import scipy.linalg as linalg
import scipy.signal as spsig
from abc import abstractmethod
import ancsim.utilities as util

import matplotlib.pyplot as plt

import ancsim.adaptivefilter.base as base
import ancsim.signal.filterclasses as fc
import ancsim.adaptivefilter.diagnostics as diag
import ancsim.soundfield.kernelinterpolation as ki
import ancsim.signal.filterdesign as fd
import ancsim.integration.montecarlo as mc


class SoundzoneFIR(base.AudioProcessor):
    def __init__(self, sim_info, arrays, blockSize, src, ctrlFiltLen):
        super().__init__(sim_info, arrays, blockSize)
        self.name = "Abstract Soundzone Processor"
        self.src = src
        self.ctrlFiltLen = ctrlFiltLen

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

        if "region_bright" in self.arrays:
            self.createNewBuffer("reg_error_bright", self.arrays["region_bright"].num)
            self.createNewBuffer("reg_desired", self.arrays["region_bright"].num)
            self.desiredFilterSpatial = fc.createFilter(ir=self.arrays.paths["speaker"]["region_bright"], broadcastDim=1, sumOverInput=False)
            self.diag.addNewDiagnostic("spatial_bright_zone_power", diag.SignalPower(self.sim_info, "region_bright", blockSize=self.blockSize))
            self.diag.addNewDiagnostic("spatial_bright_zone_error", diag.SignalPowerRatio(self.sim_info, "reg_error_bright", "reg_desired", blockSize=self.blockSize))
            if "region_dark" in self.arrays:
                self.diag.addNewDiagnostic("spatial_acoustic_contrast", diag.SignalPowerRatio(self.sim_info, "region_bright", "region_dark", blockSize=self.blockSize))
                self.diag.addNewDiagnostic("spatial_dark_zone_power", diag.SignalPower(self.sim_info, "region_dark", blockSize=self.blockSize))
                
        self.desiredFilter = fc.createFilter(ir=self.pathsBright, broadcastDim=1, sumOverInput=False)

        self.metadata["source type"] = self.src.__class__.__name__
        self.metadata["source"] = self.src.metadata
        self.metadata["control filter length"] = self.ctrlFiltLen

    @abstractmethod
    def calcControlFilter(self):
        pass

    def process(self, numSamples):
        x = self.src.getSamples(self.blockSize)
        self.sig["orig_sig"][:,self.idx:self.idx+self.blockSize] = x
        self.sig["speaker"][:,self.idx:self.idx+self.blockSize] = self.controlFilter.process(x)


        self.sig["desired"][:,self.idx:self.idx+self.blockSize] = np.squeeze(self.desiredFilter.process(x), axis=-2)[0,...]
        self.sig["error_bright"][:,self.idx-self.blockSize:self.idx] = self.sig["mic_bright"][:,self.idx-self.blockSize:self.idx] - \
                                                                        self.sig["desired"][:,self.idx-self.blockSize:self.idx]

        if "region_bright" in self.arrays:
            self.sig["reg_desired"][:,self.idx:self.idx+self.blockSize] = np.squeeze(self.desiredFilterSpatial.process(x), axis=-2)[0,...]
            self.sig["reg_error_bright"][:,self.idx-self.blockSize:self.idx] = self.sig["region_bright"][:,self.idx-self.blockSize:self.idx] - \
                                                                                self.sig["reg_desired"][:,self.idx-self.blockSize:self.idx]
            


class PressureMatchingWhiteNoise(SoundzoneFIR):
    def __init__(self, sim_info, arrays, blockSize, src, ctrlFiltLen):
        super().__init__(sim_info, arrays, blockSize, src, ctrlFiltLen)
        self.name = "Pressure Matching White Noise"
        self.controlFilter = fc.createFilter(ir=self.calcControlFilter())

    def calcControlFilter(self):
        Rb = self.spatialCorrMatrix(self.pathsBright)
        testCorrMatrix(Rb, "Rb")
        Rd = self.spatialCorrMatrix(self.pathsDark)
        testCorrMatrix(Rd, "Rd")
        R = Rb + Rd
        testCorrMatrix(R, "R")

        rb = self.spatialCorrVector(self.pathsBright)
        optFilt = linalg.solve(R, rb, assume_a="sym")
        optFilt = optFilt.reshape((1, self.ctrlFiltLen,self.numSpeaker))
        optFilt = np.moveaxis(optFilt, 1, 2)
        return optFilt
    
    def spatialCorrMatrix(self, paths):
        extendedPath = np.pad(paths, ((0,0),(0,0),(self.ctrlFiltLen,self.ctrlFiltLen-1)))
        extLen = extendedPath.shape[-1]
        
        R = np.zeros((self.ctrlFiltLen*self.numSpeaker, self.ctrlFiltLen*self.numSpeaker))
        for i in range(self.ctrlFiltLen):
            for j in range(self.ctrlFiltLen):
                R[self.numSpeaker*i:self.numSpeaker*(i+1),
                  self.numSpeaker*j:self.numSpeaker*(j+1)] = \
                        np.sum(extendedPath[:,None,:,self.ctrlFiltLen-j:extLen-j] * \
                                extendedPath[None,:,:,self.ctrlFiltLen-i:extLen-i], axis=(-2,-1))
        R /= paths.shape[1]
        return R

    def spatialCorrVector(self, paths):
        d = paths[0,:,:]
        #d = np.sum(paths,axis=0)

        extendedPath = np.pad(paths, ((0,0), (0,0), (self.ctrlFiltLen,0)))
        extLen = extendedPath.shape[-1]
        
        r = np.zeros((self.numSpeaker*self.ctrlFiltLen,1))
        for i in range(self.ctrlFiltLen):
            r[self.numSpeaker*i:self.numSpeaker*(i+1),0] = \
                np.sum(extendedPath[:,:,self.ctrlFiltLen-i:extLen-i] * d[None,:,:], axis=(-2,-1))
        r /= self.numMicBright
        return r


class PressureMatching(SoundzoneFIR):
    def __init__(self, sim_info, arrays, blockSize, src, ctrlFiltLen, corrAverageLen):
        super().__init__(sim_info, arrays, blockSize, src, ctrlFiltLen)
        self.name = "Pressure Matching"
        self.corrAverageLen = corrAverageLen
        
        self.controlFilter = fc.createFilter(ir=self.calcControlFilter())

        self.metadata["correlation average length"] = self.corrAverageLen

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
        testCorrMatrix(Rb, "Rb")
        Rd = self.spatialCorrMatrix(xd)
        testCorrMatrix(Rd, "Rd")
        R = Rb + Rd #+ 1e-2*np.eye(self.numSpeaker*self.ctrlFiltLen)
        testCorrMatrix(R, "R")

        d = xb[0,:,:]#np.sum(xb, axis=0)
        rb = self.spatialCorrVector(xb, d)
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
        rb /= self.numMicBright
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
        R /= filteredSignal.shape[-2]
        return R
                


class VAST(SoundzoneFIR):
    def __init__(self, sim_info, arrays, blockSize, src, ctrlFiltLen, corrAverageLen, rank, mu, regRd):
        super().__init__(sim_info, arrays, blockSize, src, ctrlFiltLen)
        self.name = "VAST"
        self.corrAverageLen = corrAverageLen
        self.rank = rank
        self.mu = mu
        self.regRd = regRd

        self.matSize = self.numSpeaker*self.ctrlFiltLen
        assert 1 <= rank <= self.matSize
        self.controlFilter = fc.createFilter(ir=self.calcControlFilter())


        self.metadata["rank"] = self.rank
        self.metadata["mu"] = self.mu
        self.metadata["regularization for Rd"] = self.regRd
        self.metadata["correlation average length"] = self.corrAverageLen
    
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

        print("vast")
        print(np.abs(np.mean(Rb)))
        print(np.abs(np.mean(Rd)))

        testCorrMatrix(Rb, "Rb")
        testCorrMatrix(Rd, "Rd")
        
        d = np.sum(xb, axis=0)
        rb = self.spatialCorrVector(xb, d)

        Rd += self.regRd*np.eye(self.matSize)
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
    def __init__(self, sim_info, arrays, blockSize, 
                src, ctrlFiltLen, corrAverageLen, rank, mu, regRd,
                brightRegion, darkRegion, kernelReg, kiFiltLen, integrationSamples):
        super().__init__(sim_info, arrays, blockSize, src, ctrlFiltLen)
        self.name = "KI-VAST"
        self.corrAverageLen = corrAverageLen
        self.rank = rank
        self.mu = mu
        self.regRd = regRd
        self.brightRegion = brightRegion
        self.darkRegion = darkRegion
        self.kernelReg = kernelReg
        self.kiFiltLen = kiFiltLen
        self.integrationSamples = integrationSamples

        self.matSize = self.numSpeaker*self.ctrlFiltLen
        assert 1 <= rank <= self.matSize

        self.kiMatBright, kiTruncErrorBright = self.computeKIFilter(self.arrays["mic_bright"].pos, 
                                                self.brightRegion, 
                                                self.kiFiltLen, 
                                                self.kernelReg,
                                                4096)
        self.kiMatDark, kiTruncErrorDark = self.computeKIFilter(self.arrays["mic_dark"].pos, 
                                                self.darkRegion, 
                                                self.kiFiltLen, 
                                                self.kernelReg,
                                                4096)

        self.controlFilter = fc.createFilter(ir=self.calcControlFilter())

        self.metadata["correlation average length"] = self.corrAverageLen
        self.metadata["rank"] = self.rank
        self.metadata["mu"] = self.mu
        self.metadata["regularization for Rd"] = self.regRd
        self.metadata["kernel regularization"] = self.kernelReg
        self.metadata["KI filter length"] = self.kiFiltLen
        self.metadata["KI interpolation samples"] = self.integrationSamples
        self.metadata["KI truncation error bright zone"] = kiTruncErrorBright
        self.metadata["KI truncation error dark zone"] = kiTruncErrorDark


    def computeKIFilter(self, micPos, region, irLen, kernelReg, numFreq):
        """return filter ordered [numMics1, kiLen1, numMics2, kiLen2]
            The matrix as written in the paper would have been [numMics1*kiLen1, numMics2*kiLen2]
            It is only not reshaped for ease of use"""
        assert irLen % 2 == 1
        assert numFreq % 2 == 0

        freqs = fd.getFrequencyValues(numFreq, self.sim_info.samplerate)
        waveNum = 2 * np.pi * freqs / self.sim_info.c

        def genKernelWeighting(sampledPos):
            numSamples = sampledPos.shape[0]
            freqFilt = ki.getKRRParameters(ki.kernelHelmholtz3d, kernelReg, sampledPos, micPos, waveNum)
            freqFilt = fd.insertNegativeFrequencies(freqFilt, even=True)
            ir,_ = fd.firFromFreqsWindow(freqFilt, irLen)
            ir = ir.reshape(numSamples,-1)
            kiMat = ir[:,:,None] * ir[:,None,:]
            #kiMat = ir[:,:,:,None,None] * ir[:,None,None,:,:]
            return np.moveaxis(kiMat, 0, -1)

        kiMatrix = mc.integrate(genKernelWeighting, region.sample_points, self.integrationSamples, region.volume)
        kiMatrix = kiMatrix / region.volume

        sampledPos = region.sample_points(50)
        freqFilt = ki.getKRRParameters(ki.kernelHelmholtz3d, kernelReg, sampledPos, micPos, waveNum)
        freqFilt = fd.insertNegativeFrequencies(freqFilt, even=True)
        _,truncError = fd.firFromFreqsWindow(freqFilt, irLen)

        return kiMatrix, truncError

    # def spatialCorrMatrix(self, filtSig, kiMat):
    #     numMic = kiMat.shape[0]
    #     kiMatReshaped = kiMat.reshape((numMic, self.kiFiltLen, -1))
    #     kiMatReshaped = np.moveaxis(kiMatReshaped, 2,1)
    #     kiFilt = fc.createFilter(ir=kiMatReshaped)
    #     for l1 in range(self.numSpeaker):
    #         for l2 in range(self.numSpeaker):
    #             kiSig = kiFilt.process(filtSig[l1,:,:])
    #             kiSig = kiSig.reshape(numMic, self.kiFiltLen, -1)
        
    #@util.measure("pedantic")
    # def spatialCorrMatrix_pedantic(self, origSig, acousticPath):
    #     numMic = acousticPath.shape[1]
    #     pathLen = acousticPath.shape[-1]
    #     margin = pathLen + self.ctrlFiltLen
    #     H = np.moveaxis(acousticPath, 1, 0).reshape(numMic,-1)

    #     R = np.zeros((self.numSpeaker*self.ctrlFiltLen, self.numSpeaker*self.ctrlFiltLen))
    #     X_block = np.zeros((self.numSpeaker * pathLen, self.numSpeaker * self.ctrlFiltLen))
    #     for n in range(margin, self.corrAverageLen+margin):
    #         X_block.fill(0)
    #         #print(n)
    #         col = np.flip(origSig[0,n-pathLen:n])
    #         row = np.flip(origSig[0,n-self.ctrlFiltLen:n])
    #         X = linalg.toeplitz(col, row)
    #         for k in range(self.numSpeaker):
    #             X_block[k*pathLen:(k+1)*pathLen, k*self.ctrlFiltLen:(k+1)*self.ctrlFiltLen] = X
    #         G = H @ X_block
    #         R += G.T @ G
    #     R = R / self.corrAverageLen
    #     return R


    # def spatialCorrVector_pedantic(self, origSig, acousticPath):
    #     numMic = acousticPath.shape[1]
    #     pathLen = acousticPath.shape[-1]
    #     margin = pathLen + self.ctrlFiltLen
    #     H = np.moveaxis(acousticPath, 1, 0).reshape(numMic,-1)
    #     i = np.eye(self.ctrlFiltLen)[:,:1]
    #     i = np.concatenate((i,)*self.numSpeaker, axis=0)

    #     r = np.zeros((self.numSpeaker*self.ctrlFiltLen, 1))
    #     X_block = np.zeros((self.numSpeaker * pathLen, self.numSpeaker * self.ctrlFiltLen))
    #     for n in range(margin, self.corrAverageLen+margin):
    #         X_block.fill(0)
    #         #print(n)
    #         col = np.flip(origSig[0,n-pathLen:n])
    #         row = np.flip(origSig[0,n-self.ctrlFiltLen:n])
    #         X = linalg.toeplitz(col, row)
    #         for k in range(self.numSpeaker):
    #             X_block[k*pathLen:(k+1)*pathLen, k*self.ctrlFiltLen:(k+1)*self.ctrlFiltLen] = X
    #         G = H @ X_block
    #         #plt.matshow(G)
    #         #plt.show()
    #         r += G.T @ G @ i
    #         #plt.matshow(R)
    #         #plt.show()
    #     r = r / self.corrAverageLen
    #     return r
    def acousticPathMatrix(self, paths, combinedFiltLen):
        numSpeaker = paths.shape[0]
        numMic = paths.shape[1]
        pathLen = paths.shape[2]
        otherFiltLen = combinedFiltLen - pathLen + 1

        H = np.zeros((combinedFiltLen*numSpeaker, numMic*otherFiltLen))
        for m in range(numMic):
            for l in range(numSpeaker):
                H_ml = linalg.toeplitz(np.pad(paths[l,m,:], (0,otherFiltLen-1)), 
                                    np.pad(paths[l,m,0:1],(0,otherFiltLen-1)))
                H[l*combinedFiltLen:(l+1)*combinedFiltLen, 
                    m*otherFiltLen:(m+1)*otherFiltLen] = H_ml
        return H

    @util.measure("sqrt")
    def spatialCorrMatrix_sqrt(self, origSig, acousticPath, kiMat):
        numMic = acousticPath.shape[1]
        pathLen = acousticPath.shape[2]
        combinedFiltLen = pathLen + self.kiFiltLen - 1
        
        kiSqrt = np.real(linalg.sqrtm(kiMat))
        H = self.acousticPathMatrix(acousticPath, combinedFiltLen)
        
        B = H @ kiSqrt
        R = np.zeros((self.numSpeaker*self.ctrlFiltLen, self.numSpeaker*self.ctrlFiltLen))
        X_block = np.zeros((self.numSpeaker * combinedFiltLen, self.numSpeaker * self.ctrlFiltLen))
        margin = combinedFiltLen + self.ctrlFiltLen
        for n in range(margin, self.corrAverageLen+margin):
            #print(n)
            X = linalg.toeplitz(np.flip(origSig[0,n-combinedFiltLen:n]), np.flip(origSig[0,n-self.ctrlFiltLen:n]))
            for k in range(self.numSpeaker):
                X_block[k*combinedFiltLen:(k+1)*combinedFiltLen, k*self.ctrlFiltLen:(k+1)*self.ctrlFiltLen] = X
            C = X_block.T @ B
            R += C @ C.T
        R = R / self.corrAverageLen
        return R

    @util.measure("reg")
    def spatialCorrMatrix(self, origSig, acousticPath, kiMat):
        numMic = acousticPath.shape[1]
        pathLen = acousticPath.shape[2]
        combinedFiltLen = pathLen + self.kiFiltLen - 1
        H = self.acousticPathMatrix(acousticPath, combinedFiltLen)
        
        B = H @ kiMat @ H.T
        R = np.zeros((self.numSpeaker*self.ctrlFiltLen, self.numSpeaker*self.ctrlFiltLen))
        X_block = np.zeros((self.numSpeaker * combinedFiltLen, self.numSpeaker * self.ctrlFiltLen))
        margin = combinedFiltLen + self.ctrlFiltLen
        for n in range(margin, self.corrAverageLen+margin):
            #print(n)
            X = linalg.toeplitz(np.flip(origSig[0,n-combinedFiltLen:n]), np.flip(origSig[0,n-self.ctrlFiltLen:n]))
            for k in range(self.numSpeaker):
                X_block[k*combinedFiltLen:(k+1)*combinedFiltLen, k*self.ctrlFiltLen:(k+1)*self.ctrlFiltLen] = X
            R += X_block.T @ B @ X_block
        R = R / self.corrAverageLen
        return R

    def spatialCorrVector(self, origSig, acousticPath, kiMat, desired):
        #numMic = acousticPath.shape[1]
        pathLen = acousticPath.shape[2]
        combinedFiltLen = pathLen + self.kiFiltLen - 1
        H = self.acousticPathMatrix(acousticPath, combinedFiltLen)
        B = H @ kiMat

        r = np.zeros((self.numSpeaker*self.ctrlFiltLen, 1))
        X_block = np.zeros((self.numSpeaker * combinedFiltLen, self.numSpeaker * self.ctrlFiltLen))
        margin = combinedFiltLen + self.ctrlFiltLen
        for n in range(margin, self.corrAverageLen+margin):
            #print(n)
            X = linalg.toeplitz(np.flip(origSig[0,n-combinedFiltLen:n]), np.flip(origSig[0,n-self.ctrlFiltLen:n]))
            for k in range(self.numSpeaker):
                X_block[k*combinedFiltLen:(k+1)*combinedFiltLen, k*self.ctrlFiltLen:(k+1)*self.ctrlFiltLen] = X
            
            r += X_block.T @ B @ np.flip(desired[:,n-self.kiFiltLen:n],axis=-1).reshape(-1, 1)
        r = r / self.corrAverageLen
        return r

    def calcControlFilter(self):
        pathLen = np.max((self.pathsBright.shape[-1],self.pathsDark.shape[-1]))
        totSamples = self.kiFiltLen + pathLen - 1 + self.ctrlFiltLen + self.corrAverageLen
        x = self.src.getSamples(totSamples)

        brightPathFilt = fc.createFilter(ir=self.pathsBright, broadcastDim=1, sumOverInput=False)
        xb = np.squeeze(brightPathFilt.process(x), axis=-2)
        #xb = xb[...,pathLen:]
        #d = np.sum(xb, axis=0)
        d = xb[0,:,:]

        rb = self.spatialCorrVector(x, self.pathsBright, self.kiMatBright, d)

        Rb = self.spatialCorrMatrix_sqrt(x, self.pathsBright, self.kiMatBright)
        Rd = self.spatialCorrMatrix_sqrt(x, self.pathsDark, self.kiMatDark)
        # brightPathFilt = fc.createFilter(ir=self.pathsBright, broadcastDim=1, sumOverInput=False)
        # xb = np.squeeze(brightPathFilt.process(x), axis=-2)
        # xb = xb[...,pathLen:]
        # darkPathFilt = fc.createFilter(ir=self.pathsDark, broadcastDim=1, sumOverInput=False)
        # xd = np.squeeze(darkPathFilt.process(x), axis=-2)
        # xd = xd[...,pathLen:]

        # Rb = self.spatialCorrMatrix(xb, self.kiMatBright)
        # Rd = self.spatialCorrMatrix(xd, self.kiMatDark)
        testCorrMatrix(Rb, "Rb")
        testCorrMatrix(Rd, "Rd")

        Rd += self.regRd*np.eye(self.matSize)
        eigVal, eigVec = linalg.eigh(Rb, Rd, type=1)
        eigVal = eigVal[-self.rank:]
        eigVec = eigVec[:,-self.rank:]
        #, subset_by_index=(self.matSize-self.rank,self.matSize-1)
        optFilt = eigVec @ np.diag(1 / (eigVal + self.mu*np.ones(self.rank))) @ eigVec.T @ rb
        
        #optFilt = linalg.solve(R, rb, assume_a="sym")
        optFilt = optFilt.reshape((1, self.numSpeaker,-1))
        return optFilt

class KI_like_VAST(SoundzoneFIR):
    def __init__(self, sim_info, arrays, blockSize, 
                src, ctrlFiltLen, corrAverageLen, rank, mu, regRd):
        super().__init__(sim_info, arrays, blockSize, src, ctrlFiltLen)
        self.name = "KI-like-VAST"
        self.corrAverageLen = corrAverageLen
        self.rank = rank
        self.mu = mu
        self.regRd = regRd

        self.matSize = self.numSpeaker*self.ctrlFiltLen
        assert 1 <= rank <= self.matSize

        self.controlFilter = fc.createFilter(ir=self.calcControlFilter())
        self.metadata["correlation average length"] = self.corrAverageLen

    def acousticPathMatrix(self, paths):
        numMic = paths.shape[1]
        H = np.moveaxis(paths, 1, 0).reshape(numMic,-1).T
        return H

    def spatialCorrMatrix(self, origSig, acousticPath):
        numMic = acousticPath.shape[1]
        pathLen = acousticPath.shape[2]
        H = self.acousticPathMatrix(acousticPath)
        B = H @ H.T
        R = np.zeros((self.numSpeaker*self.ctrlFiltLen, self.numSpeaker*self.ctrlFiltLen))
        #X_block = np.zeros((self.numSpeaker * pathLen, self.numSpeaker * self.ctrlFiltLen))

        margin = max(pathLen, self.ctrlFiltLen)
        for n in range(margin, self.corrAverageLen+margin):
            print(n)
            col = np.flip(origSig[0,n-pathLen:n])
            row = np.flip(origSig[0,n-self.ctrlFiltLen:n])
            for l1 in range(self.numSpeaker):
                for l2 in range(self.numSpeaker):
                    res = linalg.matmul_toeplitz((row,col), B[l1*pathLen:(l1+1)*pathLen,l2*pathLen:(l2+1)*pathLen])
                    R[l1*self.ctrlFiltLen:(l1+1)*self.ctrlFiltLen,
                       l2*self.ctrlFiltLen:(l2+1)*self.ctrlFiltLen] += linalg.matmul_toeplitz((row, col), res.T)

            # for k in range(self.numSpeaker):
            #     X_block[k*pathLen:(k+1)*pathLen, k*self.ctrlFiltLen:(k+1)*self.ctrlFiltLen] = X
            # R += X_block.T @ B @ X_block
        R = R / self.corrAverageLen
        return R

    @util.measure("pedantic")
    def spatialCorrMatrix_pedantic(self, origSig, acousticPath):
        numMic = acousticPath.shape[1]
        pathLen = acousticPath.shape[-1]
        margin = pathLen + self.ctrlFiltLen
        H = np.moveaxis(acousticPath, 1, 0).reshape(numMic,-1)

        R = np.zeros((self.numSpeaker*self.ctrlFiltLen, self.numSpeaker*self.ctrlFiltLen))
        X_block = np.zeros((self.numSpeaker * pathLen, self.numSpeaker * self.ctrlFiltLen))
        for n in range(margin, self.corrAverageLen+margin):
            X_block.fill(0)
            #print(n)
            col = np.flip(origSig[0,n-pathLen:n])
            row = np.flip(origSig[0,n-self.ctrlFiltLen:n])
            X = linalg.toeplitz(col, row)
            for k in range(self.numSpeaker):
                X_block[k*pathLen:(k+1)*pathLen, k*self.ctrlFiltLen:(k+1)*self.ctrlFiltLen] = X
            G = H @ X_block
            #plt.matshow(G)
            #plt.show()
            R += G.T @ G
            #plt.matshow(R)
            #plt.show()
        R = R / (self.corrAverageLen * numMic)
        return R


    def spatialCorrVector_pedantic(self, origSig, acousticPath):
        numMic = acousticPath.shape[1]
        pathLen = acousticPath.shape[-1]
        margin = pathLen + self.ctrlFiltLen
        H = np.moveaxis(acousticPath, 1, 0).reshape(numMic,-1)
        i = np.zeros((self.ctrlFiltLen*self.numSpeaker, 1))
        i[0,0] = 1
        # i = np.eye(self.ctrlFiltLen)[:,:1]
        # i = np.concatenate((i,)*self.numSpeaker, axis=0)

        r = np.zeros((self.numSpeaker*self.ctrlFiltLen, 1))
        X_block = np.zeros((self.numSpeaker * pathLen, self.numSpeaker * self.ctrlFiltLen))
        for n in range(margin, self.corrAverageLen+margin):
            X_block.fill(0)
            #print(n)
            col = np.flip(origSig[0,n-pathLen:n])
            row = np.flip(origSig[0,n-self.ctrlFiltLen:n])
            X = linalg.toeplitz(col, row)
            for k in range(self.numSpeaker):
                X_block[k*pathLen:(k+1)*pathLen, k*self.ctrlFiltLen:(k+1)*self.ctrlFiltLen] = X
            G = H @ X_block
            r += G.T @ G @ i
        r = r / (self.corrAverageLen * numMic)
        return r

    #@util.measure("plain")
    def spatialCorrMatrix_plain(self, origSig, acousticPath):
        numMic = acousticPath.shape[1]
        pathLen = acousticPath.shape[2]
        H = self.acousticPathMatrix(acousticPath)
        B = H @ H.T
        R = np.zeros((self.numSpeaker*self.ctrlFiltLen, self.numSpeaker*self.ctrlFiltLen))
        X_block = np.zeros((self.numSpeaker * pathLen, self.numSpeaker * self.ctrlFiltLen))

        margin = max(pathLen, self.ctrlFiltLen)
        for n in range(margin, self.corrAverageLen+margin):
            #print(n)
            col = np.flip(origSig[0,n-pathLen:n])
            row = np.flip(origSig[0,n-self.ctrlFiltLen:n])
            X = linalg.toeplitz(col, row)

            for k in range(self.numSpeaker):
                X_block[k*pathLen:(k+1)*pathLen, k*self.ctrlFiltLen:(k+1)*self.ctrlFiltLen] = X
            
            R += X_block.T @ B @ X_block
        R = R / self.corrAverageLen
        return R

    def spatialCorrVector(self, origSig, acousticPath, desired):
        numMic = acousticPath.shape[1]
        pathLen = acousticPath.shape[2]
        #H = self.acousticPathMatrix(acousticPath)
        H = np.moveaxis(acousticPath, 1, 0).reshape(numMic,-1)

        r = np.zeros((self.numSpeaker*self.ctrlFiltLen, 1))
        X_block = np.zeros((self.numSpeaker * pathLen, self.numSpeaker * self.ctrlFiltLen))
        margin = pathLen + self.ctrlFiltLen
        for n in range(margin, self.corrAverageLen+margin):
            #print(n)
            X = linalg.toeplitz(np.flip(origSig[0,n-pathLen:n]), np.flip(origSig[0,n-self.ctrlFiltLen:n]))
            for k in range(self.numSpeaker):
                X_block[k*pathLen:(k+1)*pathLen, k*self.ctrlFiltLen:(k+1)*self.ctrlFiltLen] = X
            
            r += X_block.T @ H.T @ desired[:,n-1:n]
        r = r / self.corrAverageLen
        return r

    def calcControlFilter(self):
        pathLen = np.max((self.pathsBright.shape[-1],self.pathsDark.shape[-1]))
        totSamples = pathLen + self.ctrlFiltLen + self.corrAverageLen
        x = self.src.getSamples(totSamples)

        brightPathFilt = fc.createFilter(ir=self.pathsBright, broadcastDim=1, sumOverInput=False)
        xb = np.squeeze(brightPathFilt.process(x), axis=-2)
        #xb = xb[...,pathLen:]
        #d = np.sum(xb, axis=0)
        d = xb[0,...]
        Rb = self.spatialCorrMatrix_pedantic(x, self.pathsBright)
        Rd = self.spatialCorrMatrix_pedantic(x, self.pathsDark)
        rb = self.spatialCorrVector_pedantic(x, self.pathsBright)
        print("ki-like-vast")
        print(np.abs(np.mean(Rb)))
        print(np.abs(np.mean(Rd)))

        testCorrMatrix(Rb, "Rb")
        testCorrMatrix(Rd, "Rd")

        Rd += self.regRd*np.eye(self.matSize)
        eigVal, eigVec = linalg.eigh(Rb, Rd, type=1)
        eigVal = eigVal[-self.rank:]
        eigVec = eigVec[:,-self.rank:]
        #, subset_by_index=(self.matSize-self.rank,self.matSize-1)
        optFilt = eigVec @ np.diag(1 / (eigVal + self.mu*np.ones(self.rank))) @ eigVec.T @ rb
        
        #optFilt = linalg.solve(R, rb, assume_a="sym")
        optFilt = optFilt.reshape((1, self.numSpeaker,-1))
        return optFilt





def testCorrMatrix(mat, name):
    print(f"Rank of matrix size {mat.shape} is {np.linalg.matrix_rank(mat)}")
    print(f"{name} is symmetric: {np.allclose(mat, mat.T)}")
    try:
        np.linalg.cholesky(mat)
        print(f"{name} is positive definite")
    except np.linalg.LinAlgError:
        print(f"{name} is not positive definite")
    val, vec = np.linalg.eig(mat)
    print(f"Lowest eigenvalue {np.min(np.real(val))}")
    print(f"Smallest magnitude eigenvalue {np.min(np.abs(val))}")
    print(f"Largest magnitude eigenvalue {np.max(np.abs(val))}")

