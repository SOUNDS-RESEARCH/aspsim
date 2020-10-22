import numpy as np

import ancsim.settings as s
import ancsim.utilities as util
from ancsim.adaptivefilter.base import AdaptiveFilterFF
from ancsim.adaptivefilter.mpc import FxLMS_FF
from ancsim.signal.filterclasses import FilterMD_IntBuffer, FilterSum_IntBuffer, Filter_IntBuffer



#Derivation 11 with normalization from IWAENC paper. Simple fxlms like normalization
class KernelIP11_papernorm(FxLMS_FF):
    def __init__(self, config, mu, beta, speakerRIR, tdA):
        super().__init__(config, mu, beta, speakerRIR)
        self.name = "Kernel IP 11"
        self.A = np.transpose(tdA, (1,2,0))
        self.combinedFilt = np.zeros((self.secPathFilt.ir.shape[0], 
                                        self.secPathFilt.ir.shape[1], 
                                        self.secPathFilt.ir.shape[-1]+tdA.shape[0]-1))
        for i in range(tdA.shape[-1]):
            for j in range(tdA.shape[-1]):
                for k in range(self.secPathFilt.ir.shape[0]):
                    self.combinedFilt[k,j,:] += np.convolve(self.secPathFilt.ir[k,i,:], tdA[:,i,j], "full")

        self.secPathNormFilt = Filter_IntBuffer(np.sum(self.secPathFilt.ir**2, axis=(0,1)))
        self.buffers["xfnorm"] = np.zeros((1, self.simChunkSize+self.simBuffer))

        self.xKernelFilt = FilterMD_IntBuffer((self.numRef,),ir=self.combinedFilt)
        self.eKernelFilt = FilterSum_IntBuffer(self.A)
        self.buffers["xKern"] = np.zeros((self.numRef, self.numSpeaker, self.numError, self.simChunkSize+self.simBuffer))
        self.buffers["eKern"] =  np.zeros((self.numError, self.simChunkSize+self.simBuffer))
        
    def prepare(self):
        self.buffers["xf"][:,:,:,0:self.idx] = np.transpose(self.secPathXfFilt.process(
                                                self.x[:,0:self.idx]), (2,0,1,3))

        self.buffers["xfnorm"][:,0:self.idx] = self.secPathNormFilt.process(
                                                np.sum(self.x[:,0:self.idx]**2,axis=0,keepdims=True))

    #@util.measure("Kernel 11 update")
    def updateFilter(self):
        grad = np.zeros_like(self.controlFilt.ir)
        M = self.A.shape[-1]//2
        
        self.buffers["xf"][:,:,:,self.updateIdx:self.idx] = np.transpose(self.secPathXfFilt.process(
                                                self.x[:,self.updateIdx:self.idx]), (2,0,1,3))

        self.buffers["xfnorm"][:,self.updateIdx:self.idx] = self.secPathNormFilt.process(
                                                np.sum(self.x[:,self.updateIdx:self.idx]**2,axis=0,keepdims=True))

        self.buffers["xKern"][:,:,:,self.updateIdx:self.idx] = np.transpose(self.xKernelFilt.process(self.x[:,self.updateIdx:self.idx]), (2,0,1,3))
        self.buffers["eKern"][:,self.updateIdx:self.idx] = self.eKernelFilt.process(self.e[:,self.updateIdx:self.idx])
        
        for n in range(self.updateIdx, self.idx):
            term1 = self.buffers["eKern"][None,None:,n,None] * np.flip(self.buffers["xf"][:,:,:,n-M-self.filtLen+1:n-M+1],axis=-1)
            term2 = np.flip(self.buffers["xKern"][:,:,:,n-self.filtLen+1:n+1],axis=-1) * self.e[None, None,:,n-M,None]
            #norm = 1 / (np.sum(self.xf[:,:,:,self.updateIdx-M:self.idx-M]**2) + self.beta)

            refPower= np.sum(self.buffers["xfnorm"][:,n-self.filtLen-M+1:n+1])
            grad += np.sum((term1 + term2),axis=2) / (refPower + self.beta)

        self.controlFilt.ir -= grad * self.mu
        self.updateIdx = self.idx
        



#Derivation 11 - also with papernorm
# DOES NOT WORK
class KernelIP11_minimumdelay(AdaptiveFilterFF):
    def __init__(self, config, mu, beta, speakerRIR, tdA):
        super().__init__(config, mu, beta, speakerRIR)
        self.name = "Kernel IP 11 minimum delay"
        self.A = np.transpose(tdA, (1,2,0))
        self.combinedFilt = np.zeros((secPathError.shape[0], secPathError.shape[1], secPathError.shape[-1]+tdA.shape[0]-1))
        for i in range(tdA.shape[-1]):
            for j in range(tdA.shape[-1]):
                for k in range(secPathError.shape[0]):
                    self.combinedFilt[k,j,:] += np.convolve(secPathError[k,i,:], tdA[:,i,j], "full")



        self.secPathNormFilt = Filter_IntBuffer(np.sum(secPathError**2, axis=(0,1)))
        self.buffers["xfnorm"] = np.zeros((1, s.SIMBUFFER+self.simChunkSize))

        self.xKernelFilt = FilterMD_IntBuffer((self.numRef,),ir=self.combinedFilt)
        self.eKernelFilt = FilterSum_IntBuffer(self.A)
        self.buffers["xKern"] = np.zeros((self.numRef, self.numSpeaker, self.numError, s.SIMBUFFER+self.simChunkSize))
        self.buffers["eKern"] =  np.zeros((self.numError, s.SIMBUFFER+self.simChunkSize))

        self.M = self.A.shape[-1]//2
        
    def calcPureDelay(relLim=1e-2):
        maxVal = np.abs(self.combinedFilt).max()
        for i in range(self.combinedFilt.shape[-1]):
            if np.abs(self.combinedFilt[:,:,i]).max() / maxVal > relLim:
                firstNonZeroIndex = i
                break


    #@util.measure("Kernel 11 update")
    def updateFilter(self):
        grad = np.zeros_like(self.H)
        
        
        #numParams = self.filtLen * self.numSpeaker * self.numRef

        self.buffers["xKern"][:,:,:,self.updateIdx:self.idx] = np.transpose(self.xKernelFilt.process(self.x[:,self.updateIdx:self.idx]), (2,0,1,3))
        self.buffers["eKern"][:,self.updateIdx:self.idx] = self.eKernelFilt.process(self.e[:,self.updateIdx:self.idx])
        #i = 0
        for n in range(self.updateIdx, self.idx):
            term1 = self.buffers["eKern"][None,None:,n,None] * np.flip(self.xf[:,:,:,n-self.M-self.filtLen+1:n-self.M+1],axis=-1)
            term2 = np.flip(self.buffers["xKern"][:,:,:,n-self.filtLen+1:n+1],axis=-1) * self.e[None, None,:,n-self.M,None]
            #norm = 1 / (np.sum(self.xf[:,:,:,self.updateIdx-M:self.idx-M]**2) + self.beta)

            refPower= np.sum(self.buffers["xfnorm"][:,n-self.filtLen-self.M+1:n+1])
            grad += np.sum((term1 + term2),axis=2) / (refPower + self.beta)

        self.H -= grad * self.mu
        self.updateIdx = self.idx
        
    #@util.measure("Kernel 11 forward pass")
    def forwardPassImplement(self, numSamples, noiseAtError, noiseAtRef, errorMicNoise):
        for i in range(numSamples):
            n = self.idx + i
            self.x[:,n] = np.squeeze(noiseAtRef[:,i:i+1])
            X = np.flip(self.x[:,n-self.filtLen+1:n+1], axis=-1)
            self.y[:,n] = np.sum(X[:,None,:]*self.H, axis=(0,-1)) 

        yf = self.secPathErrorFilt.process(self.y[:,self.idx:self.idx+numSamples])
        self.e[:,self.idx:self.idx+numSamples] = noiseAtError + yf + errorMicNoise

        self.xf[:,:,:,self.idx:self.idx+numSamples] = np.transpose(self.secPathXfFilt.process(
                                                        self.x[:,self.idx:self.idx+numSamples]), (2,0,1,3))

        self.buffers["xfnorm"][:,self.idx:self.idx+numSamples] = self.secPathNormFilt.process(
            np.sum(self.x[:,self.idx:self.idx+numSamples]**2,axis=0,keepdims=True)
        )


#Derivation 11
class KernelIP11(AdaptiveFilterFF):
    def __init__(self, config, mu, beta, speakerRIR, tdA):
        super().__init__(config, mu, beta, speakerRIR)
        self.name = "Kernel IP 11"
        self.A = np.transpose(tdA, (1,2,0))
        self.combinedFilt = np.zeros((secPathError.shape[0], secPathError.shape[1], secPathError.shape[-1]+tdA.shape[0]-1))
        for i in range(tdA.shape[-1]):
            for j in range(tdA.shape[-1]):
                for k in range(secPathError.shape[0]):
                    self.combinedFilt[k,j,:] += np.convolve(secPathError[k,i,:], tdA[:,i,j], "full")

        self.xKernelFilt = FilterMD_IntBuffer((self.numRef,),ir=self.combinedFilt)
        self.eKernelFilt = FilterSum_IntBuffer(self.A)
        self.buffers["xKern"] = np.zeros((self.numRef, self.numSpeaker, self.numError, s.SIMBUFFER+self.simChunkSize))
        self.buffers["eKern"] =  np.zeros((self.numError, s.SIMBUFFER+self.simChunkSize))
        
    #@util.measure("Kernel 11 update")
    def updateFilter(self):
        grad = np.zeros_like(self.H)
        M = self.A.shape[-1]//2
        
        #numParams = self.filtLen * self.numSpeaker * self.numRef

        self.buffers["xKern"][:,:,:,self.updateIdx:self.idx] = np.transpose(self.xKernelFilt.process(self.x[:,self.updateIdx:self.idx]), (2,0,1,3))
        self.buffers["eKern"][:,self.updateIdx:self.idx] = self.eKernelFilt.process(self.e[:,self.updateIdx:self.idx])
        #i = 0
        for n in range(self.updateIdx, self.idx):
            term1 = self.buffers["eKern"][None,None:,n,None] * np.flip(self.xf[:,:,:,n-M-self.filtLen+1:n-M+1],axis=-1)
            term2 = np.flip(self.buffers["xKern"][:,:,:,n-self.filtLen+1:n+1],axis=-1) * self.e[None, None,:,n-M,None]
            norm = 1 / (np.sum(self.xf[:,:,:,self.updateIdx-M:self.idx-M]**2) + self.beta)
            #norm = 1 / (np.sum(self.xf[:,:,:,n-M-self.filtLen+1:n-M+1]**2) + self.beta)
            grad += np.sum((term1 + term2),axis=2) * norm

        self.H -= grad * self.mu
        self.updateIdx = self.idx
        
    #@util.measure("Kernel 11 forward pass")
    def forwardPassImplement(self, numSamples, noiseAtError, noiseAtRef, errorMicNoise):
            for i in range(numSamples):
                n = self.idx + i
                self.x[:,n] = np.squeeze(noiseAtRef[:,i:i+1])
                X = np.flip(self.x[:,n-self.filtLen+1:n+1], axis=-1)
                self.y[:,n] = np.sum(X[:,None,:]*self.H, axis=(0,-1)) 

            yf = self.secPathErrorFilt.process(self.y[:,self.idx:self.idx+numSamples])
            self.e[:,self.idx:self.idx+numSamples] = noiseAtError + yf + errorMicNoise

            self.xf[:,:,:,self.idx:self.idx+numSamples] = np.transpose(self.secPathXfFilt.process(
                                                            self.x[:,self.idx:self.idx+numSamples]), (2,0,1,3))





class KernelIP_simpleFromCostNewNorm(AdaptiveFilterFF):
    def __init__(self, config, mu, beta, speakerRIR, tdA):
        super().__init__(config, mu, beta, speakerRIR)
        self.name = "Kernel IP Timedomain simple new crosscost"
        self.A = np.transpose(tdA, (1,2,0))
        self.combinedFilt = np.zeros((secPathError.shape[0], secPathError.shape[1], secPathError.shape[-1]+tdA.shape[0]-1))
        for i in range(tdA.shape[-1]):
            for j in range(tdA.shape[-1]):
                for k in range(secPathError.shape[0]):
                    self.combinedFilt[k,j,:] += np.convolve(secPathError[k,i,:], tdA[:,i,j], "full")

        self.xKernelFilt = FilterMD_IntBuffer((self.numRef,),ir=self.combinedFilt)
        self.eKernelFilt = FilterSum_IntBuffer(self.A)
        self.buffers["xKern"] = np.zeros((self.numRef, self.numSpeaker, self.numError, s.SIMBUFFER+self.simChunkSize))
        self.buffers["eKern"] =  np.zeros((self.numError, s.SIMBUFFER+self.simChunkSize))
        

    def updateFilter(self):
        grad = np.zeros_like(self.H)
        M = self.A.shape[-1]//2

        self.buffers["xKern"][:,:,:,self.updateIdx:self.idx] = np.transpose(self.xKernelFilt.process(self.x[:,self.updateIdx:self.idx]), (2,0,1,3))
        self.buffers["eKern"][:,self.updateIdx:self.idx] = self.eKernelFilt.process(self.e[:,self.updateIdx:self.idx])
        #i = 0
        for n in range(self.updateIdx, self.idx):
            #Xf = np.flip(self.xf[:,:,:,n-self.filtLen+1:n+1], axis=-1)
   
            term1 = self.buffers["eKern"][None,None:,n,None] * np.flip(self.xf[:,:,:,n-M-self.filtLen+1:n-M+1],axis=-1)

            #term1 = np.sum(Xf* self.e[None, None,:,n-M,None],axis=2)

            term2 = np.flip(self.buffers["xKern"][:,:,:,n-self.filtLen+1:n+1],axis=-1) * self.e[None, None,:,n-M,None]
            norm = 1 / (np.sum(self.xf[:,:,:,self.updateIdx:self.idx]**2) + self.beta)
            grad += np.sum((term1 + term2),axis=2) * norm

            #i += 1
        self.H -= grad * self.mu
        self.updateIdx = self.idx

    def forwardPassImplement(self, numSamples, noiseAtError, noiseAtRef, errorMicNoise):
            for i in range(numSamples):
                n = self.idx + i
                self.x[:,n] = np.squeeze(noiseAtRef[:,i:i+1])
                X = np.flip(self.x[:,n-self.filtLen+1:n+1], axis=-1)
                self.y[:,n] = np.sum(X[:,None,:]*self.H, axis=(0,-1)) 

            yf = self.secPathErrorFilt.process(self.y[:,self.idx:self.idx+numSamples])
            self.e[:,self.idx:self.idx+numSamples] = noiseAtError + yf + errorMicNoise

            self.xf[:,:,:,self.idx:self.idx+numSamples] = np.transpose(self.secPathXfFilt.process(
                                                            self.x[:,self.idx:self.idx+numSamples]), (2,0,1,3))




#Derivation 11
class KernelIP11_postErrorNorm(AdaptiveFilterFF):
    def __init__(self, config, mu, beta, speakerRIR, tdA):
        super().__init__(config, mu, beta, speakerRIR)
        self.name = "Kernel IP 11 with posterior error minimization norm"
        self.A = np.transpose(tdA, (1,2,0))
        self.combinedFilt = np.zeros((secPathError.shape[0], secPathError.shape[1], secPathError.shape[-1]+tdA.shape[0]-1))
        for i in range(tdA.shape[-1]):
            for j in range(tdA.shape[-1]):
                for k in range(secPathError.shape[0]):
                    self.combinedFilt[k,j,:] += np.convolve(secPathError[k,i,:], tdA[:,i,j], "full")

        self.xKernelFilt = FilterMD_IntBuffer((self.numRef,),ir=self.combinedFilt)
        self.eKernelFilt = FilterSum_IntBuffer(self.A)
        self.buffers["xKern"] = np.zeros((self.numRef, self.numSpeaker, self.numError, s.SIMBUFFER+self.simChunkSize))
        self.buffers["eKern"] =  np.zeros((self.numError, s.SIMBUFFER+self.simChunkSize))
        

    def updateFilter(self):
        grad = np.zeros_like(self.H)
        M = self.A.shape[-1]//2

        self.buffers["xKern"][:,:,:,self.updateIdx:self.idx] = np.transpose(self.xKernelFilt.process(self.x[:,self.updateIdx:self.idx]), (2,0,1,3))
        self.buffers["eKern"][:,self.updateIdx:self.idx] = self.eKernelFilt.process(self.e[:,self.updateIdx:self.idx])
        #i = 0
        for n in range(self.updateIdx, self.idx):
            #Xf = np.flip(self.xf[:,:,:,n-self.filtLen+1:n+1], axis=-1)
   
            term1 = self.buffers["eKern"][None,None:,n,None] * np.flip(self.xf[:,:,:,n-M-self.filtLen+1:n-M+1],axis=-1)

            #term1 = np.sum(Xf* self.e[None, None,:,n-M,None],axis=2)
            term2 = np.flip(self.buffers["xKern"][:,:,:,n-self.filtLen+1:n+1],axis=-1) * self.e[None, None,:,n-M,None]
            currentGrad = np.sum((term1 + term2),axis=2)
            
            P = np.sum(np.flip(self.xf[:,:,:,n-M-self.filtLen+1:n-M+1],axis=-1) * currentGrad[:,:,None,:], axis=(0,1,3))
            norm = np.sum(self.e[:,n-M] * P) / (np.sum(P**2) + self.beta)
            #norm = 1 / (np.sum(self.xf[:,:,:,self.updateIdx:self.idx]**2) + self.beta)
            grad += currentGrad * norm

            #i += 1
        self.H -= grad * self.mu
        self.updateIdx = self.idx

    def forwardPassImplement(self, numSamples, noiseAtError, noiseAtRef, errorMicNoise):
            for i in range(numSamples):
                n = self.idx + i
                self.x[:,n] = np.squeeze(noiseAtRef[:,i:i+1])
                X = np.flip(self.x[:,n-self.filtLen+1:n+1], axis=-1)
                self.y[:,n] = np.sum(X[:,None,:]*self.H, axis=(0,-1)) 

            yf = self.secPathErrorFilt.process(self.y[:,self.idx:self.idx+numSamples])
            self.e[:,self.idx:self.idx+numSamples] = noiseAtError + yf + errorMicNoise

            self.xf[:,:,:,self.idx:self.idx+numSamples] = np.transpose(self.secPathXfFilt.process(
                                                            self.x[:,self.idx:self.idx+numSamples]), (2,0,1,3))


