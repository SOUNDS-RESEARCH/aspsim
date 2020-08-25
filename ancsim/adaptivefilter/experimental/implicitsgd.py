import numpy as np

from ancsim.adaptivefilter.base import AdaptiveFilterFF
from ancsim.signal.filterclasses import FilterSum_IntBuffer
import ancsim.settings as s



class ImplicitMPC (AdaptiveFilterFF):
    def __init__(self, mu, beta, secPathError, secPathTarget, secPathEvals):
        self.name = "Implicit Update FF MPC"
        super().__init__(mu, beta, secPathError, secPathTarget, secPathEvals)
        self.buffers["yfilt"] = np.zeros((s.NUMERROR, s.SIMBUFFER+s.SIMCHUNKSIZE))
       

    def updateFilter(self):
        vecLen = s.NUMREF*s.NUMSPEAKER*s.FILTLENGTH
        #grad = np.zeros_like(vecLen, s.NUMERROR)
        
        wVec = np.zeros((vecLen, 1))
        for u in range(s.NUMREF):
            for l in range(s.NUMSPEAKER):
                for i in range(s.FILTLENGTH):
                    wVec[u*(s.FILTLENGTH*s.NUMSPEAKER) + l*s.FILTLENGTH + i,0] = self.H[u,l,i]
            

        for n in range(self.updateIdx, self.idx):

            
            xf = np.zeros((vecLen, s.NUMERROR))
            
            for u in range(s.NUMREF):
                for l in range(s.NUMSPEAKER):
                    for i in range(s.FILTLENGTH):
                        xf[u*(s.FILTLENGTH*s.NUMSPEAKER) + l*s.FILTLENGTH + i, :] = self.xf[u,l,:,n-i]
            
            mat = np.sum(xf[:,None,:] * xf[None,:,:], axis=-1)
            matInv = np.linalg.pinv(np.eye(vecLen) + self.mu * mat)


            factor2 = wVec - self.mu * np.sum(xf * (self.e[:,n:n+1].T - self.buffers["yfilt"][:,n:n+1].T),axis=-1, keepdims=True)

            wVec = matInv @ factor2


        for u in range(s.NUMREF):
            for l in range(s.NUMSPEAKER):
                for i in range(s.FILTLENGTH):
                    self.H[u,l,i] = wVec[u*(s.FILTLENGTH*s.NUMSPEAKER)+l*s.FILTLENGTH+i,:]
        self.updateIdx = self.idx

    def forwardPassImplement(self, numSamples, noiseAtError, noiseAtRef, errorMicNoise):
            for i in range(numSamples):
                n = self.idx + i
                self.x[:,n] = np.squeeze(noiseAtRef[:,i:i+1])
                X = np.flip(self.x[:,n-s.FILTLENGTH+1:n+1], axis=-1)
                self.y[:,n] = np.sum(X[:,None,:]*self.H, axis=(0,-1)) 

            yf = self.secPathErrorFilt.process(self.y[:,self.idx:self.idx+numSamples])
            self.buffers["yfilt"][:,self.idx:self.idx+numSamples] = yf
            self.e[:,self.idx:self.idx+numSamples] = noiseAtError + yf + errorMicNoise

            self.xf[:,:,:,self.idx:self.idx+numSamples] = np.transpose(self.secPathXfFilt.process(
                                                            self.x[:,self.idx:self.idx+numSamples]), (2,0,1,3))
