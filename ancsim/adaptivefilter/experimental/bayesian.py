
import numpy as np







class MAPLMS_mine(MPC_FF):
    def __init__(self, config, mu, beta, speakerRIR, blockSize):
        super().__init__(config, mu, beta, speakerRIR, blockSize)
        self.name = "MAP LMS my first derivation"

    def updateFilter(self):
        grad = np.zeros_like(self.H)
        for i in range(self.blockSize):
            n = self.idx - 1 - i
            Xf = np.squeeze(np.flip(self.xf[:,:,:,n-self.filtLen+1:n+1], axis=-1), axis=(0,1))
            factor = (self.e[:,n] - Xf @ np.squeeze(self.H,axis=0).T) / (np.sum(Xf**2) + self.mu)
            grad += factor * Xf[None,:,:]
        self.H += grad


class MAPLMS_paper(MPC_FF):
    def __init__(self, config, mu, beta, speakerRIR, blockSize):
        super().__init__(config, mu, beta, speakerRIR, blockSize)
        self.name = "MAP LMS slow version from paper"

    def updateFilter(self):
        grad = np.zeros_like(self.H)
        for i in range(self.blockSize):
            n = self.idx - 1 - i
            Xf = np.squeeze(np.flip(self.xf[:,:,:,n-self.filtLen+1:n+1], axis=-1), axis=(0,1))
            fact1 = np.linalg.pinv(np.eye(self.filtLen) + self.mu * Xf.T@Xf)
            fact2 = self.H - self.mu * Xf * self.e[:,n]

            currentGrad = fact1 @ np.squeeze(fact2,axis=0).T
            self.H = np.transpose(currentGrad[:,:,None], (2,1,0))
        
