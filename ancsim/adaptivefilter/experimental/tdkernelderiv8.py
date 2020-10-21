os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
import tensorflow as tf


# From derivation 8a, with normalization 1.
class KernelIP_FB(AdaptiveFilterFB):
    def __init__(self, config, mu, beta,speakerRIR, kernelFilt):
        super().__init__(config, mu, beta, speakerRIR)
        self.name = "KernelIP FB 8a"
        self.c = tf.convert_to_tensor(kernelFilt) 
        self.Htf = tf.Variable(tf.zeros(self.H.shape, dtype=tf.float64), dtype=tf.float64)
        
        spec = [tf.TensorSpec(shape=kernelFilt.shape, dtype=tf.float64),
                tf.TensorSpec(shape=(self.numError, None), dtype=tf.float64),
                tf.TensorSpec(shape=(self.numError, self.numSpeaker, self.numError, None), dtype=tf.float64),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.float64),
                tf.TensorSpec(shape=(), dtype=tf.float64)]
                                          
        self.computeGradient = tf.function(self.computeGradient, input_signature=spec)
        self.updateIdx = s.SIMBUFFER

    def updateFilter(self):
        M = int(np.max(self.c.shape)) // 2
        idxMax = self.idx - M
        idxMin = self.updateIdx
        
        etf = tf.convert_to_tensor(self.e)
        xftf = tf.convert_to_tensor(self.xf)
        
        if idxMin < idxMax:
            self.H = self.computeGradient(self.c, etf, xftf, 
                        tf.convert_to_tensor(idxMin), 
                        tf.convert_to_tensor(idxMax), 
                        tf.convert_to_tensor(M), 
                        tf.convert_to_tensor(np.float64(self.mu)),
                        tf.convert_to_tensor(np.float64(self.beta))).numpy()

            self.updateIdx = self.idx - M

    def computeGradient(self, c, e, xf, idxMin, idxMax, M, mu,beta):
        filtLen = self.Htf.shape[-1]
        numError = xf.shape[0]
        numSpeaker = xf.shape[1]

        for n in range(idxMin, idxMax):
            I = tf.transpose(tf.reverse(e[:,n-M:n+M+1,None,None], axis=(1,)),(1,0,2,3)) * c
            I = tf.reduce_sum(I, axis=(0,1))
            
            grad = [tf.nn.conv1d(tf.transpose(tf.reverse(xf[u,:,:,n-M-filtLen+1:n+M+1],axis=(-1,)), (0,2,1)), 
                                tf.reverse(I[:,:,None], axis=(0,)), 
                                stride=1, padding="VALID") 
                  for u in range(numError)]

            grad = tf.squeeze(tf.stack(grad))
            
            P = tf.reduce_sum(grad[:,:,None,:] * tf.reverse(xf[:,:,:,n-filtLen+1:n+1], axis=(-1,)), axis=(0,1,3))
            norm = tf.reduce_sum(e[:,n] * P) / (tf.reduce_sum(tf.square(P)) + beta*0.0001)
            self.Htf.assign(self.Htf - mu*grad*norm)
        return self.Htf
    
    def forwardPassImplement(self, numSamples, noiseAtError, noiseAtRef, errorMicNoise):
        for i in range(numSamples):
            n = self.idx + i
            
            Y = np.flip(self.y[:,n-self.secPathError.shape[-1]:n], axis=-1)
            yf = np.sum(Y[:,np.newaxis,:]*self.secPathError, axis=(0,-1))
            self.e[:,n] = np.squeeze(noiseAtError[:,i] + yf + errorMicNoise[:,i])
            
            self.x[:,n] = self.e[:,n] - yf
            
            X = np.flip(self.x[:,n-self.filtLen+1:n+1], axis=-1)
            self.y[:,n] = np.sum(X[:,None,:]*self.H, axis=(0,-1)) 

        self.xf[:,:,:,self.idx:self.idx+numSamples] = np.transpose(
                self.secPathXfFilt.process(self.x[:,self.idx:self.idx+numSamples]), (2,0,1,3))











class KernelIP_FF(AdaptiveFilterFF):
    def __init__(self, mu, beta, secPathError, secPathTarget, secPathEvals, kernelFilt):
        super().__init__(mu, beta, secPathError, secPathTarget, secPathEvals)
        self.name = "KernelIP FF 8a"
        self.c = tf.convert_to_tensor(kernelFilt) 
        self.Htf = tf.Variable(tf.zeros(self.H.shape, dtype=tf.float64), dtype=tf.float64)
        
        spec = [tf.TensorSpec(shape=kernelFilt.shape, dtype=tf.float64),
                tf.TensorSpec(shape=(self.numError, None), dtype=tf.float64),
                tf.TensorSpec(shape=(self.numRef, self.numSpeaker, self.numError, None), dtype=tf.float64),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.float64),
                tf.TensorSpec(shape=(), dtype=tf.float64)]
                                                 
        self.computeGradient = tf.function(self.computeGradient, input_signature=spec)
        self.updateIdx = s.SIMBUFFER

    def updateFilter(self):
        M = int(np.max(self.c.shape)) // 2
        idxMax = self.idx - M
        idxMin = self.updateIdx
        
        etf = tf.convert_to_tensor(self.e)
        xftf = tf.convert_to_tensor(self.xf)
        
        if idxMin < idxMax:
            self.H = self.computeGradient(self.c, etf, xftf, 
                        tf.convert_to_tensor(idxMin), 
                        tf.convert_to_tensor(idxMax), 
                        tf.convert_to_tensor(M), 
                        tf.convert_to_tensor(np.float64(self.mu)),
                        tf.convert_to_tensor(np.float64(self.beta))).numpy()

            self.updateIdx = self.idx - M

    def computeGradient(self, c, e, xf, idxMin, idxMax, M, mu,beta):
        filtLen = self.Htf.shape[-1]
        numRef = xf.shape[0]
        numSpeaker = xf.shape[1]

        for n in range(idxMin, idxMax):
            I = tf.transpose(tf.reverse(e[:,n-M:n+M+1,None,None], axis=(1,)),(1,0,2,3)) * c
            I = tf.reduce_sum(I, axis=(0,1))
            
            grad = [tf.nn.conv1d(tf.transpose(tf.reverse(xf[u,:,:,n-M-filtLen+1:n+M+1],axis=(-1,)), (0,2,1)), 
                                tf.reverse(I[:,:,None], axis=(0,)), 
                                stride=1, padding="VALID") 
                  for u in range(numRef)]

            grad = tf.squeeze(tf.stack(grad), axis=(-1,))
            
            P = tf.reduce_sum(grad[:,:,None,:] * tf.reverse(xf[:,:,:,n-filtLen+1:n+1], axis=(-1,)), axis=(0,1,3))
            norm = tf.reduce_sum(e[:,n] * P) / (tf.reduce_sum(tf.square(P)) + beta)
            self.Htf.assign(self.Htf - mu*grad*norm)
        return self.Htf

    def forwardPassImplement(self, numSamples, noiseAtError, noiseAtRef, errorMicNoise):
        for i in range(numSamples):
            n = self.idx + i
            self.x[:,n] = noiseAtRef[:,i]
            X = np.flip(self.x[:,n-self.filtLen+1:n+1], axis=-1)
            self.y[:,n] = np.sum(X[:,None,:]*self.H, axis=(0,-1)) 

        yf = self.secPathErrorFilt.process(self.y[:,self.idx:self.idx+numSamples])
        self.e[:,self.idx:self.idx+numSamples] = noiseAtError + yf + errorMicNoise

        self.xf[:,:,:,self.idx:self.idx+numSamples] = np.transpose(
            self.secPathXfFilt.process(self.x[:,self.idx:self.idx+numSamples]), (2,0,1,3))




class KernelIP_FFavgnorm(AdaptiveFilterFF):
    def __init__(self, config, mu, beta, speakerRIR, kernelFilt):
        super().__init__(config, mu, beta, speakerRIR)
        self.name = "KernelIP FF 8a block-normalization"
        self.c = tf.convert_to_tensor(kernelFilt) 
        self.Htf = tf.Variable(tf.zeros(self.H.shape, dtype=tf.float64), dtype=tf.float64)
        self.grad = tf.Variable(tf.zeros(self.H.shape, dtype=tf.float64), dtype=tf.float64)
        self.norm = tf.Variable(tf.zeros((), dtype=tf.float64), dtype=tf.float64)
        
        spec = [tf.TensorSpec(shape=kernelFilt.shape, dtype=tf.float64),
                tf.TensorSpec(shape=(self.numError, None), dtype=tf.float64),
                tf.TensorSpec(shape=(self.numRef, self.numSpeaker, self.numError, None), dtype=tf.float64),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.float64),
                tf.TensorSpec(shape=(), dtype=tf.float64)]
                                                  
        self.computeGradient = tf.function(self.computeGradient, input_signature=spec)
        self.updateIdx = s.SIMBUFFER
        
    #@util.measure("Kernelip 8a FF avgnorm")
    def updateFilter(self):
        M = int(np.max(self.c.shape)) // 2
        idxMax = self.idx - M
        idxMin = self.updateIdx
        
        etf = tf.convert_to_tensor(self.e)
        xftf = tf.convert_to_tensor(self.xf)
        
        if idxMin < idxMax:
            self.H = self.computeGradient(self.c, etf, xftf, 
                        tf.convert_to_tensor(idxMin), 
                        tf.convert_to_tensor(idxMax), 
                        tf.convert_to_tensor(M), 
                        tf.convert_to_tensor(np.float64(self.mu)),
                        tf.convert_to_tensor(np.float64(self.beta))).numpy()

            self.updateIdx = self.idx - M

    def computeGradient(self, c, e, xf, idxMin, idxMax, M, mu,beta):
        filtLen = self.Htf.shape[-1]
        numRef = xf.shape[0]
        numSpeaker = xf.shape[1]
        self.grad.assign(tf.zeros_like(self.grad))
        self.norm.assign(tf.zeros_like(self.norm))

        for n in range(idxMin, idxMax):
            I = tf.transpose(tf.reverse(e[:,n-M:n+M+1,None,None], axis=(1,)),(1,0,2,3)) * c
            I = tf.reduce_sum(I, axis=(0,1))
            
            grad = [tf.nn.conv1d(tf.transpose(tf.reverse(xf[u,:,:,n-M-filtLen+1:n+M+1],axis=(-1,)), (0,2,1)), 
                                tf.reverse(I[:,:,None], axis=(0,)), 
                                stride=1, padding="VALID") 
                  for u in range(numRef)]

            grad = tf.squeeze(tf.stack(grad), axis=(-1,))
            
            P = tf.reduce_sum(grad[:,:,None,:] * tf.reverse(xf[:,:,:,n-filtLen+1:n+1], axis=(-1,)), axis=(0,1,3))
            self.norm.assign_add(tf.reduce_sum(e[:,n] * P) / (tf.reduce_sum(tf.square(P)) + beta))
            #self.norm1.assign_add(tf.reduce_sum(e[:,n] * P))

            self.grad.assign_add(grad)
        self.norm.assign(self.norm / (tf.cast(idxMax, dtype=tf.float64)-tf.cast(idxMin, dtype=tf.float64)))
        self.Htf.assign(self.Htf - mu*self.grad*self.norm)
        return self.Htf


    def forwardPassImplement(self, numSamples, noiseAtError, noiseAtRef, errorMicNoise):
        for i in range(numSamples):
            n = self.idx + i
            self.x[:,n] = noiseAtRef[:,i]
            X = np.flip(self.x[:,n-self.filtLen+1:n+1], axis=-1)
            self.y[:,n] = np.sum(X[:,None,:]*self.H, axis=(0,-1)) 

        yf = self.secPathErrorFilt.process(self.y[:,self.idx:self.idx+numSamples])
        self.e[:,self.idx:self.idx+numSamples] = noiseAtError + yf + errorMicNoise

        self.xf[:,:,:,self.idx:self.idx+numSamples] = np.transpose(
            self.secPathXfFilt.process(self.x[:,self.idx:self.idx+numSamples]), (2,0,1,3))




class KernelIPFreqMyderiv(ConstrainedFastBlockFxLMS):
    def __init__(self,config, mu, beta, speakerRIR, blockSize, kernFilt):
        super().__init__(config, mu,beta,speakerRIR, blockSize)
        self.R = kernFilt
        self.name = "Kernel IP freq myderivation"


    def updateFilter(self):
        assert(self.updated == False)
        grad = self.G.conj() @ self.E @ np.transpose(self.X.conj(),(0,2,1))

        tdGrad = np.fft.ifft(grad, axis=0)
        tdGrad[self.blockSize:,:] = 0
        grad = np.fft.fft(tdGrad, axis=0)
        
        norm = 1 / ((np.sum(np.abs(self.G)**2) * np.sum(np.abs(self.X)**2) / (2*self.blockSize)**2) + self.beta)
        #norm = 1 / (np.sum(np.abs(self.G)**2, axis=(1,2),keepdims=True) * self.refPowerEstimate + self.beta)
        self.H -= self.mu * norm * grad

        self.updateIdx += self.blockSize
        self.update = True
