import tensorflow as tf







class AdaptiveFilter_tf(ABC):
    def __init__(self, mu, beta, secPathError, secPathTarget, secPathEvals):
        self.name = "Adaptive Filter Base"
        self.H = tf.Variable(tf.zeros((s.NUMREF,s.NUMSPEAKER,s.FILTLENGTH), dtype=tf.float64), dtype=tf.float64)
        self.mu = tf.convert_to_tensor(mu, dtype=tf.float64)
        self.beta = tf.convert_to_tensor(beta, dtype=tf.float64)

        self.y = tf.Variable(tf.zeros((s.NUMSPEAKER,s.SIMCHUNKSIZE+s.SIMBUFFER), dtype=tf.float64), dtype=tf.float64)
        self.x = tf.Variable(tf.zeros((s.NUMREF,s.SIMCHUNKSIZE+s.SIMBUFFER), dtype=tf.float64), dtype=tf.float64)
        self.xf = tf.Variable(tf.zeros((s.NUMREF, s.NUMSPEAKER, s.NUMERROR, s.SIMCHUNKSIZE+s.SIMBUFFER), dtype=tf.float64), dtype=tf.float64)
        self.e = tf.Variable(tf.zeros((s.NUMERROR,s.SIMCHUNKSIZE+s.SIMBUFFER), dtype=tf.float64), dtype=tf.float64)
        self.eTarget = tf.Variable(tf.zeros((s.NUMTARGET,s.SIMCHUNKSIZE+s.SIMBUFFER), dtype=tf.float64), dtype=tf.float64)
        self.eEvals = tf.Variable(tf.zeros((s.NUMEVALS,s.SIMCHUNKSIZE+s.SIMBUFFER), dtype=tf.float64), dtype=tf.float64)

        self.pointNoise = tf.Variable(tf.zeros((s.NUMERROR, s.SIMCHUNKSIZE+s.SIMBUFFER), dtype=tf.float64), dtype=tf.float64)
        self.targetNoise = tf.Variable(tf.zeros((s.NUMTARGET, s.SIMCHUNKSIZE+s.SIMBUFFER), dtype=tf.float64), dtype=tf.float64)

        self.regRed = np.zeros((s.ENDTIMESTEP))
        self.eTargSmoother = Filter_IntBuffer(ir=np.ones((s.OUTPUTSMOOTHING)),numIn=s.NUMTARGET)
        self.targNoiseSmoother = Filter_IntBuffer(ir=np.ones((s.OUTPUTSMOOTHING)),numIn=s.NUMTARGET)
        self.pointRed = np.zeros((s.ENDTIMESTEP))
        self.eSmoother = Filter_IntBuffer(ir=np.ones((s.OUTPUTSMOOTHING)),numIn=s.NUMERROR)
        self.pointNoiseSmoother = Filter_IntBuffer(ir=np.ones((s.OUTPUTSMOOTHING)),numIn=s.NUMERROR)
        
        self.loss = np.zeros((s.ENDTIMESTEP))

        self.secPathError = tf.convert_to_tensor(secPathError, dtype=tf.float64)
        self.secPathTarget = tf.convert_to_tensor(secPathTarget, dtype=tf.float64)
        self.secPathEvals = tf.convert_to_tensor(secPathEvals, dtype=tf.float64)
        self.J = tf.convert_to_tensor(secPathError.shape[-1])
        self.tJ = tf.convert_to_tensor(secPathTarget.shape[-1])
        self.evJ = tf.convert_to_tensor(secPathEvals.shape[-1])

        self.idx = tf.Variable(s.SIMBUFFER, dtype=tf.int64)
        self.updateIdx = tf.Variable(s.SIMBUFFER, dtype=tf.int64)
        self.bufferIdx = 0

    @abstractmethod
    def forwardPass(self):
        pass

    @abstractmethod
    def updateFilter(self):
        pass

    def resetBuffers(self):
        self.saveDiagnostics()

        self.y.assign(tf.concat((self.y[:,-s.SIMBUFFER:], tf.zeros((self.y.shape[0], s.SIMCHUNKSIZE),dtype=tf.float64)) ,axis=-1))
        self.x.assign(tf.concat((self.x[:,-s.SIMBUFFER:], tf.zeros((self.x.shape[0], s.SIMCHUNKSIZE),dtype=tf.float64)) ,axis=-1))
        self.e.assign(tf.concat((self.e[:,-s.SIMBUFFER:], tf.zeros((self.e.shape[0], s.SIMCHUNKSIZE),dtype=tf.float64)) ,axis=-1))
        self.eTarget.assign(tf.concat((self.eTarget[:,-s.SIMBUFFER:],tf.zeros((self.eTarget.shape[0], s.SIMCHUNKSIZE), dtype=tf.float64)) ,axis=-1))
        self.eEvals.assign(tf.concat((self.eEvals[:,-s.SIMBUFFER:],tf.zeros((self.eEvals.shape[0], s.SIMCHUNKSIZE), dtype=tf.float64)) ,axis=-1))
        self.xf.assign(tf.concat((self.xf[:,:,:,-s.SIMBUFFER:], 
            tf.zeros((self.xf.shape[0], self.xf.shape[1], self.xf.shape[2], s.SIMCHUNKSIZE), dtype=tf.float64)),axis=-1))
        
        self.targetNoise.assign(tf.concat((self.targetNoise[:,-s.SIMBUFFER:], 
                                    tf.zeros((self.targetNoise.shape[0], s.SIMCHUNKSIZE), dtype=tf.float64)), axis=-1))
        self.pointNoise.assign(tf.concat((self.pointNoise[:,-s.SIMBUFFER:], 
                                    tf.zeros((self.pointNoise.shape[0], s.SIMCHUNKSIZE), dtype=tf.float64)), axis=-1))
        
        self.bufferIdx += 1
        self.idx.assign(s.SIMBUFFER)
        self.updateIdx.assign(self.updateIdx - s.SIMCHUNKSIZE)

    def saveDiagnostics(self):
        ePow = self.eSmoother.process(self.e[:,s.SIMBUFFER:].numpy()**2)
        pointNoisePow = self.pointNoiseSmoother.process(self.pointNoise[:,s.SIMBUFFER:]**2)
        self.pointRed[self.bufferIdx*s.SIMCHUNKSIZE:(self.bufferIdx+1)*s.SIMCHUNKSIZE] = \
            util.pow2db(np.mean(ePow / pointNoisePow,axis=0))

        eRegPow = self.eTargSmoother.process(self.eTarget[:,s.SIMBUFFER:].numpy()**2)
        regNoisePow = self.targNoiseSmoother.process(self.targetNoise[:,s.SIMBUFFER:]**2)
        self.regRed[self.bufferIdx*s.SIMCHUNKSIZE:(self.bufferIdx+1)*s.SIMCHUNKSIZE] = \
            util.pow2db(np.mean(eRegPow / regNoisePow,axis=0))

    def saveToBuffers(self, noiseAtError, noiseAtTarget):
        numSamples = tf.constant(noiseAtError.shape[-1], dtype=tf.int64)
        self.pointNoise[:,self.idx:self.idx+numSamples].assign(noiseAtError)
        self.targetNoise[:,self.idx:self.idx+numSamples].assign(noiseAtTarget)




class MPC_FF_tf(AdaptiveFilter_tf):
    def __init__(self, mu,beta, secPathError, secPathTarget, secPathEvals, blockSize):
        super().__init__(mu, beta, secPathError, secPathTarget, secPathEvals)
        self.name = "MPC FF TF"
        self.blockSize = tf.convert_to_tensor(blockSize, dtype=tf.int64)
        #self.tfForwardPass = tf.function(self.tfForwardPass)

    def setIdx (self, newIdx):
        self.idx.assign(newIdx)
        
    def saveDiagnostics(self):
        super().saveDiagnostics()
        
        loss = np.sum(self.e[:,s.SIMBUFFER:].numpy()**2, axis=0)
        self.loss[self.bufferIdx*s.SIMCHUNKSIZE:(self.bufferIdx+1)*s.SIMCHUNKSIZE] = util.pow2db(loss)
        
    def updateFilter(self):
        for i in range(self.blockSize):
            Xf = tf.reverse(self.xf[:,:,:,self.idx-i-s.FILTLENGTH+1:self.idx-i+1], axis=(-1,))
            self.H.assign_add(self.mu * (tf.reduce_sum(Xf * self.e[None, None,:,self.idx-i,None],axis=(2,)) / 
                                (tf.reduce_sum(tf.square(Xf)) + self.beta)))

    @util.measure("TF")
    def forwardPass(self, numSamples, noiseAtError, noiseAtRef,  noiseAtTarget, noiseAtEvals, errorMicNoise):
        numSamples = tf.convert_to_tensor(numSamples, dtype=tf.int64)
        noiseAtError = tf.convert_to_tensor(noiseAtError, dtype=tf.float64)
        noiseAtRef = tf.convert_to_tensor(noiseAtRef, dtype = tf.float64)
        noiseAtTarget = tf.convert_to_tensor(noiseAtTarget, dtype = tf.float64)
        noiseAtEvals = tf.convert_to_tensor(noiseAtEvals, dtype = tf.float64)
        errorMicNoise = tf.convert_to_tensor(errorMicNoise, dtype=tf.float64)

        self.tfForwardPass(numSamples,noiseAtError, noiseAtRef,  noiseAtTarget, noiseAtEvals, errorMicNoise)

        self.idx.assign_add(numSamples)

    @tf.function
    def tfForwardPass(self, numSamples, noiseAtError, noiseAtRef, noiseAtTarget, noiseAtEvals, errorMicNoise):
        self.saveToBuffers(noiseAtError, noiseAtTarget)
        for _ in range(numSamples):
            n = self.idx

            Y = tf.reverse(self.y[:,n-self.J+1:n+1], axis=(-1,))
            yf = tf.reduce_sum(Y[:,None,:]*self.secPathError, axis=(0,-1))[:,None]
            self.e[:,n].assign(tf.squeeze(noiseAtError - yf + errorMicNoise))
            
            self.x[:,n].assign(tf.squeeze(noiseAtRef, axis=(-1,)))
            X = tf.reverse(self.x[:,n-self.J+1:n+1], axis=(-1,))
            self.xf[:,:,:,n].assign(tf.reduce_sum(X[:,None,None,:]*self.secPathError[None,:,:,:], axis=(-1,)))
            
            tY = tf.reverse(self.y[:,n-self.tJ+1:n+1], axis=(-1,))
            tyf = tf.reduce_sum(tY[:,None,:]*self.secPathTarget, axis=(0,-1))[:,None]
            self.eTarget[:,n].assign(tf.squeeze(noiseAtTarget - tyf))
            
            evY = tf.reverse(self.y[:,n-self.evJ+1:n+1], axis=(-1,))
            evyf = tf.reduce_sum(evY[:,None,:]*self.secPathEvals, axis=(0,-1))[:,None]
            self.eEvals[:,n].assign(tf.squeeze(noiseAtEvals - evyf))

            X = tf.reverse(self.x[:,n-s.FILTLENGTH+1:n+1], axis=(-1,))

            self.idx.assign_add(1)
            if self.idx >=(s.SIMCHUNKSIZE + s.SIMBUFFER):
                self.resetBuffers()
            
            self.y[:,self.idx].assign(tf.reduce_sum(X[:,None,:]*self.H, axis=(0,-1)))


    