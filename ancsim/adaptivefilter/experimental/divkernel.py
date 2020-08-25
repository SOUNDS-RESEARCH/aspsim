import numpy as np

from ancsim.adaptivefilter.base import AdaptiveFilterFF

os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
import tensorflow as tf




    
class KernelMPC_9b(AdaptiveFilterFF):
    def __init__(self, mu, secPathError, secPathTarget, secPathEvals, kernelFilt):
        super().__init__(mu, 0.001, secPathError, secPathTarget, secPathEvals)
        self.name = "Kernel IP 9b"
        self.L = tf.convert_to_tensor(kernelFilt, dtype=tf.float64) 
        self.Htf = tf.Variable(tf.zeros(self.H.shape, dtype=tf.float64), dtype=tf.float64)
        
        spec = [tf.TensorSpec(shape=kernelFilt.shape, dtype=tf.float64),
                tf.TensorSpec(shape=(s.NUMERROR, None), dtype=tf.float64),
                tf.TensorSpec(shape=(s.NUMREF, s.NUMSPEAKER, s.NUMERROR, None), dtype=tf.float64),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.float64)]
        
        spec2 = [tf.TensorSpec(shape=kernelFilt.shape, dtype=tf.float64),
                 tf.TensorSpec(shape=(s.NUMERROR, None), dtype=tf.float64),
                 tf.TensorSpec(shape=(), dtype=tf.int32)]
                                         
        #tf.config.experimental_run_functions_eagerly(True)                                                         
        self.computeGradient = tf.function(self.computeGradient, input_signature=spec)
        #self.computeLoss = tf.function(self.computeLoss, input_signature=spec2)

        self.updateIdx = s.SIMBUFFER
        

    def updateFilter(self):
        M = int(np.max(self.L.shape)) // 2
        idxMax = self.idx - M
        idxMin = self.updateIdx

        e = tf.convert_to_tensor(self.e)
        xf = tf.convert_to_tensor(self.xf)

        self.H = self.computeGradient(self.L, e, xf, 
                            tf.convert_to_tensor(idxMin), 
                            tf.convert_to_tensor(idxMax), 
                            tf.convert_to_tensor(M),
                            tf.convert_to_tensor(np.float64(self.mu))).numpy()
        self.updateIdx = self.idx - M  
    
    
    def computeGradient(self, L, e, xf, idxMin, idxMax, M, mu):
        kernLen = 2*M+1
        filtLen = self.Htf.shape[-1]
        numError = e.shape[0]

        L = tf.transpose(L, perm=(1,0))
        for n in range(idxMin, idxMax):
            eVec = tf.reverse(e[:,n-M:n+M+1], axis=(-1,))
  
            eIP = tf.reduce_sum(L * eVec, axis=(-1,))
            
            xfIP = [tf.reduce_sum(L[None,None,:,:] * tf.reverse(xf[:,:,:,n-M-i:n+M+1-i],axis=(-1,)),axis=-1) for i in range(filtLen)]
            xfIP = tf.stack(xfIP)
            xfIP = tf.transpose(xfIP, (1,2,0,3))
            
            norm = 1 / (tf.reduce_sum(tf.square(xfIP), axis=(0,1,2)) + 0.001)
            
            grad = tf.reduce_sum(eIP[None,None,None,:] * xfIP * norm[None,None,None,:], axis=-1) 
            self.Htf.assign(self.Htf - mu*grad)
        return self.Htf
        
    def computeLoss(self, e):
        return np.sum(e**2,axis=0)

    def _calcBlockSizes(self, numSamples, bufferSize=s.SIMBUFFER, chunkSize=s.SIMCHUNKSIZE):
        leftInBuffer = chunkSize+bufferSize-self.idx
        sampleCounter = 0
        blockSizes = []
        while sampleCounter < numSamples:
            bLen = np.min((numSamples-sampleCounter, leftInBuffer))
            blockSizes.append(bLen)
            sampleCounter += bLen
            leftInBuffer -= bLen 
            if leftInBuffer == 0:
                leftInBuffer = bufferSize
        return blockSizes

    def forwardPass(self, numSamples, noiseAtError, noiseAtRef, noiseAtTarget, noiseAtEvals, errorMicNoise):
        blockSizes = self._calcBlockSizes(numSamples)
        numComputed = 0
        for b in blockSizes:
            self.saveToBuffers(noiseAtError[:,numComputed:numComputed+b], noiseAtTarget[:,numComputed:numComputed+b])
            
            for i in range(b):
                n = self.idx + i
                self.x[:,n] = np.squeeze(noiseAtRef[:,numComputed+i])
                X = np.flip(self.x[:,n-s.FILTLENGTH+1:n+1], axis=-1)
                self.y[:,n] = np.sum(X[:,None,:]*self.H, axis=(0,-1)) 

            yf = self.secPathErrorFilt.process(self.y[:,self.idx:self.idx+b])
            self.e[:,self.idx:self.idx+b] = noiseAtError[:,numComputed:numComputed+b] + yf + errorMicNoise[:,numComputed:numComputed+b]
            
            tyf = self.secPathTargetFilt.process(self.y[:,self.idx:self.idx+b])
            self.eTarget[:,self.idx:self.idx+b] = noiseAtTarget[:,numComputed:numComputed+b] + tyf
            
            evyf = self.secPathEvalsFilt.process(self.y[:,self.idx:self.idx+b])
            self.eEvals[:,self.idx:self.idx+b] = noiseAtEvals[:,numComputed:numComputed+b] + evyf
            
            self.xf[:,:,:,self.idx:self.idx+b] = np.transpose(self.secPathXfFilt.process(self.x[:,self.idx:self.idx+b]), (2,0,1,3))

            self.idx += b
            numComputed += b
            if self.idx >= (s.SIMCHUNKSIZE + s.SIMBUFFER):
                self.resetBuffers()


