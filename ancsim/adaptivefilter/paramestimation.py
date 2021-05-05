




class BlockLeastMeanSquares(AudioProcessor):
    def __init__(self, config, arrays, blockSize, stepSize, beta, filtLen):
        super().__init__(config, arrays, blockSize)
        self.name = "Least Mean Squares"

        self.numIn = self.arrays["input"].num
        self.numOut = self.arrays["desired"].num
        self.stepSize = stepSize
        self.beta = beta
        self.filtLen = filtLen

        self.createNewBuffer("error", self.numOut)
        self.createNewBuffer("estimate", self.numOut)

        self.controlFilter = FilterSum_IntBuffer(irLen=self.filtLen, numIn=self.numIn, numOut=self.numOut)


        self.diag.addNewDiagnostic("inputPower", dia.SignalPower("input", self.endTimeStep, self.outputSmoothing))
        self.diag.addNewDiagnostic("desiredPower", dia.SignalPower("desired", self.endTimeStep, self.outputSmoothing))
        self.diag.addNewDiagnostic("errorPower", dia.SignalPower("error", self.endTimeStep, self.outputSmoothing))
        

    def process(self, numSamples):
        self.sig["estimate"][:,self.idx-numSamples:self.idx] = self.controlFilter.process(
            self.sig["input"][:,self.idx-numSamples:self.idx])

        self.sig["error"][:,self.idx-numSamples:self.idx] = self.sig["desired"][:,self.idx-numSamples:self.idx] - \
                                                            self.sig["estimate"][:,self.idx-numSamples:self.idx]

        grad = np.zeros_like(self.controlFilter.ir)
        for n in range(numSamples):
            grad += np.flip(self.sig["input"][None,:,self.idx-self.filtLen-n:self.idx-n], axis=-1) * \
                            self.sig["error"][:,None,self.idx-n-1:self.idx-n]

        normalization = 1 / (np.sum(self.sig["input"][:,self.idx-self.filtLen:self.idx]**2) + self.beta)
        self.controlFilter.ir += self.stepSize * normalization * grad
                        


class LeastMeanSquares(AudioProcessor):
    """Conventional Least Mean Squares
        Accepts block inputs but processes sample by sample, giving better
        performance than the block based version for long blocks. 
    """
        
    def __init__(self, config, arrays, blockSize, stepSize, beta, filtLen):
        super().__init__(config, arrays, blockSize)
        self.name = "Least Mean Squares"

        self.numIn = self.arrays["input"].num
        self.numOut = self.arrays["desired"].num
        self.stepSize = stepSize
        self.beta = beta
        self.filtLen = filtLen

        self.createNewBuffer("error", self.numOut)
        self.createNewBuffer("estimate", self.numOut)

        self.controlFilter = FilterSum_IntBuffer(irLen=self.filtLen, numIn=self.numIn, numOut=self.numOut)


        self.diag.addNewDiagnostic("inputPower", dia.SignalPower("input", self.endTimeStep, self.outputSmoothing))
        self.diag.addNewDiagnostic("desiredPower", dia.SignalPower("desired", self.endTimeStep, self.outputSmoothing))
        self.diag.addNewDiagnostic("errorPower", dia.SignalPower("error", self.endTimeStep, self.outputSmoothing))
        

    def process(self, numSamples):
        for i in range(self.idx-numSamples, self.idx):
            self.sig["estimate"][:,i] = np.squeeze(self.controlFilter.process(self.sig["input"][:,i:i+1]), axis=-1)

            self.sig["error"][:,i] = self.sig["desired"][:,i] - self.sig["estimate"][:,i]

            grad = np.flip(self.sig["input"][:,None,i+1-self.filtLen:i+1], axis=-1) * \
                            self.sig["error"][None,:,i:i+1]

            normalization = 1 / (np.sum(self.sig["input"][:,i+1-self.filtLen:i+1]**2) + self.beta)
            self.controlFilter.ir += self.stepSize * normalization * grad
