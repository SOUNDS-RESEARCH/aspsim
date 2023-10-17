raise NotImplementedError



















class BlockLeastMeanSquares(AudioProcessor):
    def __init__(self, config, arrays, block_size, stepSize, beta, filtLen):
        super().__init__(config, arrays, block_size)
        self.name = "Least Mean Squares"

        self.numIn = self.arrays["input"].num
        self.numOut = self.arrays["desired"].num
        self.stepSize = stepSize
        self.beta = beta
        self.filtLen = filtLen

        self.create_buffer("error", self.numOut)
        self.create_buffer("estimate", self.numOut)

        self.controlFilter = fc.FilterSum(irLen=self.filtLen, numIn=self.numIn, numOut=self.numOut)


        self.diag.add_diagnostic("inputPower", dia.SignalPower("input", self.sim_info.tot_samples, self.outputSmoothing))
        self.diag.add_diagnostic("desiredPower", dia.SignalPower("desired", self.sim_info.tot_samples, self.outputSmoothing))
        self.diag.add_diagnostic("errorPower", dia.SignalPower("error", self.sim_info.tot_samples, self.outputSmoothing))
        

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
        
    def __init__(self, sim_info, arrays, block_size, stepSize, beta, filtLen):
        super().__init__(sim_info, arrays, block_size)
        self.name = "Least Mean Squares"

        self.input = "input"
        self.desired = "desired"
        self.error = "error"
        self.estimate = "estimate"

        self.numIn = self.arrays[self.input].num
        self.numOut = self.arrays[self.desired].num
        self.stepSize = stepSize
        self.beta = beta
        self.filtLen = filtLen

        self.create_buffer(self.error, self.numOut)
        self.create_buffer(self.estimate, self.numOut)

        self.controlFilter = fc.FilterSum(irLen=self.filtLen, numIn=self.numIn, numOut=self.numOut)


        self.diag.add_diagnostic("inputPower", dia.SignalPower(self.sim_info, self.input))
        self.diag.add_diagnostic("desiredPower", dia.SignalPower(self.sim_info, self.desired))
        self.diag.add_diagnostic("errorPower", dia.SignalPower(self.sim_info, self.error))
        self.diag.add_diagnostic("paramError", dia.StateNMSE(
                    self.sim_info,
                    "controlFilter.ir", 
                    np.pad(self.arrays.paths["source"][self.desired],((0,0),(0,0),(0,512)), mode="constant", constant_values=0), 
                    100))
        

    def process(self, numSamples):
        for i in range(self.idx-numSamples, self.idx):
            self.sig[self.estimate][:,i] = np.squeeze(self.controlFilter.process(self.sig[self.input][:,i:i+1]), axis=-1)

            self.sig[self.error][:,i] = self.sig[self.desired][:,i] - self.sig[self.estimate][:,i]

            grad = np.flip(self.sig[self.input][:,None,i+1-self.filtLen:i+1], axis=-1) * \
                            self.sig[self.error][None,:,i:i+1]

            normalization = 1 / (np.sum(self.sig[self.input][:,i+1-self.filtLen:i+1]**2) + self.beta)
            self.controlFilter.ir += self.stepSize * normalization * grad
