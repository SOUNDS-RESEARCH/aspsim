import numpy as np
import pathlib
import scipy.signal as spsig
import soundfile as sf

import ancsim.settings as s
import ancsim.utilities as util
    
class SourceArray:
    def __init__ (self, sourceType, numSources, amplitude, *args):
        if isinstance(amplitude, (int, float)):
            amplitude = [amplitude for _ in range(numSources)]
        
        self.sources = [sourceType(amplitude[i], *args) for i in range(numSources)]
        
    def getSamples(self, numSamples):
        output = np.concatenate([src.getSamples(numSamples) for src in self.sources], axis=0)
        return output

class SineSource:
    def __init__(self, amplitude, freq, samplerate):
        self.amp = amplitude
        self.freq = freq
        self.samplerate = samplerate
        rng = np.random.RandomState(1)
        self.phase = rng.uniform(low=0, high=2*np.pi)   

    def getSamples(self, numSamples):
        phasePerSample = 2*np.pi*self.freq/self.samplerate
        noise = self.amp * np.cos(np.arange(numSamples)*phasePerSample + self.phase)[None,:]
        self.phase = (self.phase + numSamples*phasePerSample) % (2*np.pi)
        return noise
    
class MultiSineSource:
    def __init__ (self, amplitude, freq, samplerate):
        self.amp = amplitude
        self.freq = freq
        self.samplerate = samplerate
        self.numSines = freq.shape[-1]
        rng = np.random.RandomState(1)
        self.phase = rng.uniform(low=0, high=2*np.pi,size=self.numSines)
        self.phasePerSample = 2*np.pi*self.freq/self.samplerate
    
    def getSamples(self, numSamples):
        noise = np.zeros((1,numSamples))
        for i in range(self.numSines):
            noise += self.amp * np.cos(np.arange(numSamples)*self.phasePerSample[i] + self.phase[i])[None,:]
            self.phase[i] = (self.phase[i] + numSamples*self.phasePerSample[i]) % (2*np.pi)
        return noise

class LinearChirpSource:
    def __init__ (self, amplitude, freqRange, samplesToSweep, samplerate):
        assert (len(freqRange) == 2)
        self.amp = amplitude
        self.samplerate = samplerate
        self.freqRange = freqRange

        self.samplesToSweep = samplesToSweep
        self.deltaFreq = (freqRange[1]-freqRange[0]) / samplesToSweep
        self.freq = freqRange[0]
        self.freqCounter = 0
        rng = np.random.RandomState(1)
        self.phase = rng.uniform(low=0, high=2*np.pi)

    def nextPhase(self):
        self.phase += (self.freq / self.samplerate) + (self.deltaFreq / (2*self.samplerate))
        self.phase = self.phase % 1
        self.freq = self.freq + self.deltaFreq

    def getSamples(self, numSamples):
        noise = np.zeros((1, numSamples))
        samplesLeft = numSamples
        n = 0
        for blocks in range(1 + numSamples // self.samplesToSweep):
            blockSize = min(samplesLeft, self.samplesToSweep - self.freqCounter)
            for _ in range(blockSize):
                noise[0,n] = np.cos(2*np.pi*self.phase)
                self.nextPhase()
                
                n += 1
                self.freqCounter += 1

                if self.samplesToSweep <= self.freqCounter:
                    self.deltaFreq = -self.deltaFreq
                    #if self.deltaFreq > 0:
                    #    self.freq = self.freqRange[0]
                    #else:
                    #    self.freq = self.freqRange[1]
                    self.freqCounter = 0
            samplesLeft -= blockSize 
        noise *= self.amp
        return noise

class WhiteNoiseSource():
    def __init__(self, power, numChannels=1):
        self.numChannels = numChannels
        self.power = power
        self.stdDev = np.sqrt(power)
        self.rng = np.random.RandomState(1)

    def getSamples(self, numSamples):
        return self.rng.normal(loc=0, scale=self.stdDev, size=(self.numChannels,numSamples))

    def setPower(newPower):
        raise NotImplementedError
    
class BandlimitedNoiseSource():
    def __init__(self, amplitude, freqLim, samplerate):
        assert(len(freqLim) == 2)
        assert(freqLim[0]<freqLim[1])
        self.amplitude = amplitude
        self.rng = np.random.RandomState(1)

        wp = [freq for freq in freqLim]

        self.filtCoef = spsig.butter(16, wp, btype='bandpass', output='sos', fs=samplerate)

        testSig = spsig.sosfilt(self.filtCoef, self.rng.normal(size=10000))
        testSigPow = np.mean(testSig**2)
        self.normFactor = np.sqrt(0.5) / np.sqrt(testSigPow)

        self.zi = spsig.sosfilt_zi(self.filtCoef)

    def getSamples(self, numSamples):
        noise = self.rng.normal(size=numSamples) * self.amplitude
        filtNoise, self.zi = spsig.sosfilt(self.filtCoef, noise, zi=self.zi)
        filtNoise *= self.normFactor
        return filtNoise[None,:]


class AudioFileSource():
    def __init__ (self, amplitude, samplerate, filename, endTimestep, verbose=False):
        #filename = "noise_bathroom_fan.wav"
        #filename = "song_assemble.wav"
        self.amplitude = amplitude
        self.samplerate = samplerate
        self.audioSamples, srAudio = sf.read(filename)
        if self.audioSamples.ndim == 2:
            self.audioSamples = self.audioSamples[:,0]
        self.currentSample = 0
        
        if srAudio != samplerate:
            assert(srAudio % samplerate == 0)
            assert(srAudio > samplerate)
            downsamplingFactor = srAudio // samplerate
            self.audioSamples = spsig.resample_poly(self.audioSamples,up=1, down=downsamplingFactor, axis=-1)
            #srAudio = srAudio // downsamplingFactor
            if verbose:
                print("AudioFileSource downsampled audio file from ", srAudio, " to ", srAudio//downsamplingFactor, " Hz")

        #assert(srAudio == samplerate)
        assert(endTimeStep < self.audioSamples.shape[0])

    def getSamples(self, numSamples):
        sig = self.audioSamples[None, self.currentSample:self.currentSample+numSamples]
        self.currentSample += numSamples
        return sig * self.amplitude



class MLS():
    """Generates a maximum length sequence"""
    def __init__(self, order, polynomial, state=None):
        self.order = order
        self.poly = polynomial
        self.state = None
        
    def getSamples(self, numSamples):
        seq, self.state = spsig.max_len_seq(self.order, state=self.state, length=numSamples, taps=self.poly)
        return seq * 2 - 1


class GoldSequenceSource():
    preferredSequences = {5:[[2],[1,2,3]], 6:[[5],[1,4,5]], 7:[[4],[4,5,6]],
                        8:[[1,2,3,6,7],[1,2,7]], 9:[[5],[3,5,6]], 
                        10:[[2,5,9],[3,4,6,8,9]], 11:[[9],[3,6,9]]}

    def __init__ (self, order, power=np.ones((1,1)), numChannels=1):
        assert(5 <= order <= 11)
        assert(numChannels <= 2**order-1)
        self.numChannels = numChannels
        self.seqLength = 2**order-1
        self.idx = 0
        self.setPower(power)

        mls1 = MLS(order, self.preferredSequences[order][0])
        mls2 = MLS(order, self.preferredSequences[order][1])
        seq1 = mls1.getSamples(2**order-1)
        seq2 = mls2.getSamples(2**order-1)

        self.sequences = np.zeros((numChannels, self.seqLength))
        for i in range(numChannels):
            self.sequences[i,:] = np.negative(seq1 * np.roll(seq2,i)) #XOR with integer shifts

    def getSamples(self, numSamples):
        outSignal = np.zeros((self.numChannels, numSamples))
        blockLengths = util.calcBlockSizes(numSamples, self.idx, self.seqLength)
        outIdx = 0
        for blockLen in blockLengths:
            outSignal[:,outIdx:outIdx+blockLen] = self.sequences[:,self.idx:self.idx+blockLen]
            outIdx += blockLen
            self.idx = (self.idx+blockLen) % self.seqLength
        return outSignal * self.amplitude

    def setPower(self, newPower):
        self.power = newPower
        self.amplitude = np.sqrt(newPower)
        if len(self.amplitude.shape) == 0:
            self.amplitude = self.amplitude[None,None]
        elif len(self.amplitude.shape) == 1:
            self.amplitude = self.amplitude[:,None]
        elif len(self.amplitude.shape) == 2:
            assert(self.amplitude.shape[-1] == 1)
        else:
            raise ValueError
#====================================================================================


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    src = GoldSequenceSource(11)
    for i in range(3):
        sig = src.getSamples(1000)
        sig2 = src.getSamples(1000)
        sigcon = np.concatenate((sig, sig2), axis=-1)
        #plt.plot(np.linspace(0,4000, 1000), np.abs(np.fft.fft(np.squeeze(sig))))
        plt.plot(np.squeeze(sigcon))
        plt.show()