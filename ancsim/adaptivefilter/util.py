import numpy as np
import ancsim.utilities as util


class PhaseCounter:
    """
    An index counter to keep track of non-overlapping continous phases
    
    Example:
    A processor needs the first 2000 samples for an initialization, 
    then must wait 5000 samples before beginning the real processing step.
    The class can then be used by providing
    phase_def = {
        'init' : 2000,
        'wait' : 5000,
        'process' : np.inf
    }
    and then checking if phase_counter.phase == 'init'
    or if phase_counter.current_phase_is('init'):
    
    The number is how many samples that each phase should be
    The first phase will start at sample 0.

    Either np.inf or -1 represents an infinite length
    This should naturally only be used for the last phase
    If all phases has finished, the phase will be None. 

    first_sample will be True on the first sample of each phase,
    allowing running one-time functions in each phase

    Extended implementation to blocksize != 1 can be done later
    """
    def __init__(self, phase_def):
        assert isinstance(phase_def, dict)
        self.phase_def = phase_def
        self.phase = None
        self.first_sample = True

        self.phase_names = list(self.phase_def.keys())
        self.phase_names.append(None)

        self.phase_idxs = [i if i > 0 else np.inf for i in self.phase_def.values()]
        assert all([i != 0 for i in self.phase_idxs])
        self.start_idxs = np.cumsum(self.phase_idxs).tolist()
        self.start_idxs.insert(0,0)

        self.idx = 0
        self.next_phase()

    def next_phase(self):
        print(f"Changed phase from {self.phase}")
        self.phase = self.phase_names.pop(0)
        self.start_idxs.pop(0)
        self.first_sample = True
        print(f"to {self.phase}")

    def progress(self):
        self.idx += 1
        if self.idx >= self.start_idxs[0]:
            self.next_phase()
        else:
            self.first_sample = False

    def current_phase_is(self, phase_name):
        return self.phase == phase_name



class EventCounter:
    """
    An index counter to keep track of events that should 
    only happen every x samples

    event_def is a dictionary with all event
    each entry is 'event_name' : (frequency, offset)

    Example:
    event_counter = EventCounter({'event_1' : (256,0), 'event_2' : (1,0), 'event_3' : (1024,256)})
    event_2 will happen every sample, event_1 every 256 samples
    First at sample 256 all three events will happen simultaneouly. 

    To be used as:
    if 'event_name' in event_counter.event:
        do_thing()

    """
    def __init__(self, event_def):
        self.event_def = event_def
        self.event = []

        self.idx = 0

    def check_events(self):
        self.event = []
        for name, (freq, offset) in self.event_def.items():
            if (self.idx - offset) % freq == 0:
                self.event.append(name)

    def progress(self):
        self.idx += 1 
        self.check_events()



def calcBlockSizes(numSamples, idx, bufferSize, chunkSize):
    leftInBuffer = chunkSize + bufferSize - idx
    sampleCounter = 0
    blockSizes = []
    while sampleCounter < numSamples:
        bLen = np.min((numSamples - sampleCounter, leftInBuffer))
        blockSizes.append(bLen)
        sampleCounter += bLen
        leftInBuffer -= bLen
        if leftInBuffer == 0:
            leftInBuffer = chunkSize
    return blockSizes


def findFirstIndexForBlock(earliestStartIndex, indexToEndAt, blockSize):
    """If processing in fixed size blocks, this function will give the index
        to start at if the processing should end at a specific index.
        Useful for preparation processing, where the exact startpoint isn't important
        but it is important to end at the correct place. """
    numSamples = indexToEndAt - earliestStartIndex
    numBlocks = numSamples // blockSize
    indexToStartAt = indexToEndAt - blockSize*numBlocks
    return indexToStartAt

def blockProcessUntilIndex(earliestStartIndex, indexToEndAt, blockSize):
    """Use as 
        for startIdx, endIdx in blockProcessUntilIndex(earliestStart, indexToEnd, blockSize):
            process(signal[...,startIdx:endIdx])
    
        If processing in fixed size blocks, this function will give the index
        to process for, if the processing should end at a specific index.
        Useful for preparation processing, where the exact startpoint isn't important
        but it is important to end at the correct place. """
    numSamples = indexToEndAt - earliestStartIndex
    numBlocks = numSamples // blockSize
    indexToStartAt = indexToEndAt - blockSize*numBlocks

    for i in range(numBlocks):
        yield indexToStartAt+i*blockSize, indexToStartAt+(i+1)*blockSize

def getWhiteNoiseAtSNR(rng, signal, dim, snr, identicalChannelPower=False):
    """generates Additive White Gaussian Noise
    signal to calculate signal level. time dimension is last dimension
    dim is dimension of output AWGN
    snr, signal to noise ratio, in dB"""
    generatedNoise = rng.standard_normal(dim)
    if identicalChannelPower:
        powerOfSignal = util.avgPower(signal)
    else:
        powerOfSignal = util.avgPower(signal, axis=-1)[:, None]
    generatedNoise *= np.sqrt(powerOfSignal) * (util.db2mag(-snr))
    return generatedNoise


def equivalentDelay(ir):
    """Defined according to the paper with theoretical bounds for
    FxLMS for single channel"""
    eqDelay = np.zeros((ir.shape[0], ir.shape[1]))

    for i in range(ir.shape[0]):
        for j in range(ir.shape[1]):
            thisIr = ir[i, j, :]
            eqDelay[i, j] = np.sum(
                np.arange(thisIr.shape[-1]) * (thisIr ** 2)
            ) / np.sum(thisIr ** 2)
    return eqDelay
