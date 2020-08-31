import numpy as np


import ancsim.settings as s
import ancsim.soundfield.setuparrays as setup
import ancsim.soundfield.kernelinterpolation as ki
import ancsim.soundfield.roomimpulseresponse as rir
from ancsim.signal.sources import SourceArray, SineSource, RecordedNoiseSource, LinearChirpSource, BandlimitedNoiseSource
from ancsim.signal.filterclasses import FilterSum_IntBuffer

def setupIR(pos, config):
    print("Computing Room IR...")
    metadata={}

    if config["REVERBERATION"]:
        if config["SPATIALDIMENSIONS"] == 3:
            irFunc = rir.irRoomImageSource3d
        else:
            raise NotImplementedError
    else:
        if config["SPATIALDIMENSIONS"] == 3:
            irFunc = rir.irPointSource3d
        elif config["SPATIALDIMENSIONS"] == 2:
            irFunc = rir.irPointSource2d
        else:
            raise NotImplementedError

    speakerFilters = {}
    if config["REVERBERATION"]:
        sourceFilters = [FilterSum_IntBuffer(irFunc(pos.source, targetPos, config["ROOMSIZE"], config["ROOMCENTER"], 
                                                config["MAXROOMIRLENGTH"], config["RT60"])) 
                    for targetPos in [pos.error, pos.ref, pos.target, pos.evals]]
        #for pointSetTo in ["error", "target", "evals", "evals2"]
        #    speakerFilters[pointSetTo] = irFunc(pos["speaker"], pos[pointSetTo], 
        #                                        config["ROOMSIZE"], config["ROOMCENTER"], 
        #                                        config["MAXROOMIRLENGTH"])
        speakerFilters["error"], metadata["Secondary path ISM"] = irFunc(pos.speaker, pos.error, 
                                                            config["ROOMSIZE"], config["ROOMCENTER"], 
                                                             config["MAXROOMIRLENGTH"],  config["RT60"], 
                                                                calculateMetadata=True)
        speakerFilters["target"] = irFunc(pos.speaker, pos.target, config["ROOMSIZE"], config["ROOMCENTER"], 
                                        config["MAXROOMIRLENGTH"], config["RT60"])
        speakerFilters["evals"] = irFunc(pos.speaker, pos.evals, config["ROOMSIZE"], config["ROOMCENTER"], 
                                        config["MAXROOMIRLENGTH"],  config["RT60"])
    else:
        sourceFilters = [FilterSum_IntBuffer(irFunc(pos.source, targetPos)) 
                    for targetPos in [pos.error, pos.ref, pos.target, pos.evals]]
        speakerFilters["error"], metadata["Secondary path ISM"] = irFunc(pos.speaker, pos.error, calculateMetadata=True)
        speakerFilters["target"] = irFunc(pos.speaker, pos.target)
        speakerFilters["evals"] = irFunc(pos.speaker, pos.evals)

    if config["REFDIRECTLYOBTAINED"]:
        assert(s.NUMREF == s.NUMSOURCE)
        sourceFilters[1] = FilterSum_IntBuffer(np.ones((s.NUMREF, s.NUMSOURCE, 1)))
    return sourceFilters, speakerFilters, metadata

def setupPos(config):
    print("Setup Positions")
    if config["SPATIALDIMENSIONS"] == 3:
        if config["ARRAYSHAPES"] == "circle":
            if config["REFDIRECTLYOBTAINED"]:
                pos = setup.getPositionsCylinder3d_directref()
            else:
                pos = setup.getPositionsCylinder3d()
        elif config["ARRAYSHAPES"] == "rectangle":
            if config["REFDIRECTLYOBTAINED"]:
                pos = setup.getPositionsRectangle3d(config)
            else:
                raise NotImplementedError
    elif config["SPATIALDIMENSIONS"] == 2:
        if config["ARRAYSHAPES"] == "circle":
            if config["REFDIRECTLYOBTAINED"]:
                raise NotImplementedError
            else:
                pos = setup.getPositionsDisc2d()
        else:
            raise NotImplementedError
    elif config["SPATIALDIMENSIONS"] == 1:
        pos = setup.getPositionsLine1d()
    else:
        raise ValueError
    return pos


def setupSource(config):
    print("Setup Source")
    if config["SOURCETYPE"] == "sine":
        noiseSource = SourceArray(SineSource, s.NUMSOURCE, config["SOURCEAMP"], config["NOISEFREQ"], s.SAMPLERATE)
    elif config["SOURCETYPE"] == "noise":
        noiseSource = SourceArray(BandlimitedNoiseSource, s.NUMSOURCE, config["s.SOURCEAMP"], 
                                [config["NOISEFREQ"], config["NOISEFREQ"]+config["NOISEBANDWIDTH"]], s.SAMPLERATE)
    elif config["SOURCETYPE"] == "chirp":
        noiseSource = SourceArray(LinearChirpSource, s.NUMSOURCE, config["SOURCEAMP"], 
                                  [config["NOISEFREQ"], config["NOISEFREQ"]+config["NOISEBANDWIDTH"]], 8000, s.SAMPLERATE)
    elif config["SOURCETYPE"] == "recorded":
        assert(s.NUMSOURCE == 1)
        packageDir = Path(__file__).parent
        noiseSource = RecordedNoiseSource(config["SOURCEAMP"][0], s.SAMPLERATE, packageDir.joinpath("audiofiles/"+config["AUDIOFILENAME"]), verbose=True)
    else:
        raise ValueError
    return noiseSource



# def setupKernelFilters(pos, config):
#     print("Setup Kernel Filters")
#     if config["SPATIALDIMENSIONS"] == 3:
#         A = ki.getAKernelFreqDomain3d(pos.error, config["FREQKERNELFILTLEN"], config)

#         if config["FREQKERNELFILTLEN"] >= 512: 
#             #If the frequency domain filter is sampled too coarsely, 
#             #its likely better to just generate a new time domain filter
#             tdA = ki.tdAFromFreq(A, filtLen = config["KERNFILTLEN"])
#         else:
#             tdA = ki.getAKernelTimeDomain3d(pos.error, config["KERNFILTLEN"], config)
#     else:
#         raise NotImplementedError
#     return A, tdA



