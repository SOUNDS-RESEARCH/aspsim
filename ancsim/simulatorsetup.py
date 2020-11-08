import numpy as np
from pathlib import Path

import ancsim.soundfield.setuparrays as setup
import ancsim.soundfield.kernelinterpolation as ki
import ancsim.soundfield.roomimpulseresponse as rir
from ancsim.signal.sources import (
    SourceArray,
    SineSource,
    AudioFileSource,
    LinearChirpSource,
    BandlimitedNoiseSource,
)
from ancsim.signal.filterclasses import FilterSum_IntBuffer


def setupIR(pos, config):
    print("Computing Room IR...")
    metadata = {}

    if config["REVERB"] == "ism":
        if config["SPATIALDIMS"] == 3:
            irFunc = rir.irRoomImageSource3d
        else:
            raise NotImplementedError
    elif config["REVERB"] == "freespace":
        if config["SPATIALDIMS"] == 3:
            irFunc = rir.irPointSource3d
        elif config["SPATIALDIMS"] == 2:
            irFunc = rir.irPointSource2d
        else:
            raise NotImplementedError
    else:
        raise ValueError

    speakerFilters = {}
    if config["REVERB"] == "ism":
        # sourceFilters = [FilterSum_IntBuffer(irFunc(pos.source, targetPos, config["ROOMSIZE"], config["ROOMCENTER"],
        #                                         config["MAXROOMIRLENGTH"], config["RT60"], config["SAMPLERATE"]))
        #             for targetPos in [pos.error, pos.ref, pos.target]]
        sourceFilters = {
            toKey: FilterSum_IntBuffer(
                irFunc(
                    pos["source"],
                    pos[toKey],
                    config["ROOMSIZE"],
                    config["ROOMCENTER"],
                    config["MAXROOMIRLENGTH"],
                    config["RT60"],
                    config["SAMPLERATE"],
                )
            )
            for toKey in ["error", "ref", "target"]
        }

        speakerFilters["error"], metadata["Secondary path ISM"] = irFunc(
            pos["speaker"],
            pos["error"],
            config["ROOMSIZE"],
            config["ROOMCENTER"],
            config["MAXROOMIRLENGTH"],
            config["RT60"],
            config["SAMPLERATE"],
            calculateMetadata=True,
        )
        speakerFilters["target"] = irFunc(
            pos["speaker"],
            pos["target"],
            config["ROOMSIZE"],
            config["ROOMCENTER"],
            config["MAXROOMIRLENGTH"],
            config["RT60"],
            config["SAMPLERATE"],
        )
    elif config["REVERB"] == "freespace":
        sourceFilters = {
            toKey: FilterSum_IntBuffer(
                irFunc(pos["source"], pos[toKey], config["SAMPLERATE"], config["C"])
            )
            for targetPos in ["error", "ref", "target"]
        }
        speakerFilters["error"] = irFunc(
            pos["speaker"], pos["error"], config["SAMPLERATE"], config["C"]
        )
        speakerFilters["target"] = irFunc(
            pos["speaker"], pos["target"], config["SAMPLERATE"], config["C"]
        )

    if config["REFDIRECTLYOBTAINED"]:
        assert config["NUMREF"] == config["NUMSOURCE"]
        sourceFilters["ref"] = FilterSum_IntBuffer(
            np.ones((config["NUMREF"], config["NUMSOURCE"], 1))
        )
    return sourceFilters, speakerFilters, metadata


def setupPos(config):
    print("Setup Positions")
    if config["SPATIALDIMS"] == 3:
        if config["ARRAYSHAPES"] == "circle":
            if config["REFDIRECTLYOBTAINED"]:
                pos = setup.getPositionsCylinder3d_directref(config)
            else:
                pos = setup.getPositionsCylinder3d(config)
        elif config["ARRAYSHAPES"] == "cuboid":
            if config["REFDIRECTLYOBTAINED"]:
                pos = setup.getPositionsCuboid3d(config)
            else:
                raise NotImplementedError
        elif config["ARRAYSHAPES"] == "rectangle":
            if config["REFDIRECTLYOBTAINED"]:
                raise NotImplementedError
            else:
                pos = setup.getPositionsRectangle3d(config)
        elif config["ARRAYSHAPES"] == "doublerectangle":
            if config["REFDIRECTLYOBTAINED"]:
                raise NotImplementedError
            else:
                pos = setup.getPositionsDoubleRectangle3d(config)
    elif config["SPATIALDIMS"] == 2:
        if config["ARRAYSHAPES"] == "circle":
            if config["REFDIRECTLYOBTAINED"]:
                raise NotImplementedError
            else:
                pos = setup.getPositionsDisc2d(config)
        else:
            raise NotImplementedError
    elif config["SPATIALDIMS"] == 1:
        pos = setup.getPositionsLine1d(config)
    else:
        raise ValueError
    return pos


def setupSource(config):
    print("Setup Source")
    if config["SOURCETYPE"] == "sine":
        noiseSource = SourceArray(
            SineSource,
            config["NUMSOURCE"],
            config["SOURCEAMP"],
            config["NOISEFREQ"],
            config["SAMPLERATE"],
        )
    elif config["SOURCETYPE"] == "noise":
        noiseSource = SourceArray(
            BandlimitedNoiseSource,
            config["NUMSOURCE"],
            config["SOURCEAMP"],
            [config["NOISEFREQ"], config["NOISEFREQ"] + config["NOISEBANDWIDTH"]],
            config["SAMPLERATE"],
        )
    elif config["SOURCETYPE"] == "chirp":
        noiseSource = SourceArray(
            LinearChirpSource,
            config["NUMSOURCE"],
            config["SOURCEAMP"],
            [config["NOISEFREQ"], config["NOISEFREQ"] + config["NOISEBANDWIDTH"]],
            8000,
            config["SAMPLERATE"],
        )
    elif config["SOURCETYPE"] == "recorded":
        assert config["NUMSOURCE"] == 1
        packageDir = Path(__file__).parent
        noiseSource = AudioFileSource(
            config["SOURCEAMP"][0],
            config["SAMPLERATE"],
            packageDir.joinpath("audiofiles/" + config["AUDIOFILENAME"]),
            config["ENDTIMESTEP"],
            verbose=True,
        )
    else:
        raise ValueError
    return noiseSource


# def setupKernelFilters(pos, config):
#     print("Setup Kernel Filters")
#     if config["SPATIALDIMS"] == 3:
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
