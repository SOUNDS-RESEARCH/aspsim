import numpy as np
from pathlib import Path

import ancsim.soundfield.presets as setup
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


# def setupIR(pos, config):
#     print("Computing Room IR...")
#     metadata = {}
#     propagationFilters = {}

#     for micPos, srcPos in it.product(mics, sources):
#         if srcPos not in propagationFilters:
#             propagationFilters[srcPos] = {}
        
#         if config["reverb"] == "ism"
#             if config["spatial_dims"] == 3:
#             propagationFilters[srcPos][micPos], metadata[srcPos+"->"+micPos+" ISM"] = rir.irRoomImageSource3d(
#                                                 scrPos, micPos, config["room_size"], config["room_center"], 
#                                                 config["max_room_ir_length"], config["rt60"], config["samplerate"],
#                                                 calculateMetadata=True)
#             else:
#                 raise ValueError
#         elif config["reverb"] == "freespace":
#             if config["spatial_dims"] == 3:
#                 propagationFilters[srcPos][micPos] = rir.irPointSource3d(
#                 srcPos, micPos, config["samplerate"], config["c"])
#             elif config["spatial_dims"] == 2:
#                 propagationFilters[srcPos][micPos] = rir.irPointSource2d(
#                     srcPos, micPos, config["samplerate"], config["c"]
#                 )
#             else:
#                 raise ValueError

#     for srcName, irSet in propagationFilters.items():
#         for micName, ir in irSet.items():
#             propagationFilters[srcName][micName] = FilterSum_IntBuffer(ir=ir)


#     if config["REFDIRECTLYOBTAINED"]:
#         assert config["NUMREF"] == config["NUMSOURCE"]
#         propagationFilters["source"]["ref"] = FilterSum_IntBuffer(
#             np.ones((config["NUMREF"], config["NUMSOURCE"], 1))
#         )
#     return propagationFilters, metadata


# def setupPos(config):
#     print("Setup Positions")
#     if config["spatial_dims"] == 3:
#         if config["ARRAYSHAPES"] == "circle":
#             if config["REFDIRECTLYOBTAINED"]:
#                 pos = setup.getPositionsCylinder3d_directref(config)
#             else:
#                 pos = setup.getPositionsCylinder3d(config)
#         elif config["ARRAYSHAPES"] == "cuboid":
#             if config["REFDIRECTLYOBTAINED"]:
#                 pos = setup.getPositionsCuboid3d(config)
#             else:
#                 raise NotImplementedError
#         elif config["ARRAYSHAPES"] == "rectangle":
#             if config["REFDIRECTLYOBTAINED"]:
#                 raise NotImplementedError
#             else:
#                 pos = setup.getPositionsRectangle3d(config)
#         elif config["ARRAYSHAPES"] == "doublerectangle":
#             if config["REFDIRECTLYOBTAINED"]:
#                 raise NotImplementedError
#             else:
#                 pos = setup.getPositionsDoubleRectangle3d(config)
#     elif config["spatial_dims"] == 2:
#         if config["ARRAYSHAPES"] == "circle":
#             if config["REFDIRECTLYOBTAINED"]:
#                 raise NotImplementedError
#             else:
#                 pos = setup.getPositionsDisc2d(config)
#         else:
#             raise NotImplementedError
#     elif config["spatial_dims"] == 1:
#         pos = setup.getPositionsLine1d(config)
#     else:
#         raise ValueError
#     return pos


# def setupSource(config):
#     print("Setup Source")
#     if config["SOURCETYPE"] == "sine":
#         noiseSource = SourceArray(
#             SineSource,
#             config["NUMSOURCE"],
#             config["SOURCEAMP"],
#             config["NOISEFREQ"],
#             config["samplerate"],
#         )
#     elif config["SOURCETYPE"] == "noise":
#         noiseSource = SourceArray(
#             BandlimitedNoiseSource,
#             config["NUMSOURCE"],
#             config["SOURCEAMP"],
#             [config["NOISEFREQ"], config["NOISEFREQ"] + config["NOISEBANDWIDTH"]],
#             config["samplerate"],
#         )
#     elif config["SOURCETYPE"] == "chirp":
#         noiseSource = SourceArray(
#             LinearChirpSource,
#             config["NUMSOURCE"],
#             config["SOURCEAMP"],
#             [config["NOISEFREQ"], config["NOISEFREQ"] + config["NOISEBANDWIDTH"]],
#             8000,
#             config["samplerate"],
#         )
#     elif config["SOURCETYPE"] == "recorded":
#         assert config["NUMSOURCE"] == 1
#         packageDir = Path(__file__).parent
#         noiseSource = AudioFileSource(
#             config["SOURCEAMP"][0],
#             config["samplerate"],
#             packageDir.joinpath("audiofiles/" + config["AUDIOFILENAME"]),
#             config["tot_samples"],
#             verbose=True,
#         )
#     else:
#         raise ValueError
#     return noiseSource


# def setupKernelFilters(pos, config):
#     print("Setup Kernel Filters")
#     if config["spatial_dims"] == 3:
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
