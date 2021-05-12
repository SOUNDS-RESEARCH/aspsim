import numpy as np

# np.random.seed(1)
import copy

import ancsim.soundfield.generatepoints as gp
import ancsim.array as ar
import ancsim.soundfield.geometry as geo
import ancsim.signal.sources as src



def audioProcessing(config, 
                     numInput=1,
                     numOutput=1):
    if numInput > 1 or numOutput > 1:
        raise NotImplementedError

    arrays = ar.ArrayCollection()
    arrays.add_array(
        ar.FreeSourceArray("source", np.zeros((1,3)), 
            src.SineSource(numChannels=numInput, power=1,freq=100, samplerate=config["samplerate"]))
    )
    arrays.add_array(
        ar.ControllableSourceArray("output", np.zeros((numOutput,3)))
    )
    arrays.add_array(
        ar.MicArray("input", np.zeros((numInput,3)))
    )

    propPaths = {"source":{"input":"isolated"}, "output" : {"input":"none"}}
    return arrays, propPaths

def signalEstimation(config, 
                     numInput=1,
                     numOutput=1):

    arrays = ar.ArrayCollection()
    arrays.add_array(
        ar.FreeSourceArray("source", np.zeros((numInput,3)), 
            #src.SineSource(numChannels=numInput, power=1,freq=100, samplerate=config["samplerate"]))
            src.WhiteNoiseSource(numChannels=numInput, power=1))
    )
    arrays.add_array(
        ar.MicArray("input", np.zeros((numInput,3)))
    )

    arrays.add_array(
        ar.MicArray("desired", np.zeros((numOutput, 3)))
    )
    propPaths = {"source":{"input":"isolated", "desired" : "random"}}

    return arrays, propPaths

def ancMultiPoint(config, 
                    numError=4, 
                    numSpeaker=4, 
                    targetWidth=1.0,
                    targetHeight=0.2,
                    speakerWidth=2.5,
                    speakerHeight=0.2):
    arrays = ar.ArrayCollection()

    arrays.add_array(ar.MicArray("error",
        gp.FourEquidistantRectangles(
            numError,
            targetWidth,
            0.03,
            -targetHeight / 2,
            targetHeight / 2,
    )))

    arrays.add_array(ar.ControllableSourceArray("speaker",
        gp.stackedEquidistantRectangles(
            numSpeaker,
            2,
            [speakerWidth, speakerWidth],
            speakerHeight,
    )))

    source = src.BandlimitedNoiseSource(1, 1, (50,100), config["samplerate"])

    arrays.add_array(ar.FreeSourceArray("source",
        np.array([[-3.5, 0.4, 0.3]], dtype=np.float64), source))
    arrays.add_array(ar.MicArray("ref",
        np.array([[-3.5, 0.4, 0.3]], dtype=np.float64)))

    propPaths = {"speaker":{"ref":"none"}, "source" : {"ref":"isolated"}}

    return arrays, propPaths


def getPositionsCuboid3d(config, 
                        numError=28, 
                        numSpeaker=16, 
                        targetReso=(10,10,4), 
                        targetWidth=1, 
                        targetHeight=0.2, 
                        speakerWidth=2.5,
                        speakerHeight=0.2):
    arrays = ar.ArrayCollection()

    arrays.add_array(ar.MicArray("error",
        gp.FourEquidistantRectangles(
            numError,
            targetWidth,
            0.03,
            -targetHeight / 2,
            targetHeight / 2,
    )))

    arrays.add_array(ar.ControllableSourceArray("speaker",
        gp.stackedEquidistantRectangles(
            numSpeaker,
            2,
            [speakerWidth, speakerWidth],
            speakerHeight,
    )))

    target = geo.Cuboid((targetWidth, targetWidth, targetHeight), 
        point_spacing=(targetWidth/targetReso[0], targetWidth/targetReso[1], targetHeight/targetReso[2]))

    arrays.add_array(ar.RegionArray("target", target))
    # if config["TARGETPOINTSPLACEMENT"] == "target_region":

    #     pos["target"] = gp.uniformFilledCuboid_better(
    #         numTarget,
    #         (targetWidth, targetWidth, targetHeight),
    #     )
    # elif config["TARGETPOINTSPLACEMENT"] == "image":
    #     wallMargin = 0.1
    #     roomLims = (
    #         config["room_center"][0] - config["room_size"][0] / 2 + wallMargin,
    #         config["room_center"][1] - config["room_size"][1] / 2 + wallMargin,
    #         config["room_center"][0] + config["room_size"][0] / 2 - wallMargin,
    #         config["room_center"][1] + config["room_size"][1] / 2 - wallMargin,
    #     )
    #     pos["target"] = gp.uniformFilledRectangle(
    #         numTarget, roomLims, zAxis=0
    #     )

    source = src.BandlimitedNoiseSource(1, 10, (100,300), config["samplerate"])

    arrays.add_array(ar.FreeSourceArray("source",
        np.array([[-3.5, 0.4, 0.3]], dtype=np.float64), source))
    arrays.add_array(ar.MicArray("ref",
        np.array([[-3.5, 0.4, 0.3]], dtype=np.float64)))

    propPaths = {"speaker":{"ref":"none"}, "source" : {"ref":"identity"}}

    return arrays, propPaths



def getPositionsLine1d(radius):
    pos = {}
    pos["error"] = np.array([[0, 0, 0]])
    pos["speaker"] = np.array([[-1, 0, 0]])
    pos["target"] = np.array([[0, 0, 0]])
    pos["source"] = np.array([[-11, 0, 0]])
    pos["ref"] = np.array([[-10, 0, 0]])
    return pos


def getPositionsCylinder3d_directref(config, micRandomness):
    zOffset = 0.1
    numCircles = 2

    pos = {}
    pos["error"] = gp.concentricalCircles(
        config["NUMERROR"],
        numCircles,
        (config["TARGETWIDTH"], config["TARGETWIDTH"] * micRandomness),
        zOffset,
    )
    pos["speaker"] = gp.concentricalCircles(
        config["NUMSPEAKER"],
        numCircles,
        (config["TARGETWIDTH"] + 1, (config["TARGETWIDTH"] + 1) * 1.05),
        2 * zOffset,
    )
    pos["target"] = gp.uniformCylinder(config["NUMTARGET"], config["TARGETWIDTH"], 0.2)
    pos["source"] = gp.equiangularCircle(config["NUMSOURCE"], (10, 10), z=-1)
    pos["ref"] = np.copy(pos["source"])  # + np.array([[0.5,0.5,0.7]])
    return pos


def getPositionsCylinder3d(config, micRandomness):
    zOffset = 0.1
    numCircles = 2

    pos = {}
    pos["error"] = gp.concentricalCircles(
        config["NUMERROR"],
        numCircles,
        (config["TARGETWIDTH"], config["TARGETWIDTH"] * micRandomness),
        zOffset,
    )
    pos["speaker"] = gp.concentricalCircles(
        config["NUMSPEAKER"],
        numCircles,
        (config["TARGETWIDTH"] + 1, (config["TARGETWIDTH"] + 1) * 1.05),
        2 * zOffset,
    )
    pos["target"] = gp.uniformCylinder(config["NUMTARGET"], config["TARGETWIDTH"], 0.2)
    pos["source"] = gp.equiangularCircle(config["NUMSOURCE"], (10, 10), z=-1)
    pos["ref"] = gp.equiangularCircle(
        config["NUMREF"],
        (config["TARGETWIDTH"] + 2, (config["TARGETWIDTH"] + 2) * 1.05),
        np.random.rand(),
        z=0,
    )
    return pos


def getPositionsCylinder3d_middlemics(config):
    zOffset = 0.1
    numCircles = 2

    pos = {}
    pos["error"] = gp.concentricalCircles(
        config["NUMERROR"],
        numCircles,
        (0.1, 0.15),
        zOffset,
    )
    pos["speaker"] = gp.concentricalCircles(
        config["NUMSPEAKER"],
        numCircles,
        (config["TARGETWIDTH"] + 1, (config["TARGETWIDTH"] + 1) * 1.05),
        2 * zOffset,
    )
    pos["target"] = gp.uniformCylinder(config["NUMTARGET"], config["TARGETWIDTH"], 0.2)
    pos["source"] = gp.equiangularCircle(config["NUMSOURCE"], (10, 10), z=-1)
    pos["ref"] = gp.equiangularCircle(
        config["NUMREF"],
        (config["TARGETWIDTH"] + 2, (config["TARGETWIDTH"] + 2) * 1.05),
        np.random.rand(),
        z=0,
    )
    return pos


def getPositionsDisc2d(config):
    pos = {}
    pos["error"] = gp.equiangularCircle(config["NUMERROR"], (1, 1.4))
    pos["speaker"] = gp.equiangularCircle(config["NUMSPEAKER"], (2, 2.4))
    pos["target"] = gp.sunflowerPattern(config["NUMTARGET"], config["TARGETWIDTH"])
    pos["ref"] = gp.equiangularCircle(config["NUMREF"], (3, 3.4))
    # pos["source"] = np.array([[-11,0.4]])
    pos["source"] = gp.equiangularCircle(config["NUMSOURCE"], (10, 10))
    return pos



def getPositionsRectangle3d(config):
    assert config["NUMSOURCE"] == 1
    assert config["NUMREF"] == 1


def getPositionsDoubleRectangle3d(config):
    numErrorInner = config["NUMERROR"] // 2
    numErrorOuter = config["NUMERROR"] - numErrorInner