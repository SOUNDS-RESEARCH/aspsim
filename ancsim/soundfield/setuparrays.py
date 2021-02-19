import numpy as np

# np.random.seed(1)
import copy

import ancsim.soundfield.generatepoints as gp
from ancsim.array import ArrayCollection, Array, ArrayType
import ancsim.soundfield.geometry as geo
import ancsim.signal.sources as src

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



def getPositionsCuboid3d(config, 
                        numError=28, 
                        numSpeaker=16, 
                        targetReso=(10,10,4), 
                        targetWidth=1, 
                        targetHeight=0.2, 
                        speakerWidth=2.5,
                        speakerHeight=0.2):
    arrays = ArrayCollection()

    arrays.add_array(Array("error", ArrayType.MIC, 
    gp.FourEquidistantRectangles(
        numError,
        targetWidth,
        0.03,
        -targetHeight / 2,
        targetHeight / 2,
    )))

    arrays.add_array(Array("speaker", ArrayType.CTRLSOURCE,
    gp.stackedEquidistantRectangles(
        numSpeaker,
        2,
        [speakerWidth, speakerWidth],
        speakerHeight,
    )))

    target = geo.Cuboid((targetWidth, targetWidth, targetHeight), 
        point_spacing=(targetWidth/targetReso[0], targetWidth/targetReso[1], targetHeight/targetReso[2]))

    arrays.add_array(Array("target", ArrayType.REGION, target))
    # if config["TARGETPOINTSPLACEMENT"] == "target_region":

    #     pos["target"] = gp.uniformFilledCuboid_better(
    #         numTarget,
    #         (targetWidth, targetWidth, targetHeight),
    #     )
    # elif config["TARGETPOINTSPLACEMENT"] == "image":
    #     wallMargin = 0.1
    #     roomLims = (
    #         config["ROOMCENTER"][0] - config["ROOMSIZE"][0] / 2 + wallMargin,
    #         config["ROOMCENTER"][1] - config["ROOMSIZE"][1] / 2 + wallMargin,
    #         config["ROOMCENTER"][0] + config["ROOMSIZE"][0] / 2 - wallMargin,
    #         config["ROOMCENTER"][1] + config["ROOMSIZE"][1] / 2 - wallMargin,
    #     )
    #     pos["target"] = gp.uniformFilledRectangle(
    #         numTarget, roomLims, zAxis=0
    #     )

    source = src.BandlimitedNoiseSource(10, (10,100), config["SAMPLERATE"])

    arrays.add_array(Array("source", ArrayType.FREESOURCE,
        np.array([[-3.5, 0.4, 0.3]], dtype=np.float64), source))
    arrays.add_array(Array("ref", ArrayType.MIC,
        np.array([[-3.5, 0.4, 0.3]], dtype=np.float64)))

    propPaths = {"speaker":{"ref":"none"}}

    return arrays, propPaths




# def getPositionsCuboid3d(config):
#     pos = {}
#     numErrorInner = config["NUMERROR"] // 2
#     numErrorOuter = config["NUMERROR"] - numErrorInner
#     # pos["error"] = np.concatenate((gp.stackedEquidistantRectangles(numErrorInner, 2,
#     #                                 [config["TARGETWIDTH"], config["TARGETWIDTH"]],config["TARGETHEIGHT"]),
#     #                             gp.stackedEquidistantRectangles(numErrorOuter, 2,
#     #                                 [config["TARGETWIDTH"]+0.05, config["TARGETWIDTH"]+0.05], config["TARGETHEIGHT"])), axis=0)

#     pos["error"] = gp.FourEquidistantRectangles(
#         config["NUMERROR"],
#         config["TARGETWIDTH"],
#         0.03,
#         -config["TARGETHEIGHT"] / 2,
#         config["TARGETHEIGHT"] / 2,
#     )
#     pos["speaker"] = gp.stackedEquidistantRectangles(
#         config["NUMSPEAKER"],
#         2,
#         [config["SPEAKERDIM"], config["SPEAKERDIM"]],
#         config["TARGETHEIGHT"],
#     )

#     if config["TARGETPOINTSPLACEMENT"] == "target_region":
#         pos["target"] = gp.uniformFilledCuboid_better(
#             config["NUMTARGET"],
#             (config["TARGETWIDTH"], config["TARGETWIDTH"], config["TARGETHEIGHT"]),
#         )
#     elif config["TARGETPOINTSPLACEMENT"] == "image":
#         wallMargin = 0.1
#         roomLims = (
#             config["ROOMCENTER"][0] - config["ROOMSIZE"][0] / 2 + wallMargin,
#             config["ROOMCENTER"][1] - config["ROOMSIZE"][1] / 2 + wallMargin,
#             config["ROOMCENTER"][0] + config["ROOMSIZE"][0] / 2 - wallMargin,
#             config["ROOMCENTER"][1] + config["ROOMSIZE"][1] / 2 - wallMargin,
#         )
#         pos["target"] = gp.uniformFilledRectangle(
#             config["NUMTARGET"], roomLims, zAxis=0
#         )

#     # sf_image_margins = 0.2
#     # sf_image_lim = [np.min(pos["speaker"][:,0:2])-sf_image_margins,
#     #                 np.max(pos["speaker"][:,0:2])+sf_image_margins]
#     # pos.evals = gp.uniformFilledRectangle(config["NUMEVALS"], lim=sf_image_lim, zAxis=0)

#     pos["source"] = np.array([[-3.5, 0.4, 0.3]], dtype=np.float64)
#     #pos["source"] = np.array([[-15, 0, 0]], dtype=np.float64)
#     assert config["NUMSOURCE"] == 1
#     pos["ref"] = copy.deepcopy(pos["source"])
#     assert config["NUMREF"] == 1
#     return pos


def getPositionsRectangle3d(config):
    assert config["NUMSOURCE"] == 1
    assert config["NUMREF"] == 1


def getPositionsDoubleRectangle3d(config):
    numErrorInner = config["NUMERROR"] // 2
    numErrorOuter = config["NUMERROR"] - numErrorInner