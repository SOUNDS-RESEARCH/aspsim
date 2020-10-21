import numpy as np
#np.random.seed(1)
import copy

import ancsim.soundfield.generatepoints as gp
import ancsim.settings as s


def getPositionsLine1d(radius):
    class AllArrays: pass
    pos = AllArrays()
    pos.error = np.array([[0,0,0]])
    pos.speaker = np.array([[-1,0,0]])
    pos.target = np.array([[0,0,0]])
    pos.evals = np.array([[0,0,0], [0,0,0], [0,0,0], [0,0,0]])
    pos.source = np.array([[-11,0,0]])
    pos.ref = np.array([[-10,0,0]])
    return pos

def getPositionsCylinder3d_directref(config):
    zOffset = 0.1
    numCircles = 2

    class AllArrays: pass
    pos = AllArrays()
    pos.error = gp.concentricalCircles(s.NUMERROR, numCircles, (config["TARGETWIDTH"],config["TARGETWIDTH"]*config["ERRORMICRANDOMNESS"]), config["ERRORMICANGLEOFFSET"], zOffset)
    pos.speaker = gp.concentricalCircles(s.NUMSPEAKER,numCircles, (config["TARGETWIDTH"]+1, (config["TARGETWIDTH"]+1)*1.05), config["SPEAKERANGLEOFFSET"], 2*zOffset)
    pos.target = gp.uniformCylinder(s.NUMTARGET, config["TARGETWIDTH"], 0.2)
    pos.evals = gp.uniformFilledRectangle(s.NUMEVALS, zAxis=0)
    pos.source = gp.equiangularCircle(config["NUMSOURCE"], (10,10), z=-1)
    pos.ref = np.copy(pos.source)# + np.array([[0.5,0.5,0.7]])
    return pos

def getPositionsCylinder3d(config):
    zOffset = 0.1
    numCircles = 2

    class AllArrays: pass
    pos = AllArrays()
    pos.error = gp.concentricalCircles(s.NUMERROR, numCircles, (config["TARGETWIDTH"],config["TARGETWIDTH"]*config["ERRORMICRANDOMNESS"]), config["ERRORMICANGLEOFFSET"], zOffset)
    pos.speaker = gp.concentricalCircles(s.NUMSPEAKER,numCircles, (config["TARGETWIDTH"]+1, (config["TARGETWIDTH"]+1)*1.05), config["SPEAKERANGLEOFFSET"], 2*zOffset)
    pos.target = gp.uniformCylinder(s.NUMTARGET, config["TARGETWIDTH"], 0.2)
    pos.evals = gp.uniformFilledRectangle(s.NUMEVALS, zAxis=0)
    pos.source = gp.equiangularCircle(config["NUMSOURCE"], (10,10), z=-1)
    pos.ref = gp.equiangularCircle(s.NUMREF, (config["TARGETWIDTH"]+2, (config["TARGETWIDTH"]+2)*1.05), np.random.rand(), z=0)
    return pos

def getPositionsCylinder3d_middlemics(config):
    zOffset = 0.1
    numCircles = 2

    class AllArrays: pass
    pos = AllArrays()
    pos.error = gp.concentricalCircles(s.NUMERROR, numCircles, (0.1, 0.15), config["ERRORMICANGLEOFFSET"], zOffset)
    pos.speaker = gp.concentricalCircles(s.NUMSPEAKER,numCircles, (config["TARGETWIDTH"]+1, (config["TARGETWIDTH"]+1)*1.05), config["SPEAKERANGLEOFFSET"], 2*zOffset)
    pos.target = gp.uniformCylinder(s.NUMTARGET, config["TARGETWIDTH"], 0.2)
    pos.evals = gp.uniformFilledRectangle(s.NUMEVALS, zAxis=0)
    pos.source = gp.equiangularCircle(config["NUMSOURCE"], (10,10), z=-1)
    pos.ref = gp.equiangularCircle(s.NUMREF, (config["TARGETWIDTH"]+2, (config["TARGETWIDTH"]+2)*1.05), np.random.rand(), z=0)
    return pos

def getPositionsDisc2d(config):
    class AllArrays: pass
    pos = AllArrays()
    pos.error = gp.equiangularCircle(s.NUMERROR, (1,1.4))
    pos.speaker = gp.equiangularCircle(s.NUMSPEAKER, (2,2.4))
    pos.target = gp.sunflowerPattern(s.NUMTARGET, config["TARGETWIDTH"])
    pos.evals = gp.uniformFilledRectangle(s.NUMEVALS)
    pos.ref = gp.equiangularCircle(s.NUMREF, (3,3.4))
    #pos.source = np.array([[-11,0.4]])
    pos.source = gp.equiangularCircle(config["NUMSOURCE"], (10,10))
    return pos


def getPositionsRectangle3d(config):
    class AllArrays: pass
    pos = AllArrays()
    numErrorInner = s.NUMERROR //2
    numErrorOuter = s.NUMERROR - numErrorInner
    # pos.error = np.concatenate((gp.stackedEquidistantRectangles(numErrorInner, 2, 
    #                                 [config["TARGETWIDTH"], config["TARGETWIDTH"]],config["TARGETHEIGHT"]),
    #                             gp.stackedEquidistantRectangles(numErrorOuter, 2, 
    #                                 [config["TARGETWIDTH"]+0.05, config["TARGETWIDTH"]+0.05], config["TARGETHEIGHT"])), axis=0)


    pos.error = gp.FourEquidistantRectangles(s.NUMERROR, config["TARGETWIDTH"], 0.03, 
                                        -config["TARGETHEIGHT"]/2, config["TARGETHEIGHT"]/2)
    pos.speaker = gp.stackedEquidistantRectangles(s.NUMSPEAKER,2, [config["SPEAKERDIM"], config["SPEAKERDIM"]], config["TARGETHEIGHT"])
    pos.target = gp.uniformFilledCuboid_better(s.NUMTARGET, (config["TARGETWIDTH"],config["TARGETWIDTH"], config["TARGETHEIGHT"]))

    wallMargin = 0.1
    roomLims = (config["ROOMCENTER"][0] - config["ROOMSIZE"][0]/2 + wallMargin, 
                config["ROOMCENTER"][1] - config["ROOMSIZE"][1]/2 + wallMargin,
                config["ROOMCENTER"][0] + config["ROOMSIZE"][0]/2 - wallMargin,
                config["ROOMCENTER"][1] + config["ROOMSIZE"][1]/2 - wallMargin)
    #pos.evals = gp.uniformFilledRectangle(s.NUMEVALS, roomLims, zAxis=0)
    sf_image_margins = 0.2
    sf_image_lim = [np.min(pos.speaker[:,0:2])-sf_image_margins, 
                    np.max(pos.speaker[:,0:2])+sf_image_margins]
    pos.evals = gp.uniformFilledRectangle(s.NUMEVALS, lim=sf_image_lim, zAxis=0)

    pos.source = np.array([[-3.5,0.4, 0.3]], dtype=np.float64)
    assert(config["NUMSOURCE"] == 1)
    pos.ref = copy.deepcopy(pos.source)
    assert(s.NUMREF == 1)
    return pos

