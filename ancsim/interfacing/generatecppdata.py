import numpy as np
import os
import json

import ancsim.settings as s
import ancsim.soundfield.kernelinterpolation as ki
import ancsim.soundfield.setupfunctions as setup
from ancsim.configfile import config


def saveRealtimeSimSetup():

    posData = {}
    irData = {}
    metaData = {}
    pos = setup.getPositionsRectangle3d()
    irFunc = setup.pointSourceIR3d

    kiFilt = ki.getCMatrixBlock3d(pos.error, config["MCPOINTS"]).tolist()

    
    irData["sourceToError"] = irFunc(pos.source, pos.error).tolist()
    irData["sourceToRef"] = irFunc(pos.source, pos.ref).tolist()
    irData["sourceToTarget"] = irFunc(pos.source, pos.target).tolist()
    irData["sourceToEvals"] = irFunc(pos.source, pos.evals).tolist()

    irData["speakerToError"] = irFunc(pos.speaker, pos.error).tolist()
    irData["speakerToTarget"] = irFunc(pos.speaker, pos.target).tolist()
    irData["speakerToEvals"] = irFunc(pos.speaker, pos.evals).tolist()

    posData["error"] = pos.error.tolist()
    posData["ref"] = pos.ref.tolist()
    posData["speaker"] = pos.speaker.tolist()
    posData["source"] = pos.source.tolist()
    posData["target"] = pos.target.tolist()
    posData["evals"] = pos.evals.tolist()

    metaData["numError"] = s.NUMERROR
    metaData["numRef"] = s.NUMREF
    metaData["numSpeaker"] = s.NUMSPEAKER
    metaData["numSource"] = s.NUMSOURCE
    metaData["numTarget"] = s.NUMTARGET
    metaData["numEvals"] = s.NUMEVALS

    metaData["speakerToErrorLen"] = len(irData["speakerToError"][0][0])
    metaData["speakerToTargetLen"] = len(irData["speakerToTarget"][0][0])
    metaData["speakerToEvalsLen"] = len(irData["speakerToEvals"][0][0])

    metaData["sourceToErrorLen"] = len(irData["sourceToError"][0][0])
    metaData["sourceToRefLen"] = len(irData["sourceToRef"][0][0])
    metaData["sourceToTargetLen"] = len(irData["sourceToTarget"][0][0])
    metaData["sourceToEvalsLen"] = len(irData["sourceToEvals"][0][0])

    fullPath = "../real_time_anc/data/"
    with open(fullPath + "metaData.json", "w") as f:
        json.dump(metaData, f)

    with open(fullPath + "posData.json", "w") as f:
        json.dump(posData, f, indent=4)

    with open(fullPath + "irData.json", "w") as f:
        json.dump(irData, f)

    with open(fullPath + "kiFilt.json", "w") as f:
        json.dump(kiFilt, f)

def appendCppConstant(name, value, typeName, fileName, path):
    if not os.path.isfile(path + fileName):
        with open(path + fileName, "w") as f:
            f.write("#pragma once \n \n")

    with open(path + fileName, "a") as f:
        f.write("static constexpr " + typeName + " " + name.upper() + " { " + str(value) + " }; \n")


def saveRealtimeSimSetup_2():
    fullPath = "../real_time_anc/data/"
    posData = {}
    irData = {}
    metaData = {}
    pos = setup.getPositionsRectangle3d()
    irFunc = setup.pointSourceIR3d

    kiFilt = ki.getCMatrixBlock3d(pos.error, config["MCPOINTS"]).tolist()

    irData["speakerToError"] = irFunc(pos.speaker, pos.error).tolist()
    irData["speakerToTarget"] = irFunc(pos.speaker, pos.target).tolist()
    irData["speakerToEvals"] = irFunc(pos.speaker, pos.evals).tolist()

    irData["sourceToError"] = irFunc(pos.source, pos.error).tolist()
    irData["sourceToRef"] = irFunc(pos.source, pos.ref).tolist()
    irData["sourceToTarget"] = irFunc(pos.source, pos.target).tolist()
    irData["sourceToEvals"] = irFunc(pos.source, pos.evals).tolist()

    posData["error"] = pos.error.tolist()
    posData["ref"] = pos.ref.tolist()
    posData["speaker"] = pos.speaker.tolist()
    posData["source"] = pos.source.tolist()
    posData["target"] = pos.target.tolist()
    posData["evals"] = pos.evals.tolist()

    fName = "constants.h"    
    if os.path.isfile(fullPath+fName):
        os.remove(fullPath+fName)
    appendCppConstant("numError", s.NUMERROR, "int", fName, fullPath)
    appendCppConstant("numRef", s.NUMREF, "int", fName, fullPath)
    appendCppConstant("numSpeaker", s.NUMSPEAKER, "int", fName, fullPath)
    appendCppConstant("numSource", s.NUMSOURCE, "int", fName, fullPath)
    appendCppConstant("numTarget", s.NUMTARGET, "int", fName, fullPath)
    appendCppConstant("numEvals", s.NUMEVALS, "int", fName, fullPath)

    appendCppConstant("speakerToErrorLen", len(irData["speakerToError"][0][0]), "int", fName, fullPath)
    appendCppConstant("speakerToTargetLen", len(irData["speakerToTarget"][0][0]), "int", fName, fullPath)
    appendCppConstant("speakerToEvalsLen", len(irData["speakerToEvals"][0][0]), "int", fName, fullPath)

    appendCppConstant("sourceToErrorLen", len(irData["sourceToError"][0][0]), "int", fName, fullPath)
    appendCppConstant("sourceToRefLen", len(irData["sourceToRef"][0][0]), "int", fName, fullPath)
    appendCppConstant("sourceToTargetLen", len(irData["sourceToTarget"][0][0]), "int", fName, fullPath)
    appendCppConstant("sourceToEvalsLen", len(irData["sourceToEvals"][0][0]), "int", fName, fullPath)


    
    with open(fullPath + "metaData.json", "w") as f:
        json.dump(metaData, f)

    with open(fullPath + "posData.json", "w") as f:
        json.dump(posData, f, indent=4)

    with open(fullPath + "irData.json", "w") as f:
        json.dump(irData, f)

    with open(fullPath + "secPath.json", "w") as f:
        json.dump(irData["speakerToError"], f)

    with open(fullPath + "kiFilt.json", "w") as f:
        json.dump(kiFilt, f)





if __name__ == "__main__":
    saveRealtimeSimSetup_2()