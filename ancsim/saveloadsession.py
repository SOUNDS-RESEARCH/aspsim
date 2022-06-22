from pathlib import Path
import shutil
import numpy as np
import json
import yaml
import dill

import ancsim.utilities as util
import ancsim.configutil as configutil
import ancsim.array as ar



def saveSession(sessionFolder, config, arrays, simMetadata=None, extraprefix=""):
    sessionPath = util.getUniqueFolderName("session_" + extraprefix, sessionFolder)

    sessionPath.mkdir()
    saveArrays(sessionPath, arrays)
    saveConfig(sessionPath, config)
    if simMetadata is not None:
        addToSimMetadata(sessionPath, simMetadata)

def loadSession(sessionsPath, newFolderPath, chosenConfig, chosenArrays):
    """sessionsPath refers to the folder where all sessions reside 
        This function will load a session matching the chosenConfig 
        and chosenArrays if it exists"""
    sessionToLoad = searchForMatchingSession(sessionsPath, chosenConfig, chosenArrays)
    print("Loaded Session: ", str(sessionToLoad))
    loadedArrays = loadArrays(sessionToLoad)
    
    return loadedArrays
    
def loadFromPath(sessionPathToLoad, newFolderPath=None):
    loadedArrays = loadArrays(sessionPathToLoad)
    loadedConfig = loadConfig(sessionPathToLoad)

    if newFolderPath is not None:
        copySimMetadata(sessionPathToLoad, newFolderPath)
        saveConfig(newFolderPath, loadedConfig)
    return loadedConfig, loadedArrays

def copySimMetadata(fromFolder, toFolder):
    shutil.copy(
        fromFolder.joinpath("metadata_sim.json"), toFolder.joinpath("metadata_sim.json")
    )


class MatchingSessionNotFoundError(ValueError): pass

def searchForMatchingSession(sessionsPath, chosenConfig, chosenArrays):
    for dirPath in sessionsPath.iterdir():
        if dirPath.is_dir():
            #currentSess = sessionsPath.joinpath(dirPath)
            loadedConfig = loadConfig(dirPath)
            loadedArrays = loadArrays(dirPath)
            #print("configs", configutil.configMatch(chosenConfig, loadedConfig))
            #print("arrays", chosenArrays == loadedArrays)
            if configutil.configMatch(chosenConfig, loadedConfig, chosenArrays.path_type) and \
                ar.ArrayCollection.prototype_equals(chosenArrays, loadedArrays):
                return dirPath
    raise MatchingSessionNotFoundError("No matching saved sessions")


def saveRawData(filters, timeIdx, folderPath):
    controlFiltDict = {}
    for filt in filters:
        controlFiltDict[filt.processor.name] = filt.processor.controlFilter.ir

    np.savez_compressed(
        file=folderPath.joinpath("controlFilter_" + str(timeIdx)), **controlFiltDict
    )

    with open(folderPath.joinpath("input_source.pickle"), "wb") as f:
        dill.dump(filters[0].processor.src, f)


def saveConfig(pathToSave, config):
    if pathToSave is not None:
        with open(pathToSave.joinpath("config.yaml"), "w") as f:
            yaml.dump(config, f, sort_keys=False)

def loadConfig(sessionPath):
    with open(sessionPath.joinpath("config.yaml")) as f:
        config = yaml.safe_load(f)
    return config


def loadArrays(sessionPath):
    with open(sessionPath.joinpath("arrays.pickle"), "rb") as f:
        arrays = dill.load(f)
    return arrays


def saveArrays(pathToSave, arrays):
    if pathToSave is not None:
        with open(pathToSave.joinpath("arrays.pickle"), "wb") as f:
            dill.dump(arrays, f)



def addToSimMetadata(folderPath, dictToAdd):
    try:
        with open(folderPath.joinpath("metadata_sim.json"), "r") as f:
            oldData = json.load(f)
            totData = {**oldData, **dictToAdd}
    except FileNotFoundError:
        totData = dictToAdd
    with open(folderPath.joinpath("metadata_sim.json"), "w") as f:
        json.dump(totData, f, indent=4)


def writeFilterMetadata(filters, folderPath):
    if folderPath is None:
        return
        
    fileName = "metadata_processor.json"
    totMetadata = {}
    for filt in filters:
        totMetadata[filt.name] = filt.metadata
    with open(folderPath.joinpath(fileName), "w") as f:
        json.dump(totMetadata, f, indent=4)