from pathlib import Path
import shutil
import numpy as np
import json
import yaml
import dill

import ancsim.utilities as util
import ancsim.configutil as configutil
from ancsim.simulatorlogging import addToSimMetadata
from ancsim.signal.filterclasses import FilterSum
import ancsim.experiment.multiexperimentutils as meu
import ancsim.array as ar



def saveSession(sessionFolder, config, arrays, simMetadata=None, extraprefix=""):
    sessionPath = util.getUniqueFolderName("session_" + extraprefix, sessionFolder)

    # config = getConfig()
    # pos = setupPos(config)
    # sourceFilters, speakerFilters, irMetadata = setupIR(pos, config)

    sessionPath.mkdir()
    saveArrays(sessionPath, arrays)
    # savePositions(sessionPath, pos)
    # saveSourceFilters(sessionPath, sourceFilters)
    # saveSpeakerFilters(sessionPath, speakerFilters)
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
    
    # pos = loadPositions(sessionToLoad)
    # srcFilt = loadSourceFilters(sessionToLoad)
    # spkFilt = loadSpeakerFilters(sessionToLoad)
    #copySimMetadata(sessionToLoad, newFolderPath)
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


def loadControlFilter(sessionPath):
    filePath = meu.getHighestNumberedFile(sessionPath, "controlFilter_", ".npy")
    controlFilt = np.load(sessionPath.joinpath("controlFilter"))
    return controlFilt


# def saveConfig(pathToSave, config):
#     with open(pathToSave.joinpath("configfile.json"), "w") as f:
#         json.dump(config, f, indent=4)

# def loadConfig(sessionPath):
#     with open(sessionPath.joinpath("configfile.json"), "r") as f:
#         conf = json.load(f)
#     return conf
def saveConfig(pathToSave, config):
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
    with open(pathToSave.joinpath("arrays.pickle"), "wb") as f:
        dill.dump(arrays, f)

def saveSpeakerFilters(folderPath, speakerFilters):
    np.savez(folderPath.joinpath("speakerfilters.npz"), **speakerFilters)


def loadSpeakerFilters(folderPath):
    with np.load(folderPath.joinpath("speakerfilters.npz")) as loadedFile:
        speakerFilters = {}
        for key in loadedFile.files:
            speakerFilters[key] = loadedFile[key]
    return speakerFilters


def saveSourceFilters(folderPath, sourceFilters):
    np.savez(
        folderPath.joinpath("sourcefilters.npz"),
        **{key: filt.ir for key, filt in sourceFilters.items()}
    )


def loadSourceFilters(folderPath):
    with np.load(folderPath.joinpath("sourcefilters.npz")) as srcFilt:
        sourceFilters = {}
        for key in srcFilt.files:
            sourceFilters[key] = FilterSum(srcFilt[key])
    return sourceFilters


def savePositions(path, pos):
    np.savez(path.joinpath("pos.npz"), **pos)


def loadPositions(folder):
    with np.load(folder.joinpath("pos.npz")) as loadedPos:
        pos = {}
        for key in loadedPos.files:
            pos[key] = loadedPos[key]
    return pos
