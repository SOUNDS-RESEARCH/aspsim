from pathlib import Path
import shutil
import numpy as np
import os
import json

from ancsim.configfile import getConfig
import ancsim.utilities as util
from ancsim.simulatorsetup import setupPos, setupIR
from ancsim.simulatorlogging import addToSimMetadata
from ancsim.signal.filterclasses import FilterSum_IntBuffer
import ancsim.experiment.multiexperimentutils as meu

import importlib.util


def generateSession(sessionFolder, extraprefix=""):
    sessionPath = util.getUniqueFolderName("session_"+extraprefix, sessionFolder)

    config = getConfig()
    pos = setupPos(config)
    sourceFilters, speakerFilters, irMetadata = setupIR(pos, config)

    sessionPath.mkdir()
    savePositions(sessionPath, pos)
    saveSourceFilters(sessionPath, sourceFilters)
    saveSpeakerFilters(sessionPath, speakerFilters)
    saveConfig(sessionPath, config)
    addToSimMetadata(sessionPath, irMetadata)

def loadSession(newConfig, sessionsPath, newFolderPath):
    sessionToLoad = searchForMatchingSession(sessionsPath, newConfig)
    print("Loaded Session: ", str(sessionToLoad))
    pos = loadPositions(sessionToLoad)
    srcFilt = loadSourceFilters(sessionToLoad)
    spkFilt = loadSpeakerFilters(sessionToLoad)
    copySimMetadata(sessionToLoad, newFolderPath)
    return pos, srcFilt, spkFilt

def copySimMetadata(fromFolder, toFolder):
    shutil.copy(fromFolder.joinpath("simmetadata.json"), toFolder.joinpath("simmetadata.json"))

def searchForMatchingSession(sessionsPath, newConfig):
    for dirname in os.listdir(sessionsPath):
        currentSess = sessionsPath.joinpath(dirname)
        loadedConfig = loadConfig(currentSess)
        if configMatch(newConfig, loadedConfig):
            return currentSess
    raise ValueError("No matching saved sessions")

def saveRawData(filters, timeIdx, folderPath):
    controlFiltDict = {}
    for filt in filters:
        controlFiltDict[filt.name] = filt.outputRawData()
        
    np.savez_compressed(file=folderPath.joinpath("controlFilter_"+str(timeIdx)), **controlFiltDict)

def loadControlFilter(sessionPath):
    filePath = meu.getHighestNumberedFile(sessionPath, "controlFilter_", ".npy")
    controlFilt = np.load(sessionPath.joinpath("controlFilter"))
    return controlFilt

def saveSettings(pathToSave):
    packageDir = Path(__file__).parent
    shutil.copy(packageDir.joinpath("settings.py"), pathToSave.joinpath("settings.py"))

def saveConfig(pathToSave, config):
    packageDir = Path(__file__).parent
    with open (pathToSave.joinpath("configfile.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    #shutil.copy(packageDir.joinpath("configfile.py"), pathToSave.joinpath("configfile.py"))

def loadSettings(sessionPath):
    spec = importlib.util.spec_from_file_location("loadedSettingsModule", str(sessionPath.joinpath("settings.py")))
    loadedSettings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loadedSettings)
    return loadedSettings
    
def loadConfig(sessionPath):
    with open(sessionPath.joinpath("configfile.json"),"r") as f:
        conf = json.load(f)
    return conf

# def loadConfig(sessionPath):
#     spec = importlib.util.spec_from_file_location("loadedConfigModule", str(sessionPath.joinpath("configfile.py")))
#     loadedConfig = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(loadedConfig)
#     conf = loadedConfig.getConfig()
#     return conf

def saveSpeakerFilters(folderPath, speakerFilters):
    np.savez(folderPath.joinpath("speakerfilters.npz"), **speakerFilters)

def loadSpeakerFilters(folderPath):
    with np.load(folderPath.joinpath("speakerfilters.npz")) as loadedFile:
        speakerFilters = {}
        for key in loadedFile.files:
            speakerFilters[key] = loadedFile[key]
    return speakerFilters

def saveSourceFilters(folderPath, sourceFilters):
    np.savez(folderPath.joinpath("sourcefilters.npz"), **{key:filt.ir for key, filt in sourceFilters.items()})

def loadSourceFilters(folderPath):
    with np.load(folderPath.joinpath("sourcefilters.npz")) as srcFilt:
        sourceFilters = {}
        for key in srcFilt.files:
            sourceFilters[key] = FilterSum_IntBuffer(srcFilt[key])
    return sourceFilters

def savePositions(path, pos):
    np.savez(path.joinpath("pos.npz"), **pos)

def loadPositions(folder):
    with np.load(folder.joinpath("pos.npz")) as loadedPos:
        pos = {}
        for key in loadedPos.files:
            pos[key] = loadedPos[key]
    return pos

def configMatch(config1, config2):
    allowedDifferent = ["AUDIOFILENAME", "SOURCETYPE", 
                        "NOISEFREQ", "NOISEBANDWIDTH", "BLOCKSIZE", 
                        "MCPOINTS", "KERNFILTLEN",
                        "SPMFILTLEN", "SAVERAWDATA",
                        "PLOTOUTPUT", "LOADSESSION", 
                        "SAVERAWDATAFREQUENCY", "SOURCEAMP", 
                        "MICSNR", 
                        "FILTLENGTH", "OUTPUTSMOOTHING",
                        "GENSOUNDFIELDATCHUNK", "PLOTFREQUNCY"]

    confView1 = {key:value for key,value in config1.items() if not key in allowedDifferent}
    confView2 = {key:value for key,value in config2.items() if not key in allowedDifferent}
    #print("Config ", confView1 == confView2)
    return confView1 == confView2

def settingsMatch(settings1, settings2):
    # allowedDifferent = [ "ENDTIMESTEP", "ERRORMICNOISE", "FILTLENGTH", "GENSOUNDFIELDATCHUNK", 
    #                     "", "PLOTFREQUENCY", "OUTPUTSMOOTHING", "REFMICNOISE",
    #                     "SIMBUFFER", "SIMCHUNKSIZE"]

    entriesToCheck = ["SAMPLERATE", "C"]
    sView1 = {key:value for key, value in settings1.__dict__.items() if key in entriesToCheck}
    sView2 = {key:value for key, value in settings2.__dict__.items() if key in entriesToCheck}
    #print("Settings ", sView1 == sView2)
    return sView1 == sView2