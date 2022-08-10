from pathlib import Path
import shutil
import numpy as np
import json
import yaml
import dill

import ancsim.utilities as util
import ancsim.fileutilities as futil
import ancsim.configutil as configutil
import ancsim.array as ar



def save_session(sessionFolder, config, arrays, simMetadata=None, extraprefix=""):
    sessionPath = futil.get_unique_folder_name("session_" + extraprefix, sessionFolder)

    sessionPath.mkdir()
    save_arrays(sessionPath, arrays)
    save_config(sessionPath, config)
    if simMetadata is not None:
        add_to_sim_metadata(sessionPath, simMetadata)

def load_session(sessionsPath, newFolderPath, chosenConfig, chosenArrays):
    """sessionsPath refers to the folder where all sessions reside 
        This function will load a session matching the chosenConfig 
        and chosenArrays if it exists"""
    sessionToLoad = search_for_matching_session(sessionsPath, chosenConfig, chosenArrays)
    print("Loaded Session: ", str(sessionToLoad))
    loadedArrays = load_arrays(sessionToLoad)
    
    return loadedArrays
    
def load_from_path(sessionPathToLoad, newFolderPath=None):
    loadedArrays = load_arrays(sessionPathToLoad)
    loadedConfig = load_config(sessionPathToLoad)

    if newFolderPath is not None:
        copy_sim_metadata(sessionPathToLoad, newFolderPath)
        save_config(newFolderPath, loadedConfig)
    return loadedConfig, loadedArrays

def copy_sim_metadata(fromFolder, toFolder):
    shutil.copy(
        fromFolder.joinpath("metadata_sim.json"), toFolder.joinpath("metadata_sim.json")
    )


class MatchingSessionNotFoundError(ValueError): pass

def search_for_matching_session(sessionsPath, chosenConfig, chosenArrays):
    for dirPath in sessionsPath.iterdir():
        if dirPath.is_dir():
            #currentSess = sessionsPath.joinpath(dirPath)
            loadedConfig = load_config(dirPath)
            loadedArrays = load_arrays(dirPath)
            #print("configs", configutil.configMatch(chosenConfig, loadedConfig))
            #print("arrays", chosenArrays == loadedArrays)
            if configutil.config_match(chosenConfig, loadedConfig, chosenArrays.path_type) and \
                ar.ArrayCollection.prototype_equals(chosenArrays, loadedArrays):
                return dirPath
    raise MatchingSessionNotFoundError("No matching saved sessions")


def save_config(pathToSave, config):
    if pathToSave is not None:
        with open(pathToSave.joinpath("config.yaml"), "w") as f:
            yaml.dump(config, f, sort_keys=False)

def load_config(sessionPath):
    with open(sessionPath.joinpath("config.yaml")) as f:
        config = yaml.safe_load(f)
    return config


def load_arrays(sessionPath):
    with open(sessionPath.joinpath("arrays.pickle"), "rb") as f:
        arrays = dill.load(f)
    return arrays


def save_arrays(pathToSave, arrays):
    if pathToSave is not None:
        with open(pathToSave.joinpath("arrays.pickle"), "wb") as f:
            dill.dump(arrays, f)



def add_to_sim_metadata(folderPath, dictToAdd):
    try:
        with open(folderPath.joinpath("metadata_sim.json"), "r") as f:
            oldData = json.load(f)
            totData = {**oldData, **dictToAdd}
    except FileNotFoundError:
        totData = dictToAdd
    with open(folderPath.joinpath("metadata_sim.json"), "w") as f:
        json.dump(totData, f, indent=4)


def write_processor_metadata(processors, folderPath):
    if folderPath is None:
        return
        
    fileName = "metadata_processor.json"
    totMetadata = {}
    for proc in processors:
        totMetadata[proc.name] = proc.metadata
    with open(folderPath.joinpath(fileName), "w") as f:
        json.dump(totMetadata, f, indent=4)