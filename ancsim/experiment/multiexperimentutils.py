import numpy as np
import matplotlib.pyplot as plt
import json
import os
import itertools as it
import pathlib

import ancsim.settings as s
import ancsim.utilities as util

# def updateSummaryItems(summaryDict, folder):
#     if os.path.exists(folder+"/anc_summary.json"):
#         with open(folder+"/anc_summary.json", "r") as f:
#             summary = json.load(f)
#             for key, val in summaryDict.items():
#                     summary[key] = val
#         with open(folder+"/anc_summary.json", "w") as writeFile:
#                 json.dump(summary, writeFile, indent=4)
#     else:
#         with open(folder+"/anc_summary.json", "w") as writeFile:
#             json.dump(summaryDict, writeFile, indent=4)

def indicesSameInAllFolders(folderName, prefix, suffix, excludedFolders=[]):
    firstSubFolder = True
    prevGoodIndices = []
    for subFolderName in folderName.iterdir():
        if subFolderName in excludedFolders:
            continue
        if folderName.joinpath(subFolderName).is_dir():
            goodIndices = []
            for filePath in folderName.joinpath(subFolderName).iterdir(): 
                filename = filePath.name
                if filename.startswith(prefix) and filename.endswith(suffix):
                    summaryIdx = filename[len(prefix):len(filename)-len(suffix)]
                    try:
                        summaryIdx = int(summaryIdx)
                        if firstSubFolder or summaryIdx in prevGoodIndices:
                            goodIndices.append(summaryIdx)
                    except ValueError:
                        pass
            prevGoodIndices = goodIndices
            firstSubFolder = False
    return goodIndices



def getHighestNumberedFile(folder, prefix, suffix):
    highestFileIdx = -1
    for filename in os.listdir(folder):
        if filename.startswith(prefix) and filename.endswith(suffix):
            summaryIdx = filename[len(prefix):len(filename)-len(suffix)]
            try:
                summaryIdx = int(summaryIdx)
                if summaryIdx > highestFileIdx:
                    highestFileIdx = summaryIdx
            except ValueError:
                print("Warning: check prefix and suffix")
    
    if highestFileIdx == -1:
        return None
    else:
        fname = prefix + str(highestFileIdx) + suffix
        return fname


def getLatestSummary(folder):
    latestSummaryIdx = -1
    for filename in os.listdir(folder):
        if "summary" in filename:
            summaryIdx = int(filename.replace(".json","").split("_")[-1])
            if summaryIdx > latestSummaryIdx:
                latestSummaryIdx = summaryIdx
    
    if latestSummaryIdx == -1:
        return None
    else:
        fname = "summary_" + str(latestSummaryIdx) + ".json"
        return fname

def addSummaryItems(label, value, folder):
    filePath = folder.joinpath("summary.json")
    if os.path.exists(filePath):
        with open(filePath, "r") as f:
            summary = json.load(f)
            if isinstance(label, list):
                for lbl, val in zip(label, value):
                    summary[lbl] = val
            else:
                summary[label] = value
        with open(filePath, "w") as writeFile:
            json.dump(summary, writeFile, indent=4)
    else:
        with open(filePath, "w") as writeFile:
            if isinstance(label, list):
                summary = {}
                for lbl, val in zip(label, value):
                    summary[lbl] = val
                json.dump(summary, writeFile, indent=4)
            else:
                json.dump({label: value}, writeFile, indent=4)

def generateSummary(filters, timeIdx, folder):
    fullSummary = {}
    fullSummary["Timestep"] = timeIdx
    nAvg = np.min((5000, s.SIMCHUNKSIZE)).item()
    fullSummary["SamplesAveraged"] = nAvg
    fullSummary["filters"] = {}
    
    for filt in filters:
        currentSummary = {}
        
        currentSummary["avgRegionalReduction"] = util.pow2db(np.nanmean(util.db2pow(filt.diag.targ.noiseReduction[timeIdx-nAvg:timeIdx])))
        currentSummary["avgPointReduction"] = util.pow2db(np.nanmean(util.db2pow(filt.diag.mics.noiseReduction[timeIdx-nAvg:timeIdx])))
        
        fullSummary["filters"][filt.name] = currentSummary 
    
    with open(folder.joinpath("anc_summary_"+str(timeIdx)+".json"), "w") as writeFile:
        json.dump(fullSummary, writeFile, indent=4)
        
    
        

# def saveRawData_old(filters, timeIdx, folderName):
#     fileName = "rawreductiondata_"

#     if len(filters) == 4:
#         names = ["mpc", "freq", "kernel11", "kernel10"]
#         for filt, name in zip(filters, names):
#             np.savez(folderName + fileName+name+"_"+str(timeIdx), 
#                     micsEPow = filt.diag.micsEPow[:timeIdx],
#                     micsNoise = filt.diag.micsNoise[:timeIdx],
#                     targEPow = filt.diag.targEPow[:timeIdx],
#                     targNoise = filt.diag.targNoise[:timeIdx])
#         np.savez_compressed(file=folderName+"filterCoeffs_"+str(timeIdx), 
#                             mpc=filters[0].H, 
#                             freq=filters[1].H,
#                             kernel10=filters[2].H,
#                             kernel11=filters[3].H)
#     else:
#         print("WARNING: CANT SAVE RAW DATA, NOT IMPLEMENTED FOR CURRENT SETUP")

# def saveSetupData(folderName, pos, sourceFilters, speakerToEvals2):
#     print("WARNING: Deprecated - Please use saveloadsession.py functionality instead")
#     np.savez(file=folderName+"pos",
#                 ref = pos.ref,
#                 error = pos.error,
#                 source = pos.source,
#                 speaker = pos.speaker,
#                 target = pos.target,
#                 evals = pos.evals,
#                 evals2 = pos.evals2)

#     np.savez_compressed(file=folderName+"evals_ir_close", 
#                         sourceToEvals2=sourceFilters[-1].ir, 
#                         speakerToEvals2=speakerToEvals2)

def extractFilterParameters(expFolder):
    allParameters = {}
    for subFolder in expFolder.iterdir():
        if subFolder.is_dir():
            metadataFile = subFolder.joinpath("filtermetadata.json")
            with open(metadataFile,"r") as f:
                metadata = json.load(f)
            for filtName, paramDict in metadata.items():
                for paramName, paramValue in paramDict.items():
                    if filtName not in allParameters:
                        allParameters[filtName] = {}
                    if paramName not in allParameters[filtName]:
                        allParameters[filtName][paramName] = []
                    allParameters[filtName][paramName].append(paramValue)
    with open(expFolder.joinpath("full_experiment_filtermetadata.json"),"w") as f:
        json.dump(allParameters,f, indent=4)
        
def extractSummaries(expFolder, summaryIdx="latest"):
    expData = []
    expDirs = [d for d in expFolder.iterdir() if expFolder.joinpath(d).is_dir()]
    for dirName in expDirs[:]:
        try:
            if summaryIdx == "latest":
                #summaryName = getLatestSummary(expFolder.joinpath(dirName))
                summaryName = getHighestNumberedFile(expFolder.joinpath(dirName), "summary_", ".json")
            else:
                summaryName ="summary_" + str(summaryIdx) + ".json"

            if summaryName is not None:
                with open (expFolder.joinpath(dirName).joinpath(summaryName)) as jsonData:
                    expData.append(json.load(jsonData))
            else: 
                expDirs.remove(dirName)
        except FileNotFoundError:
            expDirs.remove(dirName)
            
    summary = {}
    for expIdx, expResults in enumerate(expData):
        for diagName, diagResults in expResults.items():
            for filtName, diagValue in diagResults.items():
                if diagName not in summary:
                    summary[diagName] = {}
                if filtName not in summary[diagName]:
                    summary[diagName][filtName] = []
                summary[diagName][filtName].append(diagValue)
            
    if summaryIdx == "latest":
        outputName = expFolder.joinpath("full_experiment_summary.json")
    else:
        outputName =  expFolder.joinpath("full_experiment_summary_" + str(summaryIdx) + ".json")
    with open(outputName, "w") as writeFile:
        json.dump(summary, writeFile, indent=4)
        
        

def extractAllSummaries(expFolder):
    indices = indicesSameInAllFolders(expFolder, "summary_", ".json", ["reduction_by_frequency"])
    for idx in indices:
        extractSummaries(expFolder, idx)


def extractConfigs(expFolder):
    expConfigs = []
    expDirs = [d for d in expFolder.iterdir() if expFolder.joinpath(d).is_dir()]
    for i in expDirs[:]:
        try:
            with open (expFolder.joinpath(i).joinpath("configfile.json")) as jsonData:
                expConfigs.append(json.load(jsonData))
        except FileNotFoundError:
            expDirs.remove(i)
            
    totConf = {}
    for valType in expConfigs[0]:
        totConf[valType] = []

    for conf in expConfigs:
        for key,val in conf.items():
            totConf[key].append(val)
    
    with open(expFolder.joinpath("full_experiment_config.json"), "w") as writeFile:
        json.dump(totConf, writeFile, indent=4)

def extractSettings(expFolder):
    expData = []
    expDirs = [d for d in expFolder.iterdir() if expFolder.joinpath(d).is_dir()]
    allSet = {}
    with open (expFolder.joinpath(expDirs[0]).joinpath("settings.py")) as file:
        for line in file.readlines():
            words = line.split()
            if len(words) > 0:
                allSet[words[0]] = []

    for d in expDirs:
        #with open (expFolder + "/" + d + "/settings.py") as file:
        with open (expFolder.joinpath(d).joinpath("settings.py")) as file:
            for line in file.readlines():
                words = line.split()
                if len(words) > 0:
                    val = toNum(words[2])
                    if words[2][0] == "(":
                        val=""
                        val = val.join(words[2:]).strip("()").replace(" ", "").split(",")
                        val = [toNum(v) for v in val]
                    elif words[2][0] == "[":
                        val=""
                        val = val.join(words[2:]).strip("[]").replace(" ", "").split(",")
                        val = [toNum(v) for v in val]
                    allSet[words[0]].append(val)
    with open(expFolder.joinpath("full_experiment_settings.json"), "w") as writeFile:
        json.dump(allSet, writeFile, indent=4)

def toNum(val):
    constructors = [int, float, str]
    for c in constructors:
        try:
            val = c(val)
            return val
        except ValueError:
            pass

def openData(folder, selectIdx = "latest"):
    if selectIdx == "latest":
        summaryName = "full_experiment_summary.json"
    elif isinstance(selectIdx, int):
        summaryName = "full_experiment_summary_" + str(selectIdx) +".json"
    with open(folder.joinpath(summaryName), "r") as f:
        summary = json.load(f)

    with open(folder.joinpath("full_experiment_settings.json"), "r") as f:
        settings = json.load(f)
    with open(folder.joinpath("full_experiment_config.json"), "r") as f:
        config = json.load(f)
    with open(folder.joinpath("full_experiment_filtermetadata.json"),"r") as f:
        filtParams = json.load(f)
    return summary, settings, config, filtParams

    
def findFilterChangingEntries(allParams):
    changingEntries = {}
    for filtName, params in allParams.items():
        changingEntries[filtName] = findChangingEntries(params)
    return changingEntries

def findChangingEntries(settings):
    changingEntries = []
    for entry in settings:
        for val1,val2 in it.combinations(settings[entry],2):
            if val1 != val2:
                changingEntries.append(entry)
                break 
    return changingEntries




if __name__ == "__main__":
    expFolder = "multi_experiments/" + "test_plots_2020_02_25_13_13"
    extractSettings(expFolder)
    extractSummaries(expFolder)
    generateFullExpData(expFolder)