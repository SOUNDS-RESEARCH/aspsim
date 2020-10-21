import numpy as np
np.random.seed(2)
import os
import sys
from pathlib import Path
import shutil
import json
import time
#from pathos.helpers import freeze_support

import ancsim.utilities as util
import ancsim.settings as s
from ancsim.configfile import configPreprocessing
#from ancsim.experiment.saveloadsession import loadSession
from ancsim.simulatorsetup import setupIR, setupPos, setupSource
from ancsim.simulatorlogging import getFolderName, addToSimMetadata, writeFilterMetadata
from ancsim.saveloadsession import saveSettings, saveConfig, loadSession, saveRawData

import ancsim.experiment.plotscripts as psc
import ancsim.experiment.multiexperimentutils as meu
from ancsim.adaptivefilter.diagnostics import PlotDiagnosticDispatcher

class Simulator:
    def __init__(self, config, folderForPlots=Path(__file__).parent.parent.joinpath("figs"), 
                    sessionFolder=None,
                    generateSubFolder=True):
        #PREPREPARATION
        self.filters = []
        self.config = config

        self.folderPath = getFolderName(config, folderForPlots, generateSubFolder)

            
        printInfo(config, self.folderPath)

        saveConfig(self.folderPath, config)
        saveSettings(self.folderPath)
        # with open (self.folderPath.joinpath("configfile.json"), "w") as writeConfig:
        #     json.dump(config, writeConfig, indent=4)
        # shutil.copy(packagePath.joinpath("settings.py"), str(self.folderPath.joinpath("settings.py")))

        #CONSTRUCTING SIMULATION
        if config["LOADSESSION"]:
            self.pos, self.sourceFilters, self.speakerFilters = \
                loadSession(config, s, sessionFolder, self.folderPath)
        else:
            self.pos = setupPos(config)
            self.sourceFilters, self.speakerFilters, irMetadata = setupIR(self.pos, config)
            addToSimMetadata(self.folderPath, irMetadata)
        self.noiseSource = setupSource(config, s.SAMPLERATE)

        #LOGGING AND DIAGNOSTICS
        plotAnyPos(self.pos, self.folderPath, config)
        # if config["SAVERAWDATA"]:
        #     meu.saveSetupData(self.folderPath, self.pos, self.sourceFilters, self.speakerFilters["evals"])

    def prepare(self, filters):
        assert(isinstance(filters, list))
        self.filters = filters
        self.config = configPreprocessing(self.config, len(self.filters))

        setUniqueFilterNames(self.filters)
        writeFilterMetadata(self.filters, self.folderPath)

        self.plotDispatcher = PlotDiagnosticDispatcher(self.filters, self.config["PLOTOUTPUT"])


    def runSimulation(self):
        print("Filling buffers...")
        noises = fillBuffers(self.filters, self.noiseSource, self.speakerFilters, self.sourceFilters, self.config)
        for filt in self.filters:
            filt.prepare()

        print("SIM START")
        n_tot = 0
        bufferIdx = -1
        noises = updateNoises(n_tot, noises,  self.noiseSource, self.sourceFilters, self.config["NUMEVALS"])
        noiseIndices = [s.SIMBUFFER for _ in range(len(self.filters))]
        while n_tot < s.ENDTIMESTEP-self.config["LARGESTBLOCKSIZE"]:
            bufferIdx += 1
            for n in range(s.SIMBUFFER, s.SIMBUFFER + min(s.SIMCHUNKSIZE, s.ENDTIMESTEP-n_tot)):
                if n_tot % 1000 == 0:
                    print("Timestep: ", n_tot)
                if s.ENDTIMESTEP-self.config["LARGESTBLOCKSIZE"] < n_tot:
                    break
                    
                if np.max([nIdx+bSize for nIdx, bSize in zip(noiseIndices, self.config["BLOCKSIZE"])]) >= s.SIMBUFFER+s.SIMCHUNKSIZE:
                    noises = updateNoises(n_tot, noises, self.noiseSource, self.sourceFilters)
                    noiseIndices = [i-s.SIMCHUNKSIZE for i in noiseIndices]

                for i, (filt, bSize, nIdx) in enumerate(zip(self.filters, self.config["BLOCKSIZE"], noiseIndices)):
                    if n_tot % bSize == 0:
                        filt.forwardPass(bSize, *[noiseAtPoints[:,nIdx:nIdx+bSize] for noiseAtPoints in noises])
                        filt.updateFilter()
                        noiseIndices[i] += bSize

                if n_tot % s.SIMCHUNKSIZE == 0 and n_tot > 0:# and bufferIdx % s.PLOTFREQUENCY == 0:
                    if self.config["PLOTOUTPUT"] != "none":
                        self.plotDispatcher.dispatch(self.filters, n_tot, bufferIdx, self.folderPath)
                    if self.config["SAVERAWDATA"] and (
                        bufferIdx % self.config["SAVERAWDATAFREQUENCY"] == 0 \
                            or bufferIdx-1 == 0):
                        saveRawData(self.filters, n_tot, self.folderPath)

                n_tot += 1
                
                
def printInfo(config, folderPath):
    print("Session folder: ", folderPath.name)
    print("Number of mics: ", config["NUMERROR"])
    print("Number of speakers: ", config["NUMSPEAKER"])


def setUniqueFilterNames(filters):
    names = []
    for filt in filters:
        newName = filt.name
        i = 1
        while newName in names:
            newName = filt.name + " " + str(i)
            i += 1
        names.append(newName)
        filt.name = newName


def fillBuffers(filters, noiseSource, speakerFilters, sourceFilters, config):
    noise = noiseSource.getSamples(s.SIMBUFFER)
    noises = [sf.process(noise) for sf in sourceFilters]

    maxSecPathLength = np.max([speakerFilt.shape[-1] for _, speakerFilt,  in speakerFilters.items()])
    fillStartIdx = config["LARGESTBLOCKSIZE"] + np.max((config["FILTLENGTH"]+config["KERNFILTLEN"], maxSecPathLength))
    fillNumBlocks = [(s.SIMBUFFER-fillStartIdx) // bs for bs in config["BLOCKSIZE"]]

    startIdxs = []
    for filtIdx, blockSize in enumerate(config["BLOCKSIZE"]):
        startIdxs.append([])
        for i in range(fillNumBlocks[filtIdx]):
            startIdxs[filtIdx].append(s.SIMBUFFER-((fillNumBlocks[filtIdx]-i)*blockSize))

    for filtIdx, (filt, blockSize) in enumerate(zip(filters, config["BLOCKSIZE"])):
        filt.idx = startIdxs[filtIdx][0]
        for i in startIdxs[filtIdx]:
            filt.forwardPass(blockSize, *[noiseAtPoints[:,i:i+blockSize] for noiseAtPoints in noises])

    return noises

def updateNoises(timeIdx, noises, noiseSource, sourceFilters, numEvals):
    noise = noiseSource.getSamples(s.SIMCHUNKSIZE)
    #if (timeIdx // s.SIMCHUNKSIZE) < s.GENSOUNDFIELDATCHUNK-2:

    #MADE TO NOT HAVE TO GENERATE SAMPLES TO EVALS
    #SHOULD BE EXCHANGED FOR THE OUTCOMMENTED BOTTOM EXPRESSION AT ELSE:
    noises = [np.concatenate((noiseAtPoints[:,-s.SIMBUFFER:], sf.process(noise)),axis=-1) 
                    for noiseAtPoints, sf in zip(noises[0:-1], sourceFilters[0:-1])]
    noises.append(np.zeros((numEvals,s.SIMCHUNKSIZE+s.SIMBUFFER)))
    #else:
    #    noises = [np.concatenate((noiseAtPoints[:,-s.SIMBUFFER:], sf.process(noise)),axis=-1) 
    #                    for noiseAtPoints, sf in zip(noises, sourceFilters)]
    return noises


def plotAnyPos(pos, folderPath, config):
    print("Setup Positions")
    if config["SPATIALDIMENSIONS"] == 3:
        if config["ARRAYSHAPES"] == "circle":
            psc.plotPos3dDisc(pos, folderPath,config, config["PLOTOUTPUT"])
        elif config["ARRAYSHAPES"] == "rectangle":
            if config["REVERBERATION"]:
                psc.plotPos3dRect(pos, folderPath, config, config["ROOMSIZE"], config["ROOMCENTER"], printMethod=config["PLOTOUTPUT"])
            else:
                psc.plotPos3dRect(pos, folderPath,config, printMethod=config["PLOTOUTPUT"])
    elif config["SPATIALDIMENSIONS"] == 2:
        if config["ARRAYSHAPES"] == "circle":
            if config["REFDIRECTLYOBTAINED"]:
                raise NotImplementedError
            else:
                psc.plotPos(pos, folderPath, config, config["PLOTOUTPUT"])
        else:
            raise NotImplementedError
    else:
        raise ValueError






# if __name__ == "__main__":
#     #freeze_support()
#     import configfile
#     if len(sys.argv) == 2:
#         folderForPlots = sys.argv[1]
#         sim(folderForPlots, config=configfile.config)
#     else:
#         sim(config=configfile.config)
    
    