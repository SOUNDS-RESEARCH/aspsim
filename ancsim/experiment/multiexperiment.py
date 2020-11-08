import dill

dill.settings["recurse"] = True
import pathos.multiprocessing as mp
from pathos.helpers import freeze_support

import numpy as np
import itertools as it
import os
import copy
from pathlib import Path
from enum import Enum
from copy import deepcopy
import operator as op

from ancsim.simulator import Simulator
import ancsim.utilities as util
from ancsim.configfile import getConfig
import ancsim.experiment.multiexperimentutils as meu
import ancsim.experiment.multiexperimentplots as mep


class DictIterator:
    def __init__(self, numDicts):
        self.numDicts = numDicts
        self.parameters = [{} for _ in range(numDicts)]
        self.paramIterators = []
        self.paramNames = [[] for _ in range(numDicts)]

    def addParameter(self, names, values, dictIndices):
        """names - list of strings which are keys for the dict
        values - list of lists or list of ndarray, with values to set key to
        dictIndices - list of integers, sets which dictionary the values are referring to"""
        if dictIndices == "all":
            names = [names for _ in range(self.numDicts)]
            values = [values for _ in range(self.numDicts)]
            dictIndices = range(self.numDicts)

        assert len(values) == len(names) == len(dictIndices)
        assert len(dictIndices) == len(set(dictIndices))  # No repeating indices
        numValues = len(values[0])
        assert all([numValues == len(val) for val in values])

        self.paramIterators.append(range(numValues))
        for name, val, dictIdx in zip(names, values, dictIndices):
            self.parameters[dictIdx][name] = val
            self.paramNames[dictIdx].append(name)

        for i in range(self.numDicts):
            if i not in dictIndices:
                self.paramNames[i].append(None)

    def iterate(self, fullDicts):
        assert len(fullDicts) == self.numDicts

        for allIndices in it.product(*self.paramIterators):
            newDicts = copy.deepcopy(fullDicts)
            for dictIdx, singleDictParamNames in enumerate(self.paramNames):
                for paramName, paramValueIdx in zip(singleDictParamNames, allIndices):
                    if paramName is not None:
                        newDicts[dictIdx][paramName] = self.parameters[dictIdx][
                            paramName
                        ][paramValueIdx]
            yield newDicts


class PARAMTYPE(Enum):
    config = 1
    filterArg = 2


class ParameterHandler:
    def __init__(self, numFilters):
        self.configParams = DictIterator(1)
        self.filtParams = DictIterator(numFilters)

    def generateParameters(self, baseConfig, baseFiltArguments):
        for config in self.configParams.iterate([baseConfig]):
            for filtArgs in self.filtParams.iterate(baseFiltArguments):
                yield config[0], filtArgs

    def addParameter(self, paramType, names, values, dictIndices=None):
        if paramType == PARAMTYPE.config:
            self.configParams.addParameter([names], [values], [0])
        elif paramType == PARAMTYPE.filterArg:
            self.filtParams.addParameter(names, values, dictIndices)


class ExperimentSpec:
    def __init__(self, baseConfig, baseFiltParameters, numFilters=1):
        """Baseparameters in list of dictionaries, with all
        arguments for the initialization of the used filterclasses"""
        assert len(baseFiltParameters) == numFilters
        self.baseConfig = baseConfig
        self.baseFiltParameters = baseFiltParameters

        self.betweenSets = ParameterHandler(numFilters)
        self.inSet = ParameterHandler(numFilters)

    def addParameter(self, names, values, inSet, paramType, filtIndices=None):
        if inSet:
            self.inSet.addParameter(paramType, names, values, filtIndices)
        else:
            self.betweenSets.addParameter(paramType, names, values, filtIndices)

    def genExperimentSets(self):
        for config, filtArgs in self.betweenSets.generateParameters(
            self.baseConfig, self.baseFiltParameters
        ):
            configList = []
            filtArgList = []
            for conf, fArg in self.inSet.generateParameters(config, filtArgs):
                configList.append(conf)
                filtArgList.append(fArg)
            yield configList, filtArgList


class SimAttribute:
    def __init__(self, attrName, itemName=None):
        self.attrName = attrName
        self.itemName = itemName

    def getValue(self, sim):
        # return deepcopy(getattr(sim, self.name))
        getter = op.attrgetter(self.attrName)
        val = getter(sim)
        if self.itemName is not None:
            val = val[self.itemName]
        return val


def insertSimAttributes(listOfParamDicts, sim):
    for filtIdx, paramDict in enumerate(listOfParamDicts):
        for paramName, paramValue in paramDict.items():
            if isinstance(paramValue, SimAttribute):
                listOfParamDicts[filtIdx][paramName] = copy.deepcopy(paramValue.getValue(sim))
    return listOfParamDicts

    #     if isinstance(filtArg, SimAttribute):
    #         filterArguments[i] = filtArg.getValue(sim)


# def runExperiment(spec, filterClasses, parentFolder=Path(__file__).parent.parent.parent.joinpath("multi_experiments/")):
#     mainFolder = util.getUniqueFolderName("full_exp_", parentFolder)
#     mainFolder.mkdir()


#     for config, parameters in spec.genExperimentSets():
#         setFolder = util.getUniqueFolderName("single_set_", mainFolder)
#         setFolder.mkdir()

#         numInSet = len(config)
#         mpArgs = [[setFolder for _ in range(numInSet)],
#                   [filterClasses for _ in range(numInSet)],
#                   config, parameters]

#         ncpu = mp.cpu_count()
#         #runSimMultiProcessing(*mpArgs)
#         with mp.ProcessingPool(ncpu) as p:
#             p.map(runSimMultiProcessing,*mpArgs)


def runExperiment(
    spec,
    filterClasses,
    parentFolder=Path(__file__).parent.parent.parent.joinpath("multi_experiments/"),
    sessionFolder=None,
):
    mainFolder = util.getUniqueFolderName("full_exp_", parentFolder)
    mainFolder.mkdir()
    ncpu = mp.cpu_count() - 2

    mpArgs = [[] for _ in range(5)]
    for newConfigs, newParams in spec.genExperimentSets():
        setFolder = util.getUniqueFolderName("single_set_", mainFolder)
        setFolder.mkdir()

        for conf, params in zip(newConfigs, newParams):
            # mpArgs[0].append(setFolder)
            mpArgs[1].append(filterClasses)
            mpArgs[2].append(conf)
            mpArgs[3].append(params)
            mpArgs[4].append(sessionFolder)
            if len(mpArgs[1]) == ncpu:
                mpArgs[0] = util.getMultipleUniqueFolderNames("figs_", setFolder, ncpu)
                with mp.ProcessingPool(ncpu) as p:
                    p.map(runSimMultiProcessing, *mpArgs)
                    mpArgs = [[] for _ in range(5)]

    if len(mpArgs[1]) > 0:
        mpArgs[0] = util.getMultipleUniqueFolderNames(
            "figs_", setFolder, len(mpArgs[1])
        )
        with mp.ProcessingPool(ncpu) as p:
            p.map(runSimMultiProcessing, *mpArgs)
    generateFullExpData(mainFolder)


def runSimMultiProcessing(
    mainFolder, filterClasses, config, filterArguments, sessionFolder
):
    sim = Simulator(config, mainFolder, sessionFolder, generateSubFolder=False)

    filterArguments = insertSimAttributes(filterArguments, sim)
    # for i, filtArg in enumerate(filterArguments):
    #     if isinstance(filtArg, SimAttribute):
    #         filterArguments[i] = filtArg.getValue(sim)

    filters = []
    for fc, params in zip(filterClasses, filterArguments):
        # filters.append(fc(speakerRIR=copy.deepcopy(sim.speakerFilters), **params))
        filters.append(fc(config=config, **params))
    sim.prepare(filters)
    sim.runSimulation()


def generateFullExpData(multiExpFolder, outputMethod="pdf"):
    for singleSetFolder in multiExpFolder.iterdir():
        if singleSetFolder.is_dir():
            generateSingleExpData(
                multiExpFolder.joinpath(singleSetFolder), outputMethod
            )


def generateSingleExpData(singleSetFolder, outputMethod="pdf"):
    meu.extractSettings(singleSetFolder)
    meu.extractSummaries(singleSetFolder, "latest")
    meu.extractAllSummaries(singleSetFolder)
    meu.extractConfigs(singleSetFolder)
    meu.extractFilterParameters(singleSetFolder)

    summary, settings, config, filtParams = meu.openData(singleSetFolder)
    entries = meu.findChangingEntries(config)
    for entry in entries:
        mep.plotSingleEntryMetrics(
            entry, summary, config, singleSetFolder, outputMethod
        )

    # if "NOISEFREQ" in entries:
    #     mep.reductionByFrequency(singleSetFolder, "latest")

    filtEntries = meu.findFilterChangingEntries(filtParams)
    for filtName, entries in filtEntries.items():
        for entry in entries:
            mep.plotSingleEntryMetrics(
                entry, summary, filtParams[filtName], singleSetFolder, outputMethod
            )

    # sfp.generateSoundfieldForFolder(singleSetFolder)


# def configDictGen(configParams, baseConfig):
#     paramNames, confIterables = getConfigIterables(configParams)

#     for configIndices in it.product(*confIterables):
#         params = copy.deepcopy(baseConfig)
#         for paramNum, i in enumerate(configIndices):
#             params[paramNames[paramNum]] = dictionary[paramNames[paramNum]][i]
#         yield params

# def filtParamGen(filtParams, baseFiltArguments):
#     filtParamNames, filtIterables = getFiltParamIterables(filtParams)


# def getConfigIterables(dictionary):
#     """input dictionary is all values for the experiment,
#     with the key being its config name. The values of the dicts is lists,
#     the euclidian product of all values in them should be iterated over

#     idxIterable is list of range(), if you iterate with it.product,
#     using the outputted values for the values in the input dictionary
#     you get the euclidian product for the full experiment"""
#     numKeys = []
#     keyNames = []
#     for key, val in dictionary.items():
#             numKeys.append(len(val))
#         keyNames.append(key)
#     idxIterable = [range(n) for n in numKeys]
#     return keyNames, idxIterable

# def getFiltParamIterables(parameterValues):
#     iterables = []
#     paramNames = []
#     for paramDict,filtIdx in enumerate(parameterValues):
#         iterables.append([])
#         paramNames.append([])
#         for paramName, paramValues in paramDict.values():
#             iterables[filtIdx].append(range(len(paramValues)))
#             paramNames[filtIdx].append(paramName)
#     return paramNames, iterables


# #==============================================================================0
# # DEPRECATED BELOW

# def runAllSpecs():
#     specFolders = []
#     for spec in allDefinedSpecs:
#         specFolders.append(runAllSets(spec))

#     for fld in specFolders:
#         generateFullExpData(fld)


# def runAllSets(spec):
#     # parentFolder = "multi_experiments/"
#     # mainFolder = util.getUniqueFolderName("full_exp_", parentFolder)
#     # os.mkdir(mainFolder)

#     for params in dictGenerator(spec.betweenSets, getConfig()):
#         dispatchExperimentSet(mainFolder, params, spec.inSet)
#     return mainFolder

# def dictToIterables(dictionary):
#     """input dictionary is all values for the experiment,
#         with the key being its config name. The values of the dicts is lists,
#         the euclidian product of all values in them should be iterated over

#         idxIterable is list of range(), if you iterate with it.product,
#         using the outputted values for the values in the input dictionary
#         you get the euclidian product for the full experiment"""
#     numKeys = []
#     keyNames = []
#     for key, val in dictionary.items():
#         try:
#             numKeys.append(val.shape[0])
#         except:
#             numKeys.append(len(val))
#         keyNames.append(key)
#     idxIterable = [range(n) for n in numKeys]
#     return keyNames, idxIterable

# def dictGenerator(dictionary, baseConfig):
#     keyNames, idxIterable = dictToIterables(dictionary)

#     for indices in it.product(*idxIterable):
#         params = copy.deepcopy(baseConfig)
#         for paramNum, i in enumerate(indices):
#             params[keyNames[paramNum]] = dictionary[keyNames[paramNum]][i]
#         yield params

# def dispatchExperimentSet(parentFolder, baseConfig, insetParams):
#     confSet = []
#     for params in dictGenerator(insetParams, baseConfig):
#         confSet.append(params)

#     folder = util.getUniqueFolderName("single_set_", parentFolder)
#     os.mkdir(folder)

#     mpArgs = [[folder for _ in range(len(confSet))], confSet]
#     ncpu = mp.cpu_count()
#     with mp.ProcessingPool(int(2)) as p:
#         p.map(sim,*mpArgs)


if __name__ == "__main__":
    freeze_support()
    runAllSpecs()
