import numpy as np
import json


def add_to_summary(diagName, summaryValues, timeIdx, folderPath):
    fullPath = folderPath.joinpath("summary_" + str(timeIdx) + ".json")
    try:
        with open(fullPath, "r") as f:
            summary = json.load(f)
            summary[diagName] = summaryValues
            # totData = {**oldData, **dictToAdd}
    except FileNotFoundError:
        summary = {}
        summary[diagName] = summaryValues
    with open(fullPath, "w") as f:
        json.dump(summary, f, indent=4)


def mean_near_time_idx(outputs, timeIdx):
    summaryValues = {}
    numToAverage = 3000

    for filtName, output in outputs.items():
        # val = diagnostic.getOutput()[timeIdx-numToAverage:timeIdx]
        val = output[...,timeIdx - numToAverage : timeIdx]
        filterArray = np.logical_not(np.isnan(val))
        summaryValues[filtName] = np.mean(val[filterArray])
    return summaryValues


def last_value():
    raise NotImplementedError
