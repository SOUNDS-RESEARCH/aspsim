
import numpy as np
import json



def addToSummary(diagName, summaryValues, timeIdx, folderPath):
    fullPath = folderPath.joinpath("summary_"+str(timeIdx)+".json")
    try:
        with open(fullPath,"r") as f:
            summary = json.load(f)
            summary[diagName] = summaryValues
            #totData = {**oldData, **dictToAdd}
    except FileNotFoundError:
        summary = {}
        summary[diagName] = summaryValues
    with open(fullPath, "w") as f:
        json.dump(summary, f, indent=4)


def meanNearTimeIdx(outputs, timeIdx):
    summaryValues = {}
    numToAverage = 2000
    
    for filtName, output in outputs.items():
        #val = diagnostic.getOutput()[timeIdx-numToAverage:timeIdx]
        val = output[timeIdx-numToAverage:timeIdx]
        filterArray = np.logical_not(np.isnan(val))
        summaryValues[filtName] = np.mean(val[filterArray])
    return summaryValues

def lastValue():
    raise NotImplementedError