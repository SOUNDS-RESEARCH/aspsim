import numpy as np
import datetime
import os
from pathlib import Path
import shutil
import time
from functools import wraps
import numpy as np

#only scalar for now
def isPowerOf2(x):
    return isInteger(np.log2(x))

def isInteger(x):
    if np.all(np.isclose(x, x.astype(int))):
        return True
    return False
    
def cart2pol(x, y):
    r = np.hypot(x,y)
    angle = np.arctan2(y, x)
    return (r, angle)

def pol2cart(r, angle):
    x = r * np.cos(angle)
    y = r * np.sin(angle)
    return(x, y)

def db2mag(db):
    return 10**(db / 20)

def mag2db(amp):
    return 20 * np.log10(amp)

def db2pow(db):
    return 10**(db / 10)

def pow2db (power):
    return 10 * np.log10(power)

def avgPower (signal, axis=None):
    return np.mean(np.abs(signal)**2, axis=axis)

def calcBlockSizes(numSamples, startIdx, blockSize):
    leftInBlock = blockSize-startIdx
    sampleCounter = 0
    blockSizes = []
    while sampleCounter < numSamples:
        blockLen = np.min((numSamples-sampleCounter, leftInBlock))
        blockSizes.append(blockLen)
        sampleCounter += blockLen
        leftInBlock -= blockLen 
        if leftInBlock == 0:
            leftInBlock = blockSize
    return blockSizes

def calcPowerRed(error, noise, currentIdx, avgLength):
    ePow = np.zeros((error.shape[0], error.shape[1]+avgLength-1))
    noisePow = np.zeros((noise.shape[0], noise.shape[1]+avgLength-1))
    for i in range(error.shape[0]):
        ePow[i,:] = np.convolve(error[i,:]**2, np.ones(avgLength), "full") / avgLength
        noisePow[i,:] = np.convolve(noise[i,:]**2, np.ones(avgLength), "full") / avgLength
    powRed = pow2db(np.mean(ePow / noisePow,axis=0))
    return powRed[:-avgLength+1]


# edited, originally from JBirdVegas, stackoverflow
def measure(name):
    def measure_internal(func):
        @wraps(func)
        def _time_it(*args, **kwargs):
            start = int(round(time.time() * 1000))
            try:
                return func(*args, **kwargs)
            finally:
                end_ = int(round(time.time() * 1000)) - start
                print(name + f" execution time: {end_ if end_ > 0 else 0} ms")
        return _time_it
    return measure_internal

def getTimeString(detailed=False):
    tm = datetime.datetime.now()
    timestr = str(tm.year) + "_" +\
                str(tm.month).zfill(2) + "_" +\
                str(tm.day).zfill(2) + "_" +\
                str(tm.hour).zfill(2) + "_" +\
                str(tm.minute).zfill(2)# + "_"+\
                #str(tm.second).zfill(2)
    if detailed:
        timestr += "_" + str(tm.second).zfill(2)
        timestr += "_" + str(tm.microsecond).zfill(2)
    return timestr


def getUniqueFolderName(prefix, parentFolder, detailedNaming=False):
    fileName = prefix + getTimeString(detailed=detailedNaming) 
    fileName += "_0"
    folderName = parentFolder.joinpath(fileName)
    if folderName.exists():
            idx = 1
            folderNameLen = len(folderName.name) - 2
            while folderName.exists():
                newName = folderName.name[:folderNameLen] + "_" + str(idx)
                folderName = folderName.parent.joinpath(newName)
                idx += 1
    #folderName += "/"
    return folderName

def getMultipleUniqueFolderNames(prefix, parentFolder, numNames):
    startPath = getUniqueFolderName(prefix, parentFolder)
    subFolderName = startPath.parts[-1]
    baseFolder = startPath.parent
        
    startIdx = int(subFolderName.split("_")[-1])
    startIdxLen = len(subFolderName.split("_")[-1])
    baseName = subFolderName[:-startIdxLen]
    
    folderNames = []
    for i in range(numNames):
        folderNames.append(baseFolder.joinpath(baseName + str(i+startIdx)))
        
    return folderNames

#simple tests
if __name__ == "__main__":
    x = np.arange(0,4)
    y = np.arange(4,0,-1)
    a = cart2pol(x, y)
    print(a[0])
    b = np.sqrt(x**2 + y**2)
    print(b)