import ancsim.utilities as util
import numpy as np

def getConfig():
    config = {}

    #SOURCES
    possibleSources = ["sine","noise", "chirp", "recorded"]
    possibleAudioFiles = ["noise_bathroom_fan.wav", "song_assemble.wav"]

    config["AUDIOFILENAME"] = possibleAudioFiles[0]
    config["SOURCETYPE"] = possibleSources[1]
    config["NOISEFREQ"] = 150
    config["NOISEBANDWIDTH"] = 200

    #ROOM AND SETTING
    possibleTests = ["1d","2d","3dDiscFreespace","3dDiscFreespaceb","3dRectFreespace", "3dRectReverb"]
    config["TESTSETTING"] = possibleTests[5]

    possibleShapes = ["circle", "rectangle"]
    config["ARRAYSHAPES"] = possibleShapes[1]
    config["TARGETWIDTH"] = 1
    config["TARGETHEIGHT"] = 0.2

    config["SPATIALDIMENSIONS"] = 3
    config["REFDIRECTLYOBTAINED"] = True
    config["REVERBERATION"] = True
        
    config["ROOMSIZE"] = [7, 5, 2.5]
    config["ROOMCENTER"] = [-1, 0, 0]
    config["RT60"] = 0.24

    config["MAXROOMIRLENGTH"] = 1024


    #ADAPTIVE FILTER PARAMETERS
    config["LEARNINGFACTOR"] = 1

    config["BLOCKSIZE"] = 1024

    #KERNEL INTERPOLATION
    config["MCPOINTS"] = 100
    config["KERNFILTLEN"] = 155
    config["KERNELREG"] = 0.0001

    #SECONDARY PATH MODELLING
    config["SPMFILTLEN"] = 1024#config["MAXROOMIRLENGTH"]

    #PLOTS AND MISC
    config["SAVERAWDATA"] = False
    config["SAVERAWDATAFREQUENCY"] = 5
    config["PLOTOUTPUT"] = "pdf"
    config["LOADSESSION"] = True

    configInstantCheck(config)
    return config

def configInstantCheck(conf):
    if isinstance(conf["BLOCKSIZE"], list):
        for bs in conf["BLOCKSIZE"]:
            assert(util.isPowerOf2(bs))

    assert(conf["KERNFILTLEN"] % 2 == 1)


def configPreprocessing(conf, numFilt):
    if isinstance(conf["BLOCKSIZE"], int):
        conf["BLOCKSIZE"] = [conf["BLOCKSIZE"] for _ in range(numFilt)]

    conf["LARGESTBLOCKSIZE"] = int(np.max(conf["BLOCKSIZE"]))

    configSimCheck(conf, numFilt)
    return conf
    
def configSimCheck(conf, numFilt):
    assert(numFilt == len(conf["BLOCKSIZE"]))
    

#config = createConfig()
#config = configPreprocessing(config)
#configInstantCheck(config)


