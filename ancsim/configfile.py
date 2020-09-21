import ancsim.utilities as util
import numpy as np

def getConfig():
    config = {}

    #SOURCES
    possibleSources = ["sine","noise", "chirp", "recorded"]
    possibleAudioFiles = ["noise_bathroom_fan.wav", "song_assemble.wav"]

    config["AUDIOFILENAME"] = possibleAudioFiles[1]
    config["SOURCETYPE"] = possibleSources[3]
    config["NOISEFREQ"] = 200
    config["NOISEBANDWIDTH"] = 50
    config["SOURCEAMP"] = (50, 100, 50, 25, 20, 15)

    #ROOM AND SETTING
    possibleShapes = ["circle", "rectangle"]
    config["ARRAYSHAPES"] = possibleShapes[1]
    config["TARGETWIDTH"] = 1
    config["TARGETHEIGHT"] = 0.2

    config["REALIMPULSERESPONSES"] = True
    config["SPATIALDIMENSIONS"] = 3
    config["REVERBERATION"] = True
    config["ROOMSIZE"] = [7, 5, 2.5]
    config["ROOMCENTER"] = [-1, 0, 0]
    config["RT60"] = 0.24
    config["MAXROOMIRLENGTH"] = 1024

    config["REFDIRECTLYOBTAINED"] = True

    #ADAPTIVE FILTER PARAMETERS
    config["BLOCKSIZE"] = 1024

    #KERNEL INTERPOLATION
    config["MCPOINTS"] = 1000
    config["KERNFILTLEN"] = 155

    #SECONDARY PATH MODELLING
    config["SPMFILTLEN"] = 1024

    #PLOTS AND MISC
    config["SAVERAWDATA"] = False
    config["SAVERAWDATAFREQUENCY"] = 5
    config["PLOTOUTPUT"] = "tikz"
    config["LOADSESSION"] = True

    configInstantCheck(config)
    return config

def configInstantCheck(conf):
    if isinstance(conf["BLOCKSIZE"], list):
        for bs in conf["BLOCKSIZE"]:
            assert(util.isPowerOf2(bs))

    assert(conf["KERNFILTLEN"] % 2 == 1)

    if conf["REALIMPULSERESPONSES"]:
        assert(conf["LOADSESSION"])


def configPreprocessing(conf, numFilt):
    if isinstance(conf["BLOCKSIZE"], int):
        conf["BLOCKSIZE"] = [conf["BLOCKSIZE"] for _ in range(numFilt)]

    conf["LARGESTBLOCKSIZE"] = int(np.max(conf["BLOCKSIZE"]))

    configSimCheck(conf, numFilt)
    configInstantCheck(conf)
    return conf
    
def configSimCheck(conf, numFilt):
    assert(numFilt == len(conf["BLOCKSIZE"]))
    

#config = createConfig()
#config = configPreprocessing(config)
#configInstantCheck(config)


