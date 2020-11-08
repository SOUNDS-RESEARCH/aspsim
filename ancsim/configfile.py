import ancsim.utilities as util


def getConfig():
    config = {}

    # ARRAYS
    config["NUMSOURCE"] = 1
    config["NUMREF"] = 1
    config["NUMERROR"] = 28
    config["NUMSPEAKER"] = 16
    config["NUMTARGET"] = 10**2 * 4
    # config["NUMTARGET"] = 32**2

    # SIMULATOR OPTIONS
    config["ENDTIMESTEP"] = 600000
    config["SIMBUFFER"] = 5000
    config["SIMCHUNKSIZE"] = 10000
    config["SAMPLERATE"] = 2000
    config[
        "C"
    ] = 343.0  # DO NOT CHANGE WHEN USING ISM REVERB, IT IS USING 343 INTERNALLY
    config["SPATIALDIMS"] = 3

    # SOURCES
    possibleSources = ["sine", "noise", "chirp", "recorded"]
    possibleAudioFiles = [
        "noise_bathroom_fan.wav",
        "song_assemble.wav",
        "arctic_a_speech_tight.wav",
        "secret_mountains_high_horse.wav",
        "secret_mountains_high_horse_instruments.wav",
        "port_st_willow_stay_even.wav",
        "port_st_willow_stay_even_instruments.wav",
        "night_panther_fire.wav",
        "night_panther_fire_instruments.wav",
        "leaf_come_around.wav",
        "leaf_come_around_instruments.wav",
        "james_may_all_souls_moon.wav",
        "james_may_all_souls_moon_instruments.wav",
    ]

    config["AUDIOFILENAME"] = "secret_mountains_high_horse_instruments.wav"
    config["SOURCETYPE"] = possibleSources[1]
    config["NOISEFREQ"] = 100
    config["NOISEBANDWIDTH"] = 400
    config["SOURCEAMP"] = 50

    # ARRAY PLACEMENT
    possibleShapes = ["circle", "rectangle", "doublerectangle", "cuboid"]
    config["ARRAYSHAPES"] = possibleShapes[1]
    config["TARGETWIDTH"] = 1
    config["TARGETHEIGHT"] = 0.2
    config["SPEAKERDIM"] = 3

    possibleTargetPoints = ["target_region", "image"]
    config["TARGETPOINTSPLACEMENT"] = possibleTargetPoints[0]

    irOptions = ["freespace", "ism", "recorded"]
    config["REVERB"] = irOptions[2]
    config["SPATIALDIMS"] = 3
    config["ROOMSIZE"] = [7, 5, 2.5]
    config["ROOMCENTER"] = [-1, 0, 0]
    config["RT60"] = 0.24
    config["MAXROOMIRLENGTH"] = 1024
    config["REFDIRECTLYOBTAINED"] = True

    # ADAPTIVE FILTER PARAMETERS
    config["MICSNR"] = 40
    config["BLOCKSIZE"] = 1024
    config["FILTLENGTH"] = 1024

    # OTHER ADAPTIVE FILTER PARAMETERS
    config["SPMFILTLEN"] = 1024
    config["MCPOINTS"] = 1000
    config["KERNFILTLEN"] = 155

    # PLOTS AND MISC
    config["SAVERAWDATA"] = True
    config["SAVERAWDATAFREQUENCY"] = 6
    config["PLOTFREQUENCY"] = 6
    config["GENSOUNDFIELDATCHUNK"] = 100000
    config["PLOTOUTPUT"] = "tikz"
    config["LOADSESSION"] = True
    config["OUTPUTSMOOTHING"] = 1024

    # configInstantCheck(config)
    return configInstantProcessing(config)


def configInstantProcessing(conf):
    if isinstance(conf["SOURCEAMP"], (int, float)):
        conf["SOURCEAMP"] = [conf["SOURCEAMP"] for _ in range(conf["NUMSOURCE"])]

    configInstantCheck(conf)
    return conf


def configInstantCheck(conf):
    if isinstance(conf["BLOCKSIZE"], list):
        for bs in conf["BLOCKSIZE"]:
            assert util.isPowerOf2(bs)

    assert conf["KERNFILTLEN"] % 2 == 1

    assert len(conf["SOURCEAMP"]) == conf["NUMSOURCE"]

    if conf["REVERB"] == "recorded":
        assert(conf["LOADSESSION"])

    assert(len(conf["SOURCEAMP"]) == conf["NUMSOURCE"])

def configPreprocessing(conf, numFilt):
    if isinstance(conf["BLOCKSIZE"], int):
        conf["BLOCKSIZE"] = [conf["BLOCKSIZE"] for _ in range(numFilt)]

    conf["LARGESTBLOCKSIZE"] = int(max(conf["BLOCKSIZE"]))

    configSimCheck(conf, numFilt)
    configInstantCheck(conf)
    return conf


def configSimCheck(conf, numFilt):
    assert numFilt == len(conf["BLOCKSIZE"])


# config = createConfig()
# config = configPreprocessing(config)
# configInstantCheck(config)
