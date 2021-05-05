import ancsim.utilities as util
from pathlib import Path
import yaml

#SIMULATION PARAMETERS
PARAM_CATEGORIES = {"simulation" : ["ENDTIMESTEP", "SIMBUFFER", "SIMCHUNKSIZE", "MICSNR"],
                    "audio" : ["SAMPLERATE", "C", "SPATIALDIMS", "REVERB"],
                    "ism" : ["ROOMSIZE", "ROOMCENTER", "RT60", "MAXROOMIRLENGTH"],
                    "misc" : ["SAVERAWDATA", "SAVERAWDATAFREQUENCY", "PLOTFREQUENCY", 
                                "GENSOUNDFIELDATCHUNK", "PLOTOUTPUT", "OUTPUTSMOOTHING", 
                                "AUTOSAVELOAD"]}

def getBaseConfig():
    with open(Path(__file__).parent.joinpath("config.yaml")) as f:
        config = yaml.safe_load(f)
    return config

def processConfig(conf):
    #if isinstance(conf["BLOCKSIZE"], int):
    #    conf["BLOCKSIZE"] = [conf["BLOCKSIZE"] for _ in range(numFilt)]

    checkConfig(conf)
    return conf


def checkConfig(conf):
    #check that all config params are categorized
    for key, val in conf.items():
        categorized = False
        for category, paramList in PARAM_CATEGORIES.items():
            if key in paramList:
                categorized = True
        assert categorized

    #check that all categorized param names exists in the config
    for category, paramList in PARAM_CATEGORIES.items():
        for param in paramList:
            assert param in conf




    # if isinstance(conf["BLOCKSIZE"], list):
    #     for bs in conf["BLOCKSIZE"]:
    #         assert util.isPowerOf2(bs)

   # assert numFilt == len(conf["BLOCKSIZE"])




def getAudioAffectingParams(config, path_types):
    categories_to_check = ["audio"]
    if any("ism" in src_path_types.values() for src_path_types in path_types.values()):
        categories_to_check.append("ism")

    filtered_dict = {}
    for param_name, param_val in config.items():
        for categ in categories_to_check:
            if param_name in PARAM_CATEGORIES[categ]:
                filtered_dict[param_name] = param_val
    return filtered_dict

    # return {key:val for key,val in config.items() if 
    #         key in [PARAM_CATEGORIES[category_name]] or 
    #         (config["REVERB"]=="ism" and key in PARAM_CATEGORIES["ism"]))}



def configMatch(config1, config2, path_types):
    return getAudioAffectingParams(config1, path_types) == \
            getAudioAffectingParams(config2, path_types)