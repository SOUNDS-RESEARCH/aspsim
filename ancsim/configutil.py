import ancsim.utilities as util
from pathlib import Path
import yaml



class SimulatorInfo:
    def __init__(self, config):
        checkConfig(config)
        self.tot_samples = config["tot_samples"]
        self.sim_buffer = config["sim_buffer"]
        self.sim_chunk_size = config["sim_chunk_size"]
        self.save_source_contributions = config["save_source_contributions"]
        self.array_update_freq = config["array_update_freq"]
        
        self.samplerate = config["samplerate"]
        self.c = config["c"]
        self.spatial_dims = config["spatial_dims"]
        self.reverb = config["reverb"]

        self.room_size = config["room_size"]
        self.room_center = config["room_center"]
        self.rt60 = config["rt60"]
        self.max_room_ir_length = config["max_room_ir_length"]

        self.plot_begin = config["plot_begin"]
        self.chunk_per_export = config["chunk_per_export"]
        self.plot_output = config["plot_output"]
        self.output_smoothing = config["output_smoothing"]
        self.auto_save_load = config["auto_save_load"]

#SIMULATION PARAMETERS
PARAM_CATEGORIES = {"simulation" : ["tot_samples", "sim_buffer", "sim_chunk_size", "save_source_contributions", "array_update_freq"],
                    "audio" : ["samplerate", "c", "spatial_dims", "reverb"],
                    "ism" : ["room_size", "room_center", "rt60", "max_room_ir_length"],
                    "misc" : ["plot_begin", "chunk_per_export", 
                                "plot_output", "output_smoothing", 
                                "auto_save_load"]}

def getDefaultConfig():
    with open(Path(__file__).parent.joinpath("config.yaml")) as f:
        config = yaml.safe_load(f)
    return config

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
    #         (config["reverb"]=="ism" and key in PARAM_CATEGORIES["ism"]))}



def configMatch(config1, config2, path_types):
    return getAudioAffectingParams(config1, path_types) == \
            getAudioAffectingParams(config2, path_types)


