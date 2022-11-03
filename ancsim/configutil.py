from pathlib import Path
import yaml
import dataclasses as dc


@dc.dataclass
class SimulatorInfo:
    tot_samples : int
    sim_buffer : int
    sim_chunk_size : int

    save_source_contributions : bool
    array_update_freq : int

    samplerate : int
    c : float
    spatial_dims : int
    reverb : str

    room_size : list[float]
    room_center : list[float]
    rt60 : float
    max_room_ir_length : int

    plot_begin : int
    chunk_per_export : int
    plot_output : str
    output_smoothing : int
    auto_save_load : bool


    def __post_init__(self):
        assert len(self.room_size) == self.spatial_dims
        assert len(self.room_center) == self.spatial_dims

        assert self.reverb in ("none", "identity", "freespace", "ism", "recorded")


    def save_to_file(self, path):
        if path is not None:
            with open(path.joinpath("config.yaml"), "w") as f:
                yaml.dump(dc.asdict(self), f, sort_keys=False)

def load_from_file(path):
    if path.is_dir():
        path.joinpath("config.yaml")

    with open(path) as f:
        config = yaml.safe_load(f)
    return SimulatorInfo(**config)


def load_default_config():
    path = Path(__file__).parent.joinpath("config.yaml")
    return load_from_file(path)


def equal_audio(info1, info2, path_types):
    same_audio =  info1.samplerate == info2.samplerate and \
                info1.c == info2.c and \
                info1.spatial_dims == info2.spatial_dims and \
                info1.reverb == info2.reverb
    
    if any("ism" in src_path_types.values() for src_path_types in path_types.values()):
        same_audio = same_audio and \
            info1.room_size == info2.room_size and \
            info1.room_center == info2.room_center and \
            info1.rt60 == info2.rt60 and \
            info1.max_room_ir_length == info2.max_room_ir_length
    return same_audio







    # with open() as f:
    #     config = yaml.safe_load(f)
    # return config



# def save_config(path_to_save, config):
#     if path_to_save is not None:
#         with open(path_to_save.joinpath("config.yaml"), "w") as f:
#             yaml.dump(config, f, sort_keys=False)

# def load_config(session_path):
#     with open(session_path.joinpath("config.yaml")) as f:
#         config = yaml.safe_load(f)
#     return config

    # def __init__(self, config):
    #     check_config(config)
    #     self.tot_samples = config["tot_samples"]
    #     self.sim_buffer = config["sim_buffer"]
    #     self.sim_chunk_size = config["sim_chunk_size"]
    #     self.save_source_contributions = config["save_source_contributions"]
    #     self.array_update_freq = config["array_update_freq"]
        
    #     self.samplerate = config["samplerate"]
    #     self.c = config["c"]
    #     self.spatial_dims = config["spatial_dims"]
    #     self.reverb = config["reverb"]

    #     self.room_size = config["room_size"]
    #     self.room_center = config["room_center"]
    #     self.rt60 = config["rt60"]
    #     self.max_room_ir_length = config["max_room_ir_length"]

    #     self.plot_begin = config["plot_begin"]
    #     self.chunk_per_export = config["chunk_per_export"]
    #     self.plot_output = config["plot_output"]
    #     self.output_smoothing = config["output_smoothing"]
    #     self.auto_save_load = config["auto_save_load"]

#SIMULATION PARAMETERS
# PARAM_CATEGORIES = {"simulation" : ["tot_samples", "sim_buffer", "sim_chunk_size", "save_source_contributions", "array_update_freq"],
#                     "audio" : ["samplerate", "c", "spatial_dims", "reverb"],
#                     "ism" : ["room_size", "room_center", "rt60", "max_room_ir_length"],
#                     "misc" : ["plot_begin", "chunk_per_export", 
#                                 "plot_output", "output_smoothing", 
#                                 "auto_save_load"]}



# def check_config(conf):
#     #check that all config params are categorized
#     for key, val in conf.items():
#         categorized = False
#         for category, paramList in PARAM_CATEGORIES.items():
#             if key in paramList:
#                 categorized = True
#         assert categorized

#     #check that all categorized param names exists in the config
#     for category, paramList in PARAM_CATEGORIES.items():
#         for param in paramList:
#             assert param in conf



# def get_audio_affecting_params(config, path_types):
#     categories_to_check = ["audio"]
#     if any("ism" in src_path_types.values() for src_path_types in path_types.values()):
#         categories_to_check.append("ism")

#     filtered_dict = {}
#     for param_name, param_val in config.items():
#         for categ in categories_to_check:
#             if param_name in PARAM_CATEGORIES[categ]:
#                 filtered_dict[param_name] = param_val
#     return filtered_dict



