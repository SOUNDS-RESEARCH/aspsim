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

    export_frequency : int
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
        path = path.joinpath("config.yaml")

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

