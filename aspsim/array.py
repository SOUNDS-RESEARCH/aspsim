import numpy as np
import dill
import json
from abc import ABC
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import aspsim.diagnostics.plot as dplot
import aspsim.room.roomimpulseresponse as rir
import aspsim.room.region as reg
import aspsim.room.trajectory as tr

class ArrayCollection():
    def __init__(self):
        self.arrays = {}
        self.names_mic = []
        self.names_src = []
        self.names_free_src = []
        self.paths = {}
        self.path_type = {}

        self.path_info = {}

    def __getitem__(self, key):
        return self.arrays[key]

    def __iter__(self):
        for ar in self.arrays.values():
            yield ar
    
    def __contains__(self, arrayName):
        return arrayName in self.arrays

    def save_metadata(self, filepath):
        if filepath is not None:
            self._save_metadata_arrays(filepath)
            self._save_metadata_paths(filepath)
            self._save_readable_pos(filepath)
        
    def _save_metadata_arrays(self, filepath):
        array_info = {}
        for arName, ar in self.arrays.items():
            array_info[arName] = ar.metadata
        with open(filepath.joinpath("metadata_arrays.json"), "w") as f:
            json.dump(array_info, f, indent=4)

    def _save_metadata_paths(self, filepath):
        #path_info = {}
        #for src, mic in self.mic_src_combos():
        #    path_info[f"{src.name}->{mic.name}"] = self.path_type[src.name][mic.name]
        with open(filepath.joinpath("metadata_paths.json"), "w") as f:
            json.dump(self.path_info, f, indent=4)

    def _save_readable_pos(self, filepath):
        pos = {}
        for arName, ar in self.arrays.items():
            pos[arName] = ar.pos.tolist()
        with open(filepath.joinpath("array_pos.json"), "w") as f:
            json.dump(pos, f, indent=4)


    @staticmethod
    def prototype_equals(prototype, initialized):
        """prototype and initialized are both instances of ArrayCollection
            The prototype has positions but not necessarily all paths created. They can still be strings
            This function checks if the initialized collection could be created from the prototype. i.e. if it 
            would make sense to load initialized instead of calculating the final values of the prototype"""
        #check that they have the same array names
        if prototype.arrays.keys() != initialized.arrays.keys():
            return False

        #Check that the arrays are the same
        for ar_name, ar in prototype.arrays.items():
            #if ar != initialized[ar_name]:
            #    return False
            if type(ar) is not type(initialized[ar_name]):
                return False
            if ar.num != initialized[ar_name].num:
                return False
            if not np.allclose(ar.pos,initialized[ar_name].pos):
                return False

        #Check the paths are the same
        for src, mic in prototype.mic_src_combos():
            if prototype.path_type[src.name][mic.name] == "modified":
                if not np.allclose(prototype.paths[src.name][mic.name], initialized.paths[src.name][mic.name]):
                    return False
            if prototype.path_type[src.name][mic.name] != initialized.path_type[src.name][mic.name]:
                return False

        return True

    def set_default_path_type(self, path_type):
        for src, mic in self.mic_src_combos():
            if not mic.name in self.path_type[src.name] and not mic.name in self.paths[src.name]:
                self.path_type[src.name][mic.name] = path_type
            elif mic.name in self.paths[src.name]:
                self.path_type[src.name][mic.name] = "modified"

    def empty(self):
        return len(self.arrays) == 0

    def sources(self):
        """Iterate over all source arrays"""
        for name in self.names_src:
            yield self.arrays[name]
    
    def mics(self):
        """Iterator over all mic arrays"""
        for name in self.names_mic:
            yield self.arrays[name]

    def free_sources(self):
        for name in self.names_free_src:
            yield self.arrays[name]

    def mic_src_combos(self):
        """Iterator over the euclidian product of the mics and sources"""
        for src_name in self.names_src:
            for mic_name in self.names_mic:
                yield self.arrays[src_name], self.arrays[mic_name]

    def iter_paths(self):
        for src_name in self.names_src:
            for mic_name in self.names_mic:
                yield self.arrays[src_name], self.arrays[mic_name], self.paths[src_name][mic_name]

    def add_array(self, array):
        assert array.name not in self.arrays
        self.arrays[array.name] = array
        
        if array.is_mic:
            self.names_mic.append(array.name)
        elif array.is_source:
            self.names_src.append(array.name)
            self.paths[array.name] = {}
            self.path_type[array.name] = {}
            if isinstance(array, FreeSourceArray):
                self.names_free_src.append(array.name)
        else:
            raise ValueError("Array is neither source nor microphone")

    def set_prop_paths(self, paths):
        for src_name, src_paths in paths.items():
            for mic_name, path in src_paths.items():
                self.set_prop_path(path, src_name, mic_name)

    def set_prop_path(self, path, src_name, mic_name):
        assert src_name in self.names_src
        assert mic_name in self.names_mic
        assert isinstance(path, np.ndarray)
        assert path.shape[0] == self.arrays[src_name].num
        assert path.shape[1] == self.arrays[mic_name].num
        assert path.ndim == 3
        self.paths[src_name][mic_name] = path
    
    def set_path_types(self, path_types):
        for src_name, src_path_types in path_types.items():
            for mic_name, pt in src_path_types.items():
                assert src_name in self.names_src
                assert mic_name in self.names_mic
                assert isinstance(pt, str)
                self.path_type[src_name][mic_name] = pt


    def setup_ir(self, sim_info):
        """ """
        metadata = {}
        self.sim_info = sim_info
        for src, mic in self.mic_src_combos():
            self.path_info[f"{src.name}->{mic.name}"] = {}

            if src.num == 0 or mic.num == 0:
                reverb = "none"
            else:
                reverb = self.path_type[src.name][mic.name]

            self.path_info[f"{src.name}->{mic.name}"]["type"] = reverb
            print(f"{src.name}->{mic.name} has propagation type: {reverb}")
            if reverb != "modified":
                self.paths[src.name][mic.name], path_info = self.create_path(src, mic, reverb, sim_info, True, True)
                for key, val in path_info.items():
                    self.path_info[f"{src.name}->{mic.name}"][key] = val
            

    def create_path (self, src, mic, reverb, sim_info, return_path_info=False, verbose=False):
        path_info = {}
        if reverb == "none": 
            path = np.zeros((src.num, mic.num, 1))
        elif reverb == "identity":
            path = np.ones((src.num,mic.num, 1))
        elif reverb == "isolated":
            assert src.num == mic.num
            path = np.eye(src.num, mic.num)[...,None]
        elif reverb == "random":
            path = self.rng.normal(0, 1, size=(src.num, mic.num, sim_info.max_room_ir_length))
        elif reverb == "ism":
            if sim_info.spatial_dims == 3:
                path = rir.ir_room_image_source_3d(
                        src.pos, mic.pos, sim_info.room_size, sim_info.room_center, 
                        sim_info.max_room_ir_length, sim_info.rt60, 
                        sim_info.samplerate, sim_info.c,
                        randomized_ism = sim_info.randomized_ism,
                        calculate_metadata=return_path_info,
                        verbose = verbose)
                if return_path_info:
                    path, path_info["ism_info"] = path
            else:
                raise ValueError
        elif reverb == "freespace":
            if sim_info.spatial_dims == 3:
                path = rir.ir_point_source_3d(
                src.pos, mic.pos, sim_info.samplerate, sim_info.c)
            elif sim_info.spatial_dims == 2:
                path = rir.ir_point_source_2d(
                    src.pos, mic.pos, sim_info.samplerate, sim_info.c
                )
            else:
                raise ValueError
        #elif reverb == "modified":
        #    pass
        else:
            raise ValueError
        if return_path_info:
            return path, path_info
        return path

    def update_path(self, src, mic):
        reverb = self.path_type[src.name][mic.name]
        assert reverb != "modified"
        self.paths[src.name][mic.name] = self.create_path(src, mic, reverb, self.sim_info)

    def update(self, glob_idx):
        #1. update arrays pos/properties
        #2. get info about which arrays actually changed
        #3. update the paths connecting the updated arrays
        if glob_idx % self.sim_info.array_update_freq != 0:
            return 

        changed_arrays = []

        for ar_name, ar in self.arrays.items():
            changed = ar.update(glob_idx)
            if changed:
                changed_arrays.append(ar_name)

        already_updated = []
        
        for ar_name in changed_arrays:
            if self.arrays[ar_name].is_mic:
                for src in self.sources():
                    if not src.name in already_updated:
                        self.update_path(src, self.arrays[ar_name])
            elif self.arrays[ar_name].is_source:
                for mic in self.mics():
                    if not mic.name in already_updated:
                        self.update_path(self.arrays[ar_name], mic)
            else:
                raise ValueError("Array must be mic or source")
            already_updated.append(ar_name)



    def plot(self, sim_info, fig_folder, print_method):
        fig, ax = plt.subplots()
        for ar in self.arrays.values():
            ar.plot(ax, sim_info)

        if "ism" in [self.path_type[src.name][mic.name] for src, mic in self.mic_src_combos()]:
            corner = [c - sz/2 for c, sz in zip(sim_info.room_center[:2], sim_info.room_size[:2])]
            #bottom_left_corner = sim_info.room_center[:2] - (sim_info.room_size[:2] / 2)
            width = sim_info.room_size[0]
            height = sim_info.room_size[1]
            ax.add_patch(patches.Rectangle(corner, width, height, edgecolor="k", facecolor="none", linewidth=2, alpha=1))

        ax.legend()
        ax.axis("equal")
        ax.grid(True)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        dplot.output_plot(print_method, fig_folder, "array_pos")

    def save_to_file(self, path):
        if path is not None:
            with open(path.joinpath("arrays.pickle"), "wb") as f:
                dill.dump(self, f)

def load_arrays(session_path):
        with open(session_path.joinpath("arrays.pickle"), "rb") as f:
            arrays = dill.load(f)
        return arrays






class Array(ABC):
    is_mic = False
    is_source = False
    plot_symbol = "."

    def __init__(self, name, pos):
        """
        Abstract base class for all types of arrays
        documentation refers to objects, which means microphone, loudspeaker
        or source depending on array type. 

        Parameters
        ----------
        name : str
            name of the array, which will also be the name of the
            signal associated with the array
        pos : ndarray of shape (num_objects, spatial_dim)
            or a list of ndarrays of the same shape, which will distribute
                the objects into groups.
            or a Trajectory object, which will produce a moving array
        """
        self.name = name

        if isinstance(pos, (list, tuple)):
            assert all([isinstance(p, np.ndarray) for p in pos])
            num_in_group = np.cumsum([0] + [p.shape[0] for p in pos])
            self.group_idxs = [np.arange(num_in_group[i], num_in_group[i+1]) for i in range(len(num_in_group)-1)]
            self.num_groups = len(pos)
            self.pos = np.concatenate(pos, axis=0)
            self.pos_segments = pos
            self.dynamic = False
        elif isinstance(pos, np.ndarray):
            self.group_idxs = None
            self.num_groups = 1
            self.pos = pos
            self.pos_segments = [pos]
            self.dynamic = False
        elif isinstance(pos, tr.Trajectory):
            self.trajectory = pos
            self.group_idxs = None
            self.num_groups = 1
            self.pos = self.trajectory.current_pos(0)
            self.pos_segments = [pos]
            self.dynamic = True
            self.pos_all = []
            self.time_all = []
        else:
            raise ValueError("Incorrect datatype for pos")

        self.num = self.pos.shape[0]
        assert self.pos.ndim == 2

        self.metadata = {
            "type" : self.__class__.__name__,
            "number" : self.num,
            "number of groups" : self.num_groups,
            "dynamic" : self.dynamic,
        }

    # def __eq__(self, other):
    #     if isinstance(other, Array):

    #         if self.pos != other.pos:
    #             return False
    #     return NotImplemented

    def set_groups(self, group_idxs):
        """group_idxs is a list of lists or a list of 1D nd.arrays
            Each inner list is the indices of the array elements 
            belonging to one group."""
        self.num_groups = len(group_idxs)
        self.group_idxs = group_idxs

    def plot(self, ax, sim_info):
        if self.dynamic:
            self.trajectory.plot(ax, self.plot_symbol, self.name, sim_info.tot_samples)
        else:
            ax.plot(self.pos[:,0], self.pos[:,1], self.plot_symbol, label=self.name, alpha=0.8)

    def update(self, glob_idx):
        if self.dynamic:
            self.pos = self.trajectory.current_pos(glob_idx)
            self.time_all.append(glob_idx)
            self.pos_all.append(self.pos)
            return True
        return False

class MicArray(Array):
    is_mic = True
    plot_symbol = "x"
    def __init__(self, name, pos):
        super().__init__(name, pos)

class RegionArray(MicArray):
    def __init__(self, name, region, pos=None):
        if isinstance(region, (list, tuple)):
            if len(region) > 1:
                self.region_segments = region
                region = reg.CombinedRegion(region)
            else:
                self.region_segments = region
                region = region[0]
        else:
            self.region_segments = [region]

        if pos is None:
            pos = [r.equally_spaced_points() for r in self.region_segments]
        super().__init__(name, pos)
        self.region = region
        self.metadata["region shape"] = self.region.__class__.__name__

    def plot(self, ax, sim_info):
        self.region.plot(ax, self.name)
        
class ControllableSourceArray(Array):
    is_source = True
    plot_symbol="o"
    def __init__(self, name, pos):
        super().__init__(name, pos)
        
class FreeSourceArray(Array):
    is_source = True
    plot_symbol = "s"
    def __init__(self, name, pos, source):
        super().__init__(name, pos)
        self.set_source(source)

        self.metadata["source info"] = self.source.metadata
    
    def reset_state(self):
        self.source = copy.deepcopy(self.source)
        self.source.reset()

    def get_samples(self, num_samples):
        return self.source.get_samples(num_samples)

    def set_source(self, source):
        assert source.num_channels == self.num
        self.source = source



