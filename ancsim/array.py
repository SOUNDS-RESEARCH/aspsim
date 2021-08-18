import numpy as np
import dill
import json
from abc import ABC
import matplotlib.pyplot as plt

from ancsim.experiment.plotscripts import outputPlot
import ancsim.soundfield.roomimpulseresponse as rir
import ancsim.soundfield.geometry as geo

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

    # def iter_paths_from(self, selectedSrc):
    #     assert isinstance(selectedSrc, ArrayType)
    #     for src, mic, path in self.iter_paths():
    #         if src.typ == selectedSrc:
    #             yield src, mic, path
    
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


    def setupIR(self, sim_info):
        """ """
        print("Computing Room IR...")
        metadata = {}
        
        #self.set_default_path_type(sim_info.reverb)

        for src, mic in self.mic_src_combos():
            reverb = self.path_type[src.name][mic.name]

            self.path_info[f"{src.name}->{mic.name}"] = {}
            self.path_info[f"{src.name}->{mic.name}"]["type"] = reverb
            print(f"{src.name}->{mic.name} has propagation type: {reverb}")

            if reverb == "none": 
                self.paths[src.name][mic.name] = np.zeros((src.num, mic.num, 1))
            elif reverb == "identity":
                self.paths[src.name][mic.name] = np.ones((src.num,mic.num, 1))
            elif reverb == "isolated":
                assert src.num == mic.num
                self.paths[src.name][mic.name] = np.eye(src.num, mic.num)[...,None]
            elif reverb == "random":
                self.paths[src.name][mic.name] = self.rng.normal(0, 1, size=(src.num, mic.num, sim_info.max_room_ir_length))
            elif reverb == "ism":
                if sim_info.spatial_dims == 3:
                    self.paths[src.name][mic.name], ism_info = rir.irRoomImageSource3d(
                                                        src.pos, mic.pos, sim_info.room_size, sim_info.room_center, 
                                                        sim_info.max_room_ir_length, sim_info.rt60, 
                                                        sim_info.samplerate, sim_info.c,
                                                        calculateMetadata=True)
                    self.path_info[f"{src.name}->{mic.name}"]["ism_info"] = ism_info
                else:
                    raise ValueError
            elif reverb == "freespace":
                if sim_info.spatial_dims == 3:
                    self.paths[src.name][mic.name] = rir.irPointSource3d(
                    src.pos, mic.pos, sim_info.samplerate, sim_info.c)
                elif sim_info.spatial_dims == 2:
                    self.paths[src.name][mic.name] = rir.irPointSource2d(
                        src.pos, mic.pos, sim_info.samplerate, sim_info.c
                    )
                else:
                    raise ValueError
            elif reverb == "modified":
                pass
            else:
                raise ValueError
        #return metadata




    def plot(self, fig_folder, print_method):
        fig, ax = plt.subplots()
        for ar in self.arrays.values():
            ar.plot(ax)
        ax.legend()
        ax.axis("equal")
        ax.grid(True)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        outputPlot(print_method, fig_folder, "array_pos")


    # def get_paths_src_subset(self, selectedSrc):
    #     ir = {}

    #     if isinstance(src, ArrayType):
    #         return {src, mic, path for self.iter_paths() if src.typ == selectedSrc}
    #     elif isinstance(src, str):
    #         return {src, mic, path for self.iter_paths() if src.name == selectedSrc}
        
    # def of_type(self, typ):
    #     """typ is an ArrayType Enum object, 
    #         specifying which type of array to iterate over"""
    #     for ar in self.arrays.values():
    #         if ar.typ == typ:
    #             yield ar
    


# class ArrayType(Enum):
#     MIC = 0
#     REGION = 1
#     CTRLSOURCE = 2
#     FREESOURCE = 3

#     @property
#     def is_mic(self):
#         return self in (ArrayType.MIC, ArrayType.REGION)
    
#     @property
#     def is_source(self):
#         return self in (ArrayType.CTRLSOURCE, ArrayType.FREESOURCE)



class Array(ABC):
    is_mic = False
    is_source = False
    plot_symbol = "."

    def __init__(self, name, pos):
        self.name = name

        if isinstance(pos, (list, tuple)):
            assert all([isinstance(p, np.ndarray) for p in pos])
            num_in_group = np.cumsum([0] + [p.shape[0] for p in pos])
            self.group_idxs = [np.arange(num_in_group[i], num_in_group[i+1]) for i in range(len(num_in_group)-1)]
            self.num_groups = len(pos)
            self.pos = np.concatenate(pos, axis=0)
            self.pos_segments = pos
        else:
            assert isinstance(pos, np.ndarray)
            self.group_idxs = None
            self.num_groups = 1
            self.pos = pos
            self.pos_segments = [pos]

        self.num = self.pos.shape[0]
        assert self.pos.ndim == 2

        self.metadata = {
            "type" : self.__class__.__name__,
            "number" : self.num,
            "number of groups" : self.num_groups,
        }

    def set_groups(self, group_idxs):
        """group_idxs is a list of lists or a list of 1D nd.arrays
            Each inner list is the indices of the array elements 
            belonging to one group."""
        self.num_groups = len(group_idxs)
        self.group_idxs = group_idxs

        
    def plot(self, ax):
        ax.plot(self.pos[:,0], self.pos[:,1], self.plot_symbol, label=self.name, alpha=0.8)

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
                region = geo.CombinedRegion(region)
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

    def plot(self, ax):
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
        self.setSource(source)

        self.metadata["source info"] = self.source.metadata

    def getSamples(self, numSamples):
        return self.source.getSamples(numSamples)

    def setSource(self, source):
        assert source.numChannels == self.num
        self.source = source


