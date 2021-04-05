from enum import Enum
import numpy as np
import dill

class ArrayCollection():
    def __init__(self):
        self.arrays = {}
        self.names_mic = []
        self.names_src = []
        self.paths = {}
        self.path_type = {}

    def __getitem__(self, key):
        return self.arrays[key]

    def __iter__(self):
        for ar in self.arrays.values():
            yield ar

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
            if ar.num != initialized[ar_name].num:
                return False
            if ar.typ != initialized[ar_name].typ:
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

    def mic_src_combos(self):
        for src_name in self.names_src:
            for mic_name in self.names_mic:
                yield self.arrays[src_name], self.arrays[mic_name]

    def iter_paths(self):
        for src_name in self.names_src:
            for mic_name in self.names_mic:
                yield self.arrays[src_name], self.arrays[mic_name], self.paths[src_name][mic_name]

    def iter_paths_from(self, selectedSrc):
        assert isinstance(selectedSrc, ArrayType)
        for src, mic, path in self.iter_paths():
            if src.typ == selectedSrc:
                yield src, mic, path
    
    # def get_paths_src_subset(self, selectedSrc):
    #     ir = {}

    #     if isinstance(src, ArrayType):
    #         return {src, mic, path for self.iter_paths() if src.typ == selectedSrc}
    #     elif isinstance(src, str):
    #         return {src, mic, path for self.iter_paths() if src.name == selectedSrc}
        

    # def all_combos(self):
    #     for src_name in self.names_src:
    #         for mic_name in self.names_mic:
    #             yield self.arrays[src_name], self.arrays[mic_name], self.paths[src_name][mic_name]

    def of_type(self, typ):
        """typ is an ArrayType Enum object, 
            specifying which type of array to iterate over"""
        for ar in self.arrays.values():
            if ar.typ == typ:
                yield ar
    
    def add_array(self, array):
        assert array.name not in self.arrays
        self.arrays[array.name] = array
        
        if array.is_mic:
            self.names_mic.append(array.name)
            #for src_name in self.sources():
            #    self.paths[src_name][array.name] = {}
        elif array.is_source:
            self.names_src.append(array.name)
            self.paths[array.name] = {}
            self.path_type[array.name] = {}
        else:
            raise ValueError

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
        #elif isinstance(path, str):
        #    pass
        #else:
        #    raise ValueError
        self.paths[src_name][mic_name] = path
    
    def set_path_types(self, path_types):
        for src_name, src_path_types in path_types.items():
            for mic_name, pt in src_path_types.items():
                assert src_name in self.names_src
                assert mic_name in self.names_mic
                assert isinstance(pt, str)
                self.path_type[src_name][mic_name] = pt


class ArrayType(Enum):
    MIC = 0
    REGION = 1
    CTRLSOURCE = 2
    FREESOURCE = 3

    @property
    def is_mic(self):
        return self in (ArrayType.MIC, ArrayType.REGION)
    
    @property
    def is_source(self):
        return self in (ArrayType.CTRLSOURCE, ArrayType.FREESOURCE)




class Array():
    def __init__(self, name, typ, pos, source=None):
        self.name = name
        self.typ = typ
        if typ == ArrayType.REGION:
            self.region = pos
            pos = self.region.equally_spaced_points()
        elif typ == ArrayType.FREESOURCE:
            self.source = source

        self.pos = pos
        self.num = pos.shape[0]
        
        self.is_mic = typ.is_mic
        self.is_source = typ.is_source

    # @property
    # def plot_symbol(self):
    #     if self.is_source:
    #         return "o"
    #     elif self.is_mic:
    #         return "x"

    def getSamples(self, numSamples):
        return self.source.getSamples(numSamples)
        
    def setSource(self, source):
        assert source.numChannels == self.num
        #source.numChannels
        self.source = source

    def plot(self, ax):
        pass

    # def saveArray(self, pathToSave):
    #     pass

    # @staticmethod
    # def loadArray(self, pathToArray):
    #     pass