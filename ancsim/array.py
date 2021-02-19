from enum import Enum


class ArrayCollection():
    def __init__(self):
        self.arrays = {}
        self.names_mic = []
        self.names_src = []

    def __getitem__(self, key):
        return self.arrays[key]

    def __iter__(self):
        for ar in self.arrays.values():
            yield ar
    
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

    def paths(self):
        for src_name in self.names_src:
            for mic_name in self.names_mic:
                yield self.arrays[src_name], self.arrays[mic_name]

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
        elif array.is_source:
            self.names_src.append(array.name)
        else:
            raise ValueError


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

    def getSamples(numSamples):
        return self.source.getSamples(numSamples)
        
    def setSource(source):
        self.source = source