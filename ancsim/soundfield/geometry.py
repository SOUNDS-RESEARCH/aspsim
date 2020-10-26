import numpy as np
from abc import ABC, abstractmethod

class Region (ABC):
    def __init__ (self):
        pass


    def equally_spaced_in_volume(self):
        pass

    def sample_points(self, num_points):
        pass
