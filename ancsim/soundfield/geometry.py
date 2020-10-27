import numpy as np
from abc import ABC, abstractmethod


class Region(ABC):
    def __init__(self):
        pass

    @sbstractmethod
    def is_in_region(self, coordinate):
        pass

    @abstractmethod
    def equally_spaced_points(self):
        pass

    @abstractmethod
    def sample_points(self, num_points):
        pass


class Cuboid(Region):
    def __init__(self, side_lengths, center=(0, 0, 0)):
        self.side_lengths = np.array(side_lengths)
        self.center = np.array(center)
        self.low_lim = self.center - self.side_lengths / 2
        self.high_lim = self.center + self.side_lengths / 2
        self.volume = np.prod(side_lengths)
        self.rng = np.random.RandomState(1)

    def is_in_region(self, coordinate):
        return all(
            [
                low_lim <= coord <= high_lim
                for low_lim, high_lim, coord in zip(
                    self.low_lim, self.high_lim, coordinate
                )
            ]
        )

    def equally_spaced_points(self, point_dist):
        if isinstance(point_dist, (int, float)):
            point_dist = np.ones(3) * point_dist
        else:
            point_dist = np.array(point_dist)

        num_points = np.floor(self.side_lengths / point_dist)
        assert min(num_points) >= 1

        single_axes = [
            np.arange(num_p) * p_dist for num_p, p_dist in zip(num_points, point_dist)
        ]
        remainders = [
            s_len - single_ax[-1]
            for single_ax, s_len in zip(single_axes, self.side_lengths)
        ]
        single_axes = [
            single_ax + remainder / 2
            for single_ax, remainder in zip(single_axes, remainders)
        ]
        all_points = np.meshgrid(*single_axes)
        all_points = np.concatenate(
            [points.flatten()[:, None] for points in all_points], axis=-1
        )
        all_points = all_points + self.low_lim
        return all_points

    def sample_points(self, num_points):
        samples = [
            self.rng.uniform(low_lim, high_lim, (num_points, 1))
            for low_lim, high_lim in zip(self.low_lim, self.high_lim)
        ]

        samples = np.concatenate(samples, axis=-1)
        return samples


class Cylinder(Region):
    def __init__(self):
        pass
