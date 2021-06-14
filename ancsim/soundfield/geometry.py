import numpy as np
from abc import ABC, abstractmethod
import matplotlib.patches as patches


class Region(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def is_in_region(self, coordinate):
        pass

    @abstractmethod
    def equally_spaced_points(self):
        pass

    @abstractmethod
    def sample_points(self, num_points):
        pass

    @abstractmethod
    def plot(self, ax, label=None):
        pass

class Cuboid(Region):
    def __init__(self, side_lengths, center=(0, 0, 0), point_spacing=(1,1,1)):
        self.side_lengths = np.array(side_lengths)
        self.center = np.array(center)
        self.low_lim = self.center - self.side_lengths / 2
        self.high_lim = self.center + self.side_lengths / 2
        self.volume = np.prod(side_lengths)
        self.rng = np.random.default_rng(1)
        self.point_spacing = point_spacing

    def is_in_region(self, coordinate):
        return all(
            [
                low_lim <= coord <= high_lim
                for low_lim, high_lim, coord in zip(
                    self.low_lim, self.high_lim, coordinate
                )
            ]
        )

    def equally_spaced_points(self, point_dist=None):
        if point_dist is None:
            point_dist = self.point_spacing
        if isinstance(point_dist, (int, float)):
            point_dist = np.ones(3) * point_dist
        else:
            point_dist = np.array(point_dist)

        num_points = np.floor(self.side_lengths / point_dist)
        assert min(num_points) >= 1

        single_axes = [np.arange(num_p) * p_dist for num_p, p_dist in zip(num_points, point_dist)]
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

    def plot(self, ax, label=None):
        rect = patches.Rectangle(self.low_lim, self.side_lengths[0], self.side_lengths[1], fill=True, alpha=0.3, label=label)
        ax.add_patch(rect)


class Rectangle(Region):
    def __init__(self, side_lengths, center=(0, 0, 0), point_spacing=(1,1)):
        self.side_lengths =  np.array(side_lengths)
        self.center = np.array(center)
        self.point_spacing = np.array(point_spacing)

        self.low_lim = self.center[:2] - self.side_lengths / 2
        self.high_lim = self.center[:2] + self.side_lengths / 2
        self.area = np.prod(side_lengths)
        self.rng = np.random.default_rng(1)
        
    def is_in_region(self, coordinate):
        raise NotImplementedError

    def equally_spaced_points(self):
        num_points = np.floor(self.side_lengths / self.point_spacing)
        assert min(num_points) >= 1

        single_axes = [np.arange(num_p) * p_dist for num_p, p_dist in zip(num_points, self.point_spacing)]
        remainders = [s_len - single_ax[-1] for single_ax, s_len in zip(single_axes, self.side_lengths)]
        single_axes = [single_ax + remainder / 2 for single_ax, remainder in zip(single_axes, remainders)]
        all_points = np.meshgrid(*single_axes)
        all_points = np.concatenate([points.flatten()[:, None] for points in all_points], axis=-1)
        all_points = all_points + self.low_lim
        all_points = np.concatenate((all_points, np.full((all_points.shape[0],1), self.center[-1])), axis=-1)
        return all_points

    def sample_points(self, num_points):
        raise NotImplementedError

    def plot(self, ax, label=None):
        rect = patches.Rectangle(self.low_lim, self.side_lengths[0], self.side_lengths[1], fill=True, alpha=0.3, label=label)
        ax.add_patch(rect)


class Cylinder(Region):
    def __init__(self, radius, height, center=(0,0,0), point_spacing=(1,1,1)):
        self.radius = radius
        self.height = height
        self.center = np.array(center)
        self.point_spacing = np.array(point_spacing)
        #self.num_points_circle = num_points_circle
        #self.num_points_height = num_points_height
        self.rng = np.random.default_rng(10)
        self.volume = self.radius**2 * np.pi * self.height

    def is_in_region(self, coordinate):
        raise NotImplementedError

    def equally_spaced_points(self):
        point_dist = self.point_spacing
        block_dims = np.array([self.radius*2, self.radius*2, self.height])
        num_points = np.ceil(block_dims / point_dist)
        single_axes = [np.arange(num_p) * p_dist for num_p, p_dist in zip(num_points, point_dist)]
        all_points = np.concatenate(np.meshgrid(*single_axes), axis=-1).reshape(-1,3)

        shift = (num_points-1)*point_dist / 2
        all_points -= shift[None,:]

        inside_cylinder = np.sqrt(np.sum(all_points[:,:2]**2,axis=-1)) <= self.radius
        all_points = all_points[inside_cylinder,:]

        all_points += self.center[None,:]
        return all_points

    def sample_points(self, num_points):
        r = self.radius * np.sqrt(self.rng.uniform(0,1,num_points))
        angle = 2 * np.pi * self.rng.uniform(0,1,num_points)
        x = r * np.cos(angle) + self.center[0]
        y = r * np.sin(angle) + self.center[1]
        h = self.rng.uniform(-self.height/2,self.height/2, num_points) + self.center[-1]
        return np.stack((x,y,h), axis=1)

    def plot(self, ax, label=None):
        circ = patches.Circle(self.center[:2], self.radius, fill=True, alpha=0.3, label=label)
        ax.add_patch(circ)



# from abc import abstractmethod, ABC
# import numpy as np
# import itertools as it
# import ancsim.utilities as util

# class Shape(ABC):
#     def __init__(self):
#         self.area = 0
#         self.volume = 0

#     @abstractmethod
#     def draw(self, ax):
#         pass

#     @abstractmethod
#     def get_point_generator(self):
#         pass

#     @abstractmethod
#     def equivalently_spaced(self, num_points):
#         pass
    

# class Cuboid(Shape):
#     def __init__(self, size, center=(0,0,0)):
#         assert(len(size) == 3)
#         assert(len(center) == 3)
#         self.size = size
#         self.center = center
#         self.area = 2* (np.product(size[0:2]) + 
#                         np.product(size[1:3]) + 
#                         np.product(size[2:4]))
#         self.volume = np.product(size)

#         self._low_lim = center - (size / 2)
#         self._upper_lim = center + (size / 2)

#     def equivalently_spaced(self, grid_space):
#         n_points = self.size / grid_space
#         full_points = np.floor(n_points)
#         frac_points = n_points - full_points

#         np.linspace(self._low_lim[0], self._upper_lim[0], 2*n_points[0]+1)[1::2]

#     # def equivalently_spaced(self, num_points):
#     #     #pointsPerAxis = int(np.sqrt(numPoints/zNumPoints))
#     #     #p_distance = 0.05
#     #     #per_axis = self.size / p_distance

#     #     if self.size[0] == self.size[1]:
#     #         z_point_ratio = self.size[2] / self.size[0]
#     #         if util.isInteger(z_point_ratio):
                

#     #             self.size

#     #             #assert(np.isclose(pointsPerAxis**2*zNumPoints, numPoints))
#     #             x = np.linspace(self._low_lim[0], self._upper_lim[0], 2*pointsPerAxis+1)[1::2]
#     #             y = np.linspace(-dims[1]/2, dims[1]/2, 2*pointsPerAxis+1)[1::2]
#     #             z = np.linspace(-dims[2]/2, dims[2]/2, 2*zNumPoints+1)[1::2]
#     #             [xGrid, yGrid, zGrid] = np.meshgrid(x, y, z)
#     #             evalPoints = np.vstack((xGrid.flatten(), yGrid.flatten(), zGrid.flatten())).T

#     #             return evalPoints
#     #         else:
#     #             raise NotImplementedError
#     #     else:
#     #         raise NotImplementedError


        
#     def get_point_generator(self):
#         rng = np.random.RandomState(0)
#         def gen(num_samples):
#             points = np.vstack((rng.uniform(self._low_lim[0], self._upper_lim[0], num_samples)
#                                 rng.uniform(self._low_lim[1], self._upper_lim[1], num_samples)
#                                 rng.uniform(self._low_lim[2], self._upper_lim[2]], num_samples)))
#             return points
#         return gen



# class Cylinder(Shape):
#     def __init__(self, radius, height, center=(0,0,0)):
#         self.radius = radius
#         self.height = height
#         self.center = center
#         self.area = 2 * np.pi * (radius*height + radius**2)
#         self.volume = height * radius**2 * np.pi
        

#     def get_point_generator(self):
#         rng = np.random.RandomState(0)
#         def gen(numSamples):
#             r = np.sqrt(rng.rand(numSamples))*radius
#             angle = rng.rand(numSamples)*2*np.pi
#             x = r * np.cos(angle)
#             y = r * np.sin(angle)
#             z = rng.uniform(-height/2,height/2, size=numSamples)
#             points = np.stack((x,y,z))
#             return points
#         return gen
