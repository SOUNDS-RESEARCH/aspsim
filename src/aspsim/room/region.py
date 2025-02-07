import numpy as np
from abc import ABC, abstractmethod
import matplotlib.patches as patches


class Region(ABC):
    def __init__(self, rng=None):
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()
        
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

class CombinedRegion(Region):
    def __init__(self, regions, rng=None):
        super().__init__(rng)
        self.regions = regions
        assert not self._overlaps(), "Not implemented for overlapping regions"

        self.volumes = np.array([reg.volume for reg in self.regions])
        self.volume = np.sum(self.volumes)

    def _overlaps(self):
        for r in self.regions:
            points_fixed = r.equally_spaced_points()
            points_sampled = r.sample_points(100)
            for other_reg in self.regions:
                if r is other_reg:
                    continue
                if np.any(other_reg.is_in_region(points_fixed)) or \
                    np.any(other_reg.is_in_region(points_sampled)):
                    return True
        return False

    def is_in_region(self, coordinate):
        return np.logical_or([reg.is_in_region(coordinate) for reg in self.regions])

    def equally_spaced_points(self):
        return np.concatenate([reg.equally_spaced_points() for reg in self.regions], axis=0)

    def sample_points(self, num_points):
        vol_ratio = self.volumes / self.volume
        sample_limit = np.cumsum(vol_ratio)
        test_val = self.rng.uniform(0,1,num_points)
        sample_result = test_val[None,:] < sample_limit[:,None]
        reg_idx = np.sum(sample_result, axis=0) - 1
        unique, counts = np.unique(reg_idx, return_counts=True)
        sampled_points = np.concatenate([self.regions[idx].sample_points(num) 
                        for idx, num in zip(unique, counts)])
        assert sampled_points.shape[0] == num_points
        return sampled_points

    def plot(self, ax, label=None):
        for reg in self.regions:
            reg.plot(ax, label)
            

class Cuboid(Region):
    def __init__(self, side_lengths, center=(0, 0, 0), point_spacing=(1,1,1), rng=None):
        super().__init__(rng)
        self.side_lengths = np.array(side_lengths)
        self.center = np.array(center)
        self.low_lim = self.center - self.side_lengths / 2
        self.high_lim = self.center + self.side_lengths / 2
        self.volume = np.prod(side_lengths)
        self.point_spacing = point_spacing

    def is_in_region(self, coordinates, padding=[[0,0,0]]):
        is_in_coord_wise = np.logical_and(coordinates >= self.low_lim[None,:]-padding,
                                        coordinates <= self.high_lim[None,:]+padding)
        is_in = np.logical_and(np.logical_and(is_in_coord_wise[:,0], 
                                                is_in_coord_wise[:,1]),
                                                is_in_coord_wise[:,2])
        return is_in

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
        low_lim = np.array(self.low_lim)
        high_lim = np.array(self.high_lim)

        samples = self.rng.uniform(low_lim[None,:], high_lim[None,:], size=(num_points, 3))

        # samples = [
        #     self.rng.uniform(low_lim, high_lim, (num_points, 1))
        #     for low_lim, high_lim in zip(self.low_lim, self.high_lim)
        # ]

        # samples = np.concatenate(samples, axis=-1)
        return samples

    def plot(self, ax, label=None):
        rect = patches.Rectangle(self.low_lim, self.side_lengths[0], self.side_lengths[1], fill=True, alpha=0.3, label=label)
        ax.add_patch(rect)


class Rectangle(Region):
    def __init__(self, side_lengths, center, point_spacing=(1,1), spatial_dim=3, rng=None):
        """Constructs a two-dimensional rectangle

        The rectangle can be placed in 3D space by setting a 3-dimensional center
        
        Parameters
        ----------
        side_lengths : array_like of shape (2,)
            Lengths of the sides of the rectangle.
        center : array_like of shape (2,) or (3,)
            Center of the cylinder. The length of the array determines the spatial dimension.
            If the array has length 3, the rectangle is placed in 3D space with the z-coordinate
            fixed at the value of the third element.
        point_spacing : array_like of shape (2,), optional
            Spacing between points in each direction, affects the selection of points
            for the equally_spaced_points method. The default is (1,1,1).
        spatial_dim : int, optional
            Spatial dimension of the rectangle. The default is 3. Must correspond to the length
            of the center array. 
        rng : np.random.Generator, optional
            Random number generator. The default is None, in which case a new generator
            with a random seed will be created. 
            The generator affects the sampling of points in the sample_points method, and so 
            should be supplied for a reproducible result.

        Returns
        -------
        rectangle : Rectangle

        Notes
        -----
        Although there are many more rectangles in 3D, here only rectangles with a fixed
        z-coordinate is considered. A rotation could in principle be applied to the rectangle
        for other orientations.
        """
        super().__init__(rng)
        assert spatial_dim in (2,3)
        assert len(center) == spatial_dim
        self.side_lengths =  np.array(side_lengths)
        self.center = np.array(center)
        self.point_spacing = np.array(point_spacing)
        self.spatial_dim = spatial_dim

        self.low_lim = self.center[:2] - self.side_lengths / 2
        self.high_lim = self.center[:2] + self.side_lengths / 2
        self.volume = np.prod(side_lengths)
        
    def is_in_region(self, coordinates, padding=[[0,0]]):
        """Checks whether the coordinate is within the region or not. 
        
        Parameters
        ----------
        coordinates : np.ndarray
            Shape (N,3) or (N, 2) where N is the number of coordinates to check. 
            Each row is a coordinate in 3D space.      
        padding : array_like of shape (2,), optional
            Padding to add to the region. The default is [[0,0]]. This can 
            be used to check whether a point is near the region, rather than strictly 
            inside it.
            
        Returns
        -------
        is_in : boolan np.ndarray of shape (N,)
            True if the coordinate is within the region, False otherwise.
        """
        if self.spatial_dim == 3:
            if not np.allclose(coordinates[:,2], self.center[2]):
                return False

        is_in_coord_wise = np.logical_and(coordinates[:,:2] >= self.low_lim[None,:]-padding,
                                        coordinates[:,:2] <= self.high_lim[None,:]+padding)
        is_in = np.logical_and(is_in_coord_wise[:,0], is_in_coord_wise[:,1])
        return is_in

    def equally_spaced_points(self):
        """Returns a grid of points within the region
        
        Returns
        -------
        all_points : np.ndarray of shape (num_points, 3) or (num_points, 2)
            The number of points returned is determined by the radius, height and point_spacing.
            Each row is a coordinate in 3D or 2D space.
        """
        num_points = np.floor(self.side_lengths / self.point_spacing)
        assert min(num_points) >= 1

        single_axes = [np.arange(num_p) * p_dist for num_p, p_dist in zip(num_points, self.point_spacing)]
        remainders = [s_len - single_ax[-1] for single_ax, s_len in zip(single_axes, self.side_lengths)]
        single_axes = [single_ax + remainder / 2 for single_ax, remainder in zip(single_axes, remainders)]
        all_points = np.meshgrid(*single_axes)
        all_points = np.concatenate([points.flatten()[:, None] for points in all_points], axis=-1)
        all_points = all_points + self.low_lim
        if self.spatial_dim == 3:
            all_points = np.concatenate((all_points, np.full((all_points.shape[0],1), self.center[-1])), axis=-1)
        return all_points

    def sample_points(self, num_points):
        """Returns a set of points sampled uniformly within the region
        
        Parameters
        ----------
        num_points : int
            Number of points to sample.

        Returns
        -------
        points : np.ndarray of shape (num_points, 3) or (num_points, 2)
            Each row is a coordinate in 3D or 2D space.
        """
        x = self.rng.uniform(self.low_lim[0], self.high_lim[0], num_points)
        y = self.rng.uniform(self.low_lim[1], self.high_lim[1], num_points)
        points = np.stack((x,y), axis=1)

        if self.spatial_dim == 3:
            points = np.concatenate((points, self.center[2]*np.ones((points.shape[0],1))), axis=-1)
        return points

    def plot(self, ax, label=None):
        rect = patches.Rectangle(self.low_lim, self.side_lengths[0], self.side_lengths[1], fill=True, alpha=0.3, label=label)
        ax.add_patch(rect)



class Disc(Region):
    def __init__(self, radius, center, point_spacing=(1,1), spatial_dim=3, rng=None):
        super().__init__(rng)
        assert spatial_dim in (2,3)
        assert len(center) == spatial_dim
        self.radius = radius
        self.center = np.array(center)
        self.point_spacing = np.array(point_spacing)
        self.spatial_dim = spatial_dim
        self.volume = self.radius**2 * np.pi

    def is_in_region(self, coordinates):
        if self.spatial_dim == 3:
            if not np.allclose(coordinates[:,2], self.center[2]):
                return False

        centered_coords = coordinates[:,:2] - self.center[None,:2]
        norm_coords = np.sqrt(np.sum(np.square(centered_coords), axis=-1))
        is_in = norm_coords <= self.radius
        return is_in

    def equally_spaced_points(self):
        point_dist = self.point_spacing
        block_dims = np.array([self.radius*2, self.radius*2])
        num_points = np.ceil(block_dims / point_dist)
        single_axes = [np.arange(num_p) * p_dist for num_p, p_dist in zip(num_points, point_dist)]

        meshed_points = np.meshgrid(*single_axes)
        meshed_points = [mp.reshape(-1)[:,None] for mp in meshed_points]
        all_points = np.concatenate(meshed_points, axis=-1)#.reshape(-1,2)

        shift = (num_points-1)*point_dist / 2
        all_points -= shift[None,:]

        inside_disc = np.sqrt(np.sum(all_points**2,axis=-1)) <= self.radius
        all_points = all_points[inside_disc,:]

        if self.spatial_dim == 3:
            all_points = np.concatenate((all_points, np.zeros((all_points.shape[0],1))), axis=-1)

        all_points += self.center[None,:]
        return all_points

    def sample_points(self, num_points):
        r = self.radius * np.sqrt(self.rng.uniform(0,1,num_points))
        angle = 2 * np.pi * self.rng.uniform(0,1,num_points)
        x = r * np.cos(angle) + self.center[0]
        y = r * np.sin(angle) + self.center[1]
        points = np.stack((x,y), axis=1)

        if self.spatial_dim == 3:
            points = np.concatenate((points, self.center[2]*np.ones((points.shape[0],1))), axis=-1)
        return points

    def plot(self, ax, label=None):
        circ = patches.Circle(self.center[:2], self.radius, fill=True, alpha=0.3, label=label)
        ax.add_patch(circ)



class Ball(Region):
    def __init__(self, radius, center, point_spacing=(1,1,1), rng=None):
        """Constructs a ball region
        
        Parameters
        ----------
        radius : float
            Radius of the ball.
        center : array_like of shape (3,), optional
            Center of the cylinder. The default is (0,0,0).
        point_spacing : array_like of shape (3,), optional
            Spacing between points in each direction, affects the selection of points
            for the equally_spaced_points method. The default is (1,1,1).
        rng : np.random.Generator, optional
            Random number generator. The default is None, in which case a new generator
            with a random seed will be created. 
            The generator affects the sampling of points in the sample_points method, and so 
            should be supplied for a reproducible result.

        Returns
        -------
        ball : Ball
        """
        super().__init__(rng)
        self.radius = radius
        self.center = np.array(center)
        assert self.center.ndim == 1
        self.point_spacing = np.array(point_spacing)

        self.volume = (4/3) * self.radius**3 * np.pi

    def is_in_region(self, coordinates):
        centered_coords = coordinates - self.center[None,:]
        is_in = np.linalg.norm(centered_coords, axis=-1) <= self.radius
        #is_in = norm_coords <= self.radius
        return is_in

    def equally_spaced_points(self):
        cuboid = Cuboid((2*self.radius, 2*self.radius, 2*self.radius), point_spacing=self.point_spacing)
        grid_points = cuboid.equally_spaced_points()

        grid_points += self.center[None,:]
        #inside_ball = np.linalg.norm(grid_points, axis=-1) <= self.radius
        grid_points = grid_points[self.is_in_region(grid_points),:]
        #grid_points += self.center[None,:]
        return grid_points

    def sample_points(self, num_points):
        finished = False
        num_accepted = 0

        samples = np.zeros((num_points, 3))
        while not finished:
            uniform_samples = self.rng.uniform(low=-self.radius, high=self.radius, size=(num_points, 3))

            filtered_samples = uniform_samples[np.linalg.norm(uniform_samples, axis=-1) <= self.radius,:]
            num_new = filtered_samples.shape[0]
            num_to_accept = min(num_new, num_points - num_accepted)

            samples[num_accepted:num_accepted+num_to_accept,:] = filtered_samples[:num_to_accept,:] 
            num_accepted += num_to_accept

            if num_accepted >= num_points:
                finished = True

        samples += self.center
        return samples

    def plot(self, ax, label=None):
        circ = patches.Circle(self.center[:2], self.radius, fill=True, alpha=0.3, label=label)
        ax.add_patch(circ)



class Cylinder(Region):
    def __init__(self, radius, height, center=(0,0,0), point_spacing=(1,1,1), rng=None):
        """Constructs a cylinder region
        
        Parameters
        ----------
        radius : float
            Radius of the cylinder.
        height : float
            Height of the cylinder.
        center : array_like of shape (3,), optional
            Center of the cylinder. The default is (0,0,0).
        point_spacing : array_like of shape (3,), optional
            Spacing between points in each direction, affects the selection of points
            for the equally_spaced_points method. The default is (1,1,1).
        rng : np.random.Generator, optional
            Random number generator. The default is None, in which case a new generator
            with a random seed will be created. 
            The generator affects the sampling of points in the sample_points method, and so 
            should be supplied for a reproducible result.

        Returns
        -------
        cylinder : Cylinder
        """
        super().__init__(rng)
        self.radius = radius
        self.height = height
        self.center = np.array(center)
        self.point_spacing = np.array(point_spacing)
        self.volume = self.radius**2 * np.pi * self.height

    def is_in_region(self, coordinates):
        """Checks whether the coordinate is within the cylinder or not. 
        
        Parameters
        ----------
        coordinates : np.ndarray
            Shape (N,3) where N is the number of coordinates to check. 
            Each row is a coordinate in 3D space.
            If only one coordinate is to be checked, it can be supplied as
            a (3,) array        

        Returns
        -------
        is_in : boolan np.ndarray of shape (N,)
            True if the coordinate is within the cylinder, False otherwise.
        """
        if coordinates.ndim == 1:
            coordinates = coordinates[None,:]
        assert coordinates.ndim == 2
        assert coordinates.shape[1] == 3

        centered_coords = coordinates - self.center[None,:]
        norm_coords = np.sqrt(np.sum(np.square(centered_coords[:,:2]), axis=-1))
        is_in_disc = norm_coords <= self.radius
        is_in_height = np.logical_and(-self.height/2 <= centered_coords[:,2],
                                      self.height/2 >= centered_coords[:,2])
        is_in = np.logical_and(is_in_height, is_in_disc)
        return is_in

    def equally_spaced_points(self):
        """Returns a grid of points within the cylinder
        
        Returns
        -------
        all_points : np.ndarray of shape (num_points, 3)
            The number of points returned is determined by the radius, height and point_spacing.
            Each row is a coordinate in 3D space.
        """
        point_dist = self.point_spacing
        block_dims = np.array([self.radius*2, self.radius*2, self.height])
        num_points = np.ceil(block_dims / point_dist)
        single_axes = [np.arange(num_p) * p_dist for num_p, p_dist in zip(num_points, point_dist)]
        
        all_points = np.stack(np.meshgrid(*single_axes, indexing="ij"),axis=-1)
        all_points = all_points.reshape(-1,3)

        shift = (num_points-1)*point_dist / 2
        all_points -= shift[None,:]

        inside_cylinder = np.sqrt(np.sum(all_points[:,:2]**2,axis=-1)) <= self.radius
        all_points = all_points[inside_cylinder,:]

        all_points += self.center[None,:]
        return all_points

    def sample_points(self, num_points):
        """Returns a set of points sampled uniformly within the cylinder
        
        Parameters
        ----------
        num_points : int
            Number of points to sample.

        Returns
        -------
        points : np.ndarray of shape (num_points, 3)
            Each row is a coordinate in 3D space.
        """
        r = self.radius * np.sqrt(self.rng.uniform(0,1,num_points))
        angle = 2 * np.pi * self.rng.uniform(0,1,num_points)
        x = r * np.cos(angle) + self.center[0]
        y = r * np.sin(angle) + self.center[1]
        h = self.rng.uniform(-self.height/2,self.height/2, num_points) + self.center[2]
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
