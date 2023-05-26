import numpy as np

import aspsim.room.generatepoints as gp

class Trajectory:
    def __init__(self, pos_func):
        """pos_func is a function, which takes a time_index in samples and outputs a position"""
        self.pos_func = pos_func
        #self.pos = np.full((1,3), np.nan)

    def current_pos(self, time_idx):
        return self.pos_func(time_idx)

    def plot(self, ax, symbol, name):
        pass


class TrajectoryCollection(Trajectory):
    def __init__(self, trajectories):
        self.trajectories = trajectories
        #self.num_pos = len(self.trajectories)

    def current_pos(self, time_idx):
        return np.concatenate([traj.current_pos(time_idx) for traj in self.trajectories], axis=0)
    
    def plot(self, ax, symbol, name, tot_samples):
        for traj in self.trajectories:
            traj.plot(ax, symbol, name, tot_samples)

class LinearTrajectory(Trajectory):
    def __init__(self, points, period, samplerate, mode="constant_speed"):
        """
        points is array of shape (numpoints, spatial_dim) or equivalent list of lists
        period is in seconds
        update freq is in samples
        mode : {'constant_speed' or 'constant_time'}
            if 'constant_speed', the speed of the movement will be constant, and
            calibrated such that it returns to the starting position after one period.
            if 'constant_time', each segment will take equal time, and the speed will
            therefore go up for long segments and down for short segments. 
        """
        if isinstance(points, (list, tuple)):
            points = np.array(points)
        #self.num_pos = 1

        if not np.allclose(points[-1,:], points[0,:]):
            points = np.concatenate((points, points[:1,:]), axis=0)
        self.anchor_points = points
        self.period = period
        self.samplerate = samplerate

        if mode == "constant_speed":
            pos_func = self._constant_speed_pos_func(points, period, samplerate)
        elif mode == "constant_time":
            pos_func = self._constant_time_pos_func(points, period, samplerate)
        else:
            raise ValueError ("Invalid mode argument")

        super().__init__(pos_func)

    def _constant_speed_pos_func(self, points, period, samplerate):
        segment_distance = np.sqrt(np.sum((points[:-1,:] - points[1:,:])**2, axis=-1))
        assert all(segment_distance > 0)
        
        tot_distance = np.sum(segment_distance)
        distance_per_sample = tot_distance / (samplerate * period)
        self.samples_per_segment = segment_distance / distance_per_sample

        #cycle_start = 0
        self.cumulative_samples = np.concatenate(([0], np.cumsum(self.samples_per_segment)))

        def pos_func(n):
            period_samples = n % (period * samplerate)

            point_idx = np.argmax(period_samples < self.cumulative_samples)
            time_this_segment = period_samples - self.cumulative_samples[point_idx-1]
            ip_factor = time_this_segment / self.samples_per_segment[point_idx-1]
            pos = points[point_idx-1,:]*(1-ip_factor) + points[point_idx,:]*ip_factor
            return pos[None,:]
        return pos_func
    
    def _constant_time_pos_func(self, points, period, samplerate):
        num_segments = points.shape[0]
        time_per_segment = period / num_segments
        self.samples_per_segment = time_per_segment * samplerate
        segment_distance = np.sqrt(np.sum((points[:-1,:] - points[1:,:])**2, axis=-1))
        
        distance_per_time = segment_distance / time_per_segment
        distance_per_sample = distance_per_time / samplerate

        self.cumulative_samples = np.arange(num_segments) * self.samples_per_segment #np.concatenate(([0], np.cumsum(samples_per_segment)))

        def pos_func(n):
            period_samples = n % (period * samplerate)

            point_idx = np.argmax(period_samples < self.cumulative_samples)
            time_this_segment = period_samples - self.cumulative_samples[point_idx-1] # time in number of samples
            ip_factor = time_this_segment / self.samples_per_segment
            pos = points[point_idx-1,:]*(1-ip_factor) + points[point_idx,:]*ip_factor
            return pos[None,:]
        return pos_func

    def plot(self, ax, symbol, label, tot_samples=None):
        if tot_samples is None:
            points = self.anchor_points
        else:
            if tot_samples >= self.period*self.samplerate:
                points = self.anchor_points
            else:
                point_idx = np.argmax(tot_samples < self.cumulative_samples)
                points = self.anchor_points[:point_idx,:]
        ax.plot(points[:,0], points[:,1], marker=symbol, linestyle="dashed", label=label, alpha=0.8)

    # @classmethod
    # def linear_interpolation_const_speed(cls, points, period, samplerate):
    #     """
    #     points is array of shape (numpoints, spatial_dim) or equivalent list of lists
    #     period is in seconds
    #     update freq is in samples
    #     """
    #     if isinstance(points, (list, tuple)):
    #         points = np.array(points)

    #     if not np.allclose(points[-1,:], points[0,:]):
    #         points = np.concatenate((points, points[:1,:]), axis=0)

    #     segment_distance = np.sqrt(np.sum((points[:-1,:] - points[1:,:])**2, axis=-1))
    #     assert all(segment_distance > 0)
        
    #     tot_distance = np.sum(segment_distance)
    #     #updates_per_period = samplerate * period / update_freq
    #     distance_per_sample = tot_distance / (samplerate * period)
    #     segment_samples = segment_distance / distance_per_sample

    #     cycle_start = 0
    #     cumulative_samples = np.concatenate(([0], np.cumsum(segment_samples)))

    #     def pos_func(t):
    #         period_samples = t % (period * samplerate)

    #         point_idx = np.argmax(period_samples < cumulative_samples)
    #         time_this_segment = period_samples - cumulative_samples[point_idx-1]
    #         ip_factor = time_this_segment / segment_samples[point_idx-1]
    #         pos = points[point_idx-1,:]*(1-ip_factor) + points[point_idx,:]*ip_factor
    #         return pos[None,:]

    #     return cls(pos_func)




class CircularTrajectory(Trajectory):
    def __init__(self, 
            radius : tuple[float, float], 
            center : tuple[float, float, float], 
            radial_period : float, 
            angle_period : float, 
            samplerate : int):
        """
        Moves around a circle in one angle_period, while it moves from the outer radius
        to the inner radius and back again in one radial_period

        The circle is defined by its center and radius

        Parameters
        ----------
        radius : length-2 tuple of floats
            inner and outer radius. Interpreted by ancsim as meters
        center : length-3 tuple of floats
            center of the disc/circle
        radial_period : float
            time in seconds for the trajectory to go from outer radius to 
            inner radius and back again
        angle_period : float
            time in seconds for trajectory to go one lap around the circle
        samplerate : int
            samplerate of the simulation, supplied for the units of the periods 
            to make sense
        
        """
        self.radius = radius
        self.center = center
        self.radial_period = radial_period
        self.angle_period = angle_period
        self.samplerate = samplerate
        #self.num_pos = 1

        self.radius_diff = self.radius[1] - self.radius[0]
        assert self.radius_diff >= 0

        def pos_func(t):
            angle_period_samples = t % (angle_period * samplerate)
            radial_period_samples = t % (radial_period * samplerate)

            angle_portion = angle_period_samples / (angle_period * samplerate)
            radial_portion = radial_period_samples / (radial_period * samplerate)

            angle = 2 * np.pi * angle_portion

            if radial_portion < 0.5:
                rad = radius[0] + self.radius_diff * (1 - 2 * radial_portion)
            else:
                rad = radius[0] + self.radius_diff * (radial_portion-0.5)*2

            (x,y) = gp.pol2cart(rad, angle)
            return np.array([[x,y,0]]) + center

        super().__init__(pos_func)

    def plot(self, ax, symbol="o", label="", tot_samples=None):
        if tot_samples is not None:
            raise NotImplementedError
        approx_num_points = 1000
        max_samples = self.samplerate*max(self.radial_period, self.angle_period)
        samples_per_point = max_samples // approx_num_points
        t = np.arange(0, max_samples, samples_per_point)
        num_points = t.shape[0]

        pos = np.zeros((num_points, 3))
        for i in range(num_points):
            pos[i,:] = np.squeeze(self.pos_func(t[i]))
        ax.plot(pos[:,0], pos[:,1], marker=symbol, linestyle="dashed", label=label, alpha=0.8)

        # ax.plot(self.anchor_points[:,0], self.anchor_points[:,1], marker=symbol, linestyle="dashed", label=label, alpha=0.8)
