import numpy as np


class Trajectory:
    def __init__(self, pos_func):
        """pos_func is a function, which takes a time_index in samples and outputs a position"""
        self.pos_func = pos_func
        #self.pos = np.full((1,3), np.nan)

    def current_pos(self, time_idx):
        return self.pos_func(time_idx)

    def plot(self, ax, symbol, name):
        pass


class LinearTrajectory(Trajectory):
    def __init__(self, points, period, samplerate):
        """
        points is array of shape (numpoints, spatial_dim) or equivalent list of lists
        period is in seconds
        update freq is in samples
        """
        if isinstance(points, (list, tuple)):
            points = np.array(points)

        if not np.allclose(points[-1,:], points[0,:]):
            points = np.concatenate((points, points[:1,:]), axis=0)
        self.anchor_points = points

        segment_distance = np.sqrt(np.sum((points[:-1,:] - points[1:,:])**2, axis=-1))
        assert all(segment_distance > 0)
        
        tot_distance = np.sum(segment_distance)
        distance_per_sample = tot_distance / (samplerate * period)
        segment_samples = segment_distance / distance_per_sample

        #cycle_start = 0
        cumulative_samples = np.concatenate(([0], np.cumsum(segment_samples)))

        def pos_func(t):
            period_samples = t % (period * samplerate)

            point_idx = np.argmax(period_samples < cumulative_samples)
            time_this_segment = period_samples - cumulative_samples[point_idx-1]
            ip_factor = time_this_segment / segment_samples[point_idx-1]
            pos = points[point_idx-1,:]*(1-ip_factor) + points[point_idx,:]*ip_factor
            return pos[None,:]

        super().__init__(pos_func)

    def plot(self, ax, symbol, label):
        ax.plot(self.anchor_points[:,0], self.anchor_points[:,1], marker=symbol, linestyle="dashed", label=label, alpha=0.8)

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
