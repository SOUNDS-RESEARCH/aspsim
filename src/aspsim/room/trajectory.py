"""Trajectory definitions for moving sources and microphones."""

import numpy as np

import aspsim.room.generatepoints as gp


class Trajectory:
    """Base class defining a trajectory through the room.

    In order to simulate a moving source or microphone, the trajectory must be
    defined in accordance with this API. Easiest way is to create a function defining
    the position at each time index, and pass it to the constructor. Alternatively,
    subclass this class and implement the current_pos method.
    """

    def __init__(self, pos_func):
        """Define a position function from sample index to position."""
        self.pos_func = pos_func
        # self.pos = np.full((1,3), np.nan)

    def current_pos(self, time_idx):
        """Return the position at a given time index.

        Parameters
        ----------
        time_idx : int
            time index in samples

        Returns
        -------
        pos : array of shape (1,3)
            position at time time_idx
        """
        return self.pos_func(time_idx)

    def plot(self, ax, symbol, name, tot_samples):
        """Plot the trajectory if implemented."""
        pass


class TrajectoryCollection(Trajectory):
    """A class for combining multiple trajectories into one, in the case where you want to have multiple moving objects in the same array."""

    def __init__(self, trajectories):
        """Create a trajectory collection.

        Parameters
        ----------
        trajectories : list of Trajectory objects
        """
        self.trajectories = trajectories
        # self.num_pos = len(self.trajectories)

    def current_pos(self, time_idx):
        """Return the stacked positions at a given time index."""
        return np.concatenate(
            [traj.current_pos(time_idx) for traj in self.trajectories], axis=0
        )

    def plot(self, ax, symbol, name, tot_samples):
        """Plot all trajectories in the collection."""
        for traj in self.trajectories:
            traj.plot(ax, symbol, name, tot_samples)


class LinearTrajectory(Trajectory):
    """Linear trajectory through anchor points."""

    def __init__(self, points, period, samplerate, mode="constant_speed"):
        """Move through a series of points in straight lines.

        Parameters
        ----------
        points : ndarray of shape (numpoints, spatial_dim) or equivalent list of lists
            The points that the trajectory will move through. The trajectory will
            start and end at the first point.
        period : float
            The time in seconds for the trajectory to go through all the points and
            return to the starting point.
        samplerate : int
            The samplerate of the simulation.
        mode : 'constant_speed' or 'constant_time'
            if 'constant_speed', the speed of the movement will be constant, and
            calibrated such that it returns to the starting position after one period.
            if 'constant_time', each segment will take equal time, and the speed will
            therefore go up for long segments and down for short segments.
        """
        if isinstance(points, (list, tuple)):
            points = np.array(points)
        # self.num_pos = 1

        if not np.allclose(points[-1, :], points[0, :]):
            points = np.concatenate((points, points[:1, :]), axis=0)
        self.anchor_points = points
        self.period = period
        self.samplerate = samplerate

        if mode == "constant_speed":
            pos_func = self._constant_speed_pos_func(points, period, samplerate)
        elif mode == "constant_time":
            pos_func = self._constant_time_pos_func(points, period, samplerate)
        else:
            raise ValueError("Invalid mode argument")

        super().__init__(pos_func)

    def _constant_speed_pos_func(self, points, period, samplerate):
        segment_distance = np.sqrt(
            np.sum((points[:-1, :] - points[1:, :]) ** 2, axis=-1)
        )
        assert all(segment_distance > 0)

        tot_distance = np.sum(segment_distance)
        distance_per_sample = tot_distance / (samplerate * period)
        self.samples_per_segment = segment_distance / distance_per_sample

        # cycle_start = 0
        self.cumulative_samples = np.concatenate(
            ([0], np.cumsum(self.samples_per_segment))
        )

        def pos_func(n):
            period_samples = n % (period * samplerate)

            point_idx = np.argmax(period_samples < self.cumulative_samples)
            time_this_segment = period_samples - self.cumulative_samples[point_idx - 1]
            ip_factor = time_this_segment / self.samples_per_segment[point_idx - 1]
            pos = (
                points[point_idx - 1, :] * (1 - ip_factor)
                + points[point_idx, :] * ip_factor
            )
            return pos[None, :]

        return pos_func

    def _constant_time_pos_func(self, points, period, samplerate):
        num_segments = points.shape[0]
        time_per_segment = period / num_segments
        self.samples_per_segment = time_per_segment * samplerate
        segment_distance = np.sqrt(
            np.sum((points[:-1, :] - points[1:, :]) ** 2, axis=-1)
        )

        distance_per_time = segment_distance / time_per_segment
        distance_per_sample = distance_per_time / samplerate

        self.cumulative_samples = (
            np.arange(num_segments) * self.samples_per_segment
        )  # np.concatenate(([0], np.cumsum(samples_per_segment)))

        def pos_func(n):
            period_samples = n % (period * samplerate)

            point_idx = np.argmax(period_samples < self.cumulative_samples)
            time_this_segment = (
                period_samples - self.cumulative_samples[point_idx - 1]
            )  # time in number of samples
            ip_factor = time_this_segment / self.samples_per_segment
            pos = (
                points[point_idx - 1, :] * (1 - ip_factor)
                + points[point_idx, :] * ip_factor
            )
            return pos[None, :]

        return pos_func

    def plot(self, ax, symbol, name, tot_samples=None):
        """Plot the trajectory path."""
        if tot_samples is None:
            points = self.anchor_points
        else:
            if tot_samples >= self.period * self.samplerate:
                points = self.anchor_points
            else:
                point_idx = np.argmax(tot_samples < self.cumulative_samples)
                points = self.anchor_points[:point_idx, :]
        ax.plot(
            points[:, 0],
            points[:, 1],
            marker=symbol,
            linestyle="dashed",
            label=name,
            alpha=0.8,
        )

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
    """Circular trajectory with radial modulation."""

    def __init__(
        self,
        radius: tuple[float, float],
        center: tuple[float, float, float],
        radial_period: float,
        angle_period: float,
        samplerate: int,
        start_angle: float = 0,
    ):
        """Move around a circle.

        Moves around a circle in one angle_period, while it moves from the outer radius
        to the inner radius and back again in one radial_period

        The circle is defined by its center and radius

        Parameters
        ----------
        radius : length-2 tuple of floats
            inner and outer radius. Interpreted by aspsim as meters
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
        self.start_angle = start_angle
        # self.num_pos = 1

        self.radius_diff = self.radius[1] - self.radius[0]
        assert self.radius_diff >= 0

        def pos_func(t):
            angle_period_samples = t % (angle_period * samplerate)
            radial_period_samples = t % (radial_period * samplerate)

            angle_portion = angle_period_samples / (angle_period * samplerate)
            radial_portion = radial_period_samples / (radial_period * samplerate)

            angle = self.start_angle + 2 * np.pi * angle_portion
            # angle = angle % (2*np.pi)

            if radial_portion < 0.5:
                rad = radius[0] + self.radius_diff * (1 - 2 * radial_portion)
            else:
                rad = radius[0] + self.radius_diff * (radial_portion - 0.5) * 2

            (x, y) = gp.pol2cart(rad, angle)
            return np.array([[x, y, 0]]) + center

        super().__init__(pos_func)

    def plot(self, ax, symbol="o", name="", tot_samples=None):
        """Plot the circular trajectory path."""
        # if tot_samples is not None:
        #    max_samples =
        #    raise NotImplementedError

        approx_num_points = 1000
        max_samples = self.samplerate * max(self.radial_period, self.angle_period)
        samples_per_point = max_samples // approx_num_points
        t = np.arange(0, max_samples, samples_per_point)
        num_points = t.shape[0]

        pos = np.zeros((num_points, 3))
        for i in range(num_points):
            pos[i, :] = np.squeeze(self.pos_func(t[i]))
        ax.plot(
            pos[:, 0],
            pos[:, 1],
            marker=symbol,
            linestyle="dashed",
            label=name,
            alpha=0.8,
        )


class LissajousTrajectoryConstantSpeed(Trajectory):
    """Lissajous trajectory with constant speed."""

    def __init__(self, amplitude, freq, center, samplerate, target_speed, num_samples):
        """Define a position function from sample index to position.

        Implements a velocity-normalized version of a Lissajous curve, which is defined as
        pos(t) = amplitude * cos(2 * pi * freq * t + phi) + center
        where cos is interpreted elementwise.

        Parameters
        ----------
        amplitude : array of shape (1,3)
            amplitude in meters of the oscillation in each dimension
        freq : array of shape (1,3)
            frequency in Hz of the oscillation in each dimension
        center : array of shape (1,3)
            center of the trajectory
        samplerate : int
            samplerate of the simulation, supplied for the units of the periods
            to make sense
        target_speed : float
            target speed of the movement in meters per second
        num_samples : int
            number of samples to generate for the trajectory. Does not need to be exact, but should be larger or
            equal to the number of samples in the simulation, as the trajectory is pre-generated.

        """
        self.amplitude = amplitude
        self.freq = freq
        assert self.freq.shape == (1, 3)
        # self.period_len = period_len
        self.center = center
        assert self.center.shape == (1, 3)
        self.phase_offset = np.array([[0, np.pi / 2, np.pi / 2]])

        self.samplerate = samplerate
        self.speed_factor = target_speed
        self.num_samples = num_samples
        # self.pos = np.full((1,3), np.nan)

        self.all_pos = _generate_constant_speed_trajectory(
            self.r, self.velocity, self.samplerate, target_speed, num_samples
        )

    def r(self, t):
        """Return the position at time index t."""
        return self.center + self.amplitude * np.cos(
            2 * np.pi * t * self.freq / self.samplerate + self.phase_offset
        )

    def velocity(self, t):
        """Return the velocity at time index t."""
        return (
            -self.amplitude
            * 2
            * np.pi
            * self.freq
            * np.sin(2 * np.pi * t * self.freq / self.samplerate + self.phase_offset)
            / self.samplerate
        )

    def current_pos(self, time_idx):
        """Return the current position for the given time index."""
        if time_idx >= self.num_samples:
            raise ValueError(
                f"time_idx {time_idx} is out of bounds for pre-generated trajectory with num_samples {self.num_samples}"
            )
        return self.all_pos[time_idx : time_idx + 1, :]

    def plot(self, ax, symbol, name, tot_samples):
        """Plot the trajectory if needed."""
        pass


def _generate_constant_speed_trajectory(
    position, velocity, samplerate, target_speed, num_samples, tolerance=0.05
):
    """Generate a constant-speed trajectory.

    Parameters
    ----------
    target_speed : int
        Target speed in meters per second.
    """
    all_pos = np.zeros((num_samples, 3))
    all_pos[0, :] = position(0)

    target_speed_per_sample = target_speed / samplerate

    t = 0
    for n in range(1, num_samples):
        last_pos = all_pos[n - 1, :]
        v = np.linalg.norm(velocity(t))
        timestep = target_speed_per_sample / v

        candidate_pos = position(t + timestep)
        speed = np.linalg.norm(candidate_pos - last_pos)

        if (
            np.abs(speed - target_speed_per_sample) / target_speed_per_sample
            > tolerance
        ):
            if speed > target_speed_per_sample:
                t_low = 0
                t_high = timestep
            else:
                t_low = timestep
                t_high = 2 * timestep
                while (
                    np.linalg.norm(position(t + t_high) - last_pos)
                    <= target_speed_per_sample
                ):
                    t_high = 2 * t_high
            timestep = _pos_bifurcation(
                t_low, t_high, t, position, last_pos, target_speed_per_sample, tolerance
            )
            candidate_pos = position(t + timestep)

        t += timestep
        all_pos[n, :] = candidate_pos

    return all_pos


def _pos_bifurcation(low, high, offset, pos, previous_pos, desired_val, tolerance):
    """Find a time step that achieves a desired displacement."""
    speed = -1000

    while np.abs(speed - desired_val) / desired_val > tolerance:
        t = (low + high) / 2
        p = pos(offset + t)

        speed = np.linalg.norm(p - previous_pos)

        if speed <= desired_val:
            low = t
        else:
            high = t
    return t
