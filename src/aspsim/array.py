"""In this module are Arrays and collections of arrays, which are used to represent sources and microphones."""

import copy
import json
import pathlib
from abc import ABC

import aspcore.utilities as utils
import dill
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

import aspsim.configutil as cfutil
import aspsim.room.region as reg
import aspsim.room.roomimpulseresponse as rir
import aspsim.room.trajectory as tr


class ArrayCollection:
    """A class for managing arrays of sources and microphones, as well as the paths between them.

    Attributes
    ----------
    arrays : dict of Array objects
        The arrays that have been added to the collection
    names_mic : list of str
        The names of all microphone arrays
    names_src : list of str
        The names of all source arrays
    names_free_src : list of str
        The names of the free source arrays
    paths : two-layer dict conatining ndarrays
        each path can be accessed as paths[src_name][mic_name], which
        is an ndarray of shape (num_src, num_mic, num_samples), representing
        the impulse response between a source and a microphone array
    path_type : two-layer dict containing str
        each path type can be accessed as path_type[src_name][mic_name], which
        is a string representing the type of propagation between a source and a microphone array
        See create_path for more info on possible values
    path_info : dict of dicts
        Contains metadata about the paths, such as truncation error and reverberation time.
        Can be accessed as path_info[f'{src_name}->{mic_name}']
    """

    def __init__(self):
        """Initialize an empty ArrayCollection."""
        self.arrays = {}
        self.names_mic = []
        self.names_src = []
        self.names_free_src = []
        self.paths = {}
        self.path_type = {}

        self.path_info = {}

        # rir_all[src_name][mic_name] -> ndarray of shape
        # (num_updates, num_src, num_mic, ir_len). Populated in setup_ir for
        # every (src, mic) combo where at least one side is dynamic.
        self.rir_all = {}

    def __getitem__(self, key):
        """Get named array from the collection."""
        return self.arrays[key]

    def __iter__(self):
        """Iterate over all arrays in the collection."""
        for ar in self.arrays.values():
            yield ar

    def __contains__(self, arrayName):
        """Check if an array with the given name is in the collection."""
        return arrayName in self.arrays

    def save_metadata(self, folder_path):
        """Save all metadata to the given folder.

        Parameters
        ----------
        filepath : pathlib.Path
        """
        if folder_path is not None:
            self._save_metadata_arrays(folder_path)
            self._save_metadata_paths(folder_path)
            self._save_readable_pos(folder_path)

    def _save_metadata_arrays(self, filepath):
        array_info = {}
        for ar_name, ar in self.arrays.items():
            array_info[ar_name] = ar.metadata
        with open(filepath.joinpath("metadata_arrays.json"), "w") as f:
            json.dump(array_info, f, indent=4)

    def _save_metadata_paths(self, filepath):
        with open(filepath.joinpath("metadata_paths.json"), "w") as f:
            json.dump(self.path_info, f, indent=4)

    def _save_readable_pos(self, filepath):
        pos = {}
        for ar_name, ar in self.arrays.items():
            pos[ar_name] = ar.pos.tolist()
        with open(filepath.joinpath("array_pos.json"), "w") as f:
            json.dump(pos, f, indent=4)

    def set_default_path_type(self, path_type):
        """Set the path type for all paths that were not modified by the user.

        Parameters
        ----------
        path_type : str
            The path type to be set. See create_path for more info on possible values
        """
        for src, mic in self.mic_src_combos():
            if (
                mic.name not in self.path_type[src.name]
                and mic.name not in self.paths[src.name]
            ):
                self.path_type[src.name][mic.name] = path_type
            elif mic.name in self.paths[src.name]:
                self.path_type[src.name][mic.name] = "modified"

    def empty(self):
        """Return True if the collection has no arrays."""
        return len(self.arrays) == 0

    def sources(self):
        """Iterate over all source arrays."""
        for name in self.names_src:
            yield self.arrays[name]

    def mics(self):
        """Iterate over all mic arrays."""
        for name in self.names_mic:
            yield self.arrays[name]

    def free_sources(self):
        """Iterate over all free source arrays."""
        for name in self.names_free_src:
            yield self.arrays[name]

    def mic_src_combos(self):
        """Iterate over the all combinations of mics and sources."""
        for src_name in self.names_src:
            for mic_name in self.names_mic:
                yield self.arrays[src_name], self.arrays[mic_name]

    def iter_paths(self):
        """Iterate over all paths."""
        for src_name in self.names_src:
            for mic_name in self.names_mic:
                yield (
                    self.arrays[src_name],
                    self.arrays[mic_name],
                    self.paths[src_name][mic_name],
                )

    def add_array(self, array):
        """Add an array to the collection.

        Parameters
        ----------
        array : Array
            The array to be added. Can be any subclass of Array
        """
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
        """Set the propagation paths between some sources and microphones.

        Parameters
        ----------
        paths : dict of dicts of ndarrays
            paths[src_name][mic_name] is an ndarray of shape (num_src, num_mic, num_samples)
            representing the impulse response between a source and a microphone array
            the source and and mic array must already be in the collection
        """
        for src_name, src_paths in paths.items():
            for mic_name, path in src_paths.items():
                self.set_prop_path(path, src_name, mic_name)

    def set_prop_path(self, path, src_name, mic_name):
        """Set the propagation path between a source and a microphone array.

        Parameters
        ----------
        path : ndarray of shape (num_src, num_mic, num_samples)
            representing the impulse response between a source and a microphone array
            The ndarray must be of the correct shape
        src_name : str
            name of the source array.
            The array must already be in the collection
        mic_name : str
            name of the microphone array.
            The array must already be in the collection
        """
        assert src_name in self.names_src
        assert mic_name in self.names_mic
        assert isinstance(path, np.ndarray)
        assert path.shape[0] == self.arrays[src_name].num
        assert path.shape[1] == self.arrays[mic_name].num
        assert path.ndim == 3
        self.paths[src_name][mic_name] = path

    def set_path_types(self, path_types):
        """Set the propagation type between some sources and microphones.

        Parameters
        ----------
        path_types : dict of dicts of str
            path_types[src_name][mic_name] is a string representing the type of propagation
            between a source and a microphone array
            The source and and mic array must already be in the collection
            See create_path for more info on possible values
        """
        for src_name, src_path_types in path_types.items():
            for mic_name, pt in src_path_types.items():
                assert src_name in self.names_src
                assert mic_name in self.names_mic
                assert isinstance(pt, str)
                self.path_type[src_name][mic_name] = pt

    def setup_ir(self, sim_info: cfutil.SimulatorInfo):
        """Generate the impulse responses between all sources and microphones.

        The method uses the path_type attribute to determine the type of propagation

        Parameters
        ----------
        sim_info : SimulatorInfo
            The simulation info object
        """
        self.sim_info = sim_info

        for array_i in self.arrays.values():
            array_i.prepare(sim_info)

        self.path_generator = rir.PathGenerator(self.sim_info, self)

        for src, mic in self.mic_src_combos():
            self.path_info[f"{src.name}->{mic.name}"] = {}

            if src.num == 0 or mic.num == 0:
                reverb = "none"
            else:
                reverb = self.path_type[src.name][mic.name]

            self.path_info[f"{src.name}->{mic.name}"]["type"] = reverb
            print(f"{src.name}->{mic.name} has propagation type: {reverb}")
            if reverb != "modified":
                self.paths[src.name][mic.name], path_info = (
                    self.path_generator.create_path(
                        src, mic, reverb, sim_info, True, True
                    )
                )
                for key, val in path_info.items():
                    self.path_info[f"{src.name}->{mic.name}"][key] = val

                if src.dynamic or mic.dynamic:
                    self._precompute_dynamic_rirs(src, mic, reverb, sim_info)

    def _precompute_dynamic_rirs(self, src, mic, reverb, sim_info):
        """Precompute RIRs for every update step of a dynamic (src, mic) combo.

        The result is stored in ``self.rir_all[src.name][mic.name]`` as an
        ndarray of shape ``(num_updates, num_src, num_mic, ir_len)``. The
        warmup rows (``time_all[i] < 0``) all hold the ``t=0`` RIR.

        ``self.paths[src.name][mic.name]`` (the static RIR at ``t=0`` generated
        earlier in ``setup_ir``) is reused as the warmup RIR and as the
        ``t=0`` row, then the array's positions are walked through the
        post-warmup rows and one RIR per update step is generated.

        Memory note: storage is num_updates * num_src * num_mic * ir_len * 8
        bytes per dynamic path; for long simulations or long RIRs this can
        become significant.
        """
        time_all = src.time_all if src.dynamic else mic.time_all
        time_zero_idx = src._time_zero_idx if src.dynamic else mic._time_zero_idx
        num_updates = len(time_all)

        rir_t0 = self.paths[src.name][mic.name]
        ir_len = rir_t0.shape[-1]

        rir_all_path = np.empty((num_updates, src.num, mic.num, ir_len))
        rir_all_path[: time_zero_idx + 1] = rir_t0  # warmup + t=0 row

        original_src_pos = src.pos
        original_mic_pos = mic.pos
        try:
            for i in range(time_zero_idx + 1, num_updates):
                if src.dynamic:
                    src.pos = src.pos_all[i]
                if mic.dynamic:
                    mic.pos = mic.pos_all[i]
                rir_all_path[i] = self.path_generator.create_path(
                    src, mic, reverb, sim_info
                )
        finally:
            src.pos = original_src_pos
            mic.pos = original_mic_pos

        self.rir_all.setdefault(src.name, {})[mic.name] = rir_all_path

    def update_path(self, src, mic):
        """Update the path between a source and a microphone array.

        Parameters
        ----------
        src : Array
            The source array
        mic : Array
            The microphone array
        """
        reverb = self.path_type[src.name][mic.name]
        assert reverb != "modified"
        self.paths[src.name][mic.name] = self.path_generator.create_path(
            src, mic, reverb, self.sim_info
        )

    def update(self, glob_idx: int):
        """Update the arrays and paths that change over time.

        The method performs the following steps
        1. update arrays pos/properties
        2. get info about which arrays actually changed
        3. update the paths connecting the updated arrays

        Parameters
        ----------
        glob_idx : int
            The global time index of the current sample
        """
        if glob_idx % self.sim_info.array_update_freq != 0:
            return

        changed_arrays = []

        for ar_name, ar in self.arrays.items():
            changed = ar.update(glob_idx)
            if changed:
                changed_arrays.append(ar_name)

        for ar_name in changed_arrays:
            changed_ar = self.arrays[ar_name]
            i = changed_ar.pos_idx_at(glob_idx)
            if changed_ar.is_mic:
                for src in self.sources():
                    if src.name in self.rir_all and ar_name in self.rir_all[src.name]:
                        self.paths[src.name][ar_name] = self.rir_all[src.name][ar_name][
                            i
                        ]
            elif changed_ar.is_source:
                for mic in self.mics():
                    if ar_name in self.rir_all and mic.name in self.rir_all[ar_name]:
                        self.paths[ar_name][mic.name] = self.rir_all[ar_name][mic.name][
                            i
                        ]
            else:
                raise ValueError("Array must be mic or source")

    def plot(
        self,
        sim_info: cfutil.SimulatorInfo,
        fig_folder: pathlib.Path,
        print_method: str,
    ):
        """Create a plot with all arrays in the collection.

        If ISM is chosen as the propagation type, the room is also plotted

        Parameters
        ----------
        sim_info : SimInfo
            The simulation info object
        fig_folder : pathlib.Path
            The folder where the figure should be saved
        print_method : str
            The method used to save the figure. See aspsim.diagnostics.plot.output_plot
            for a list of the options
        """
        fig, ax = plt.subplots()
        for ar in self.arrays.values():
            ar.plot(ax, sim_info)

        if "ism" in [
            self.path_type[src.name][mic.name] for src, mic in self.mic_src_combos()
        ]:
            corner = (
                sim_info.room_center[0] - sim_info.room_size[0] / 2,
                sim_info.room_center[1] - sim_info.room_size[1] / 2,
            )
            # bottom_left_corner = sim_info.room_center[:2] - (sim_info.room_size[:2] / 2)
            width = sim_info.room_size[0]
            height = sim_info.room_size[1]
            ax.add_patch(
                patches.Rectangle(
                    corner,
                    width,
                    height,
                    edgecolor="k",
                    facecolor="none",
                    linewidth=2,
                    alpha=1,
                )
            )

        ax.legend()
        ax.axis("equal")
        ax.grid(True)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        utils.save_plot(print_method, fig_folder, "array_pos")

    def save_to_file(self, folder_path):
        """Save the array collection to file.

        Parameters
        ----------
        folder_path : pathlib.Path
            The path to the folder where the collection should be saved
        """
        if folder_path is not None:
            with open(folder_path.joinpath("arrays.pickle"), "wb") as f:
                dill.dump(self, f)

    def get_freq_paths(self, num_freqs, samplerate):
        """Get the frequency domain response of the paths between all sources and microphones.

        Parameters
        ----------
        arrays : ArrayCollection
            The array collection to get the frequency paths from
        num_freqs : int
            The number of frequencies to calculate the response for. The number refers to the total FFT
            length, so the number of real frequencies is num_freqs // 2 + 1, which is the number of
            frequency bins in the output.

        Returns
        -------
        freq_paths : dict of dicts of ndarrays
            freq_paths[src_name][mic_name] is a complex ndarray of shape (num_real_freqs, num_mic, num_src)
            representing the frequency response between a source and a microphone array
        freqs : ndarray of shape (num_real_freqs,)
            The real frequencies in Hz of the response

        Notes
        -----
        The frequencies returned are compatible with get_real_freqs of the package aspcol, as well as the
        output of np.fft.rfft.

        num_freqs can be safely chosen as a larger number than the number of samples in the impulse response,
        as the FFT will zero-pad the signal to the desired length. But if num_freqs is smaller, then the
        signal will be truncated.
        """
        if num_freqs % 2 == 1:
            raise NotImplementedError("Odd number of frequencies not supported yet")

        freqs = (samplerate / (num_freqs)) * np.arange(num_freqs // 2 + 1)

        fpaths = {}
        for src, mic, path in self.iter_paths():
            fpaths.setdefault(src.name, {})
            fpaths[src.name][mic.name] = np.fft.rfft(path, n=num_freqs, axis=-1).T

        return fpaths, freqs


def load_arrays(folder_path):
    """Load the array collection from file.

    Parameters
    ----------
    folder_path : pathlib.Path
        The path to the folder where the collection should be loaded from
        The file is assumed to be named arrays.pickle
    """
    with open(folder_path.joinpath("arrays.pickle"), "rb") as f:
        arrays = dill.load(f)
    return arrays


def prototype_equals(prototype, initialized):
    """Compare an initialized collection to a prototype collection.

    This function checks if the initialized collection could be created from the prototype. i.e. if it
        would make sense to load initialized instead of generating the paths of the prototype

    Parameters
    ----------
    prototype : ArrayCollection
        Trototype collection need not have the paths initialized, but must
        have arrays added with positions set.
    initialized : ArrayCollection
        The initialized collection.
    """
    # check that they have the same array names
    if prototype.arrays.keys() != initialized.arrays.keys():
        return False

    # Check that the arrays are the same
    for ar_name, ar in prototype.arrays.items():
        # if ar != initialized[ar_name]:
        #    return False
        if type(ar) is not type(initialized[ar_name]):
            return False
        if ar.num != initialized[ar_name].num:
            return False
        if not np.allclose(ar.pos, initialized[ar_name].pos):
            return False

    # Check the paths are the same
    for src, mic in prototype.mic_src_combos():
        if prototype.path_type[src.name][mic.name] == "modified":
            if not np.allclose(
                prototype.paths[src.name][mic.name],
                initialized.paths[src.name][mic.name],
            ):
                return False
        if (
            prototype.path_type[src.name][mic.name]
            != initialized.path_type[src.name][mic.name]
        ):
            return False

    return True


class Array(ABC):
    """Abstract base class for all types of arrays.

    Establishes the common attributes and methods of all array types
    """

    is_mic = False
    is_source = False
    plot_symbol = "."

    def __init__(self, name, pos, directivity_type=None, directivity_dir=None):
        """Abstract base class for all types of arrays.

        Parameters
        ----------
        name : str
            name of the array, which will also be the name of the
            signal associated with the array
        pos : ndarray of shape (num_objects, spatial_dim)
            or a list of ndarrays of the same shape, which will distribute
                the objects into groups.
            or a Trajectory object, which will produce a moving array
        directivity_type : optional, list of str
            each string is one of "omni", "cardioid", "hypercardioid",
            "supercardioid", "bidirectional".
            If supplied, directivity_dir must also be supplied
            if not supplied, the array will be omnidirectional
        directivity_dir : optional, ndarray of shape (num_objects, spatial_dim)
            each row is a unit vector pointing in the direction of the directivity for that object
            If supplied, directivity_type must also be supplied

        Notes
        -----
        Documentation refers to objects, which means any of microphone, loudspeaker
        or source depending on array type.
        """
        self.name = name

        if isinstance(pos, (list, tuple)):
            assert all([isinstance(p, np.ndarray) for p in pos])
            num_in_group = np.cumsum([0] + [p.shape[0] for p in pos])
            self.group_idxs = [
                np.arange(num_in_group[i], num_in_group[i + 1])
                for i in range(len(num_in_group) - 1)
            ]
            self.num_groups = len(pos)
            self.pos = np.concatenate(pos, axis=0)
            self.pos_segments = pos
            self.dynamic = False
        elif isinstance(pos, np.ndarray):
            self.group_idxs = None
            self.num_groups = 1
            self.pos = pos
            self.pos_segments = [pos]
            self.dynamic = False
        elif isinstance(pos, tr.Trajectory):
            self.trajectory = pos
            self.group_idxs = None
            self.num_groups = 1
            self.pos = self.trajectory.current_pos(0)
            self.pos_segments = [pos]
            self.dynamic = True
            # pos_all, time_all, _time_zero_idx, _array_update_freq are set in prepare()
        else:
            raise ValueError("Incorrect datatype for pos")

        self.num = self.pos.shape[0]
        assert self.pos.ndim == 2

        if directivity_type is None or directivity_dir is None:
            assert directivity_type is None and directivity_dir is None
            directivity_type = ["omni" for _ in range(self.num)]
        else:
            assert directivity_type is not None and directivity_dir is not None
            assert len(directivity_type) == self.num
            assert directivity_dir.shape[0] == self.num
            assert all(
                [
                    d
                    in [
                        "omni",
                        "cardioid",
                        "hypercardioid",
                        "supercardioid",
                        "bidirectional",
                    ]
                    for d in directivity_type
                ]
            )
            assert np.allclose(np.linalg.norm(directivity_dir, axis=-1), 1)
        self.directivity_type = directivity_type
        self.directivity_dir = directivity_dir

        self.metadata = {
            "type": self.__class__.__name__,
            "number": self.num,
            "number of groups": self.num_groups,
            "dynamic": self.dynamic,
        }

    def prepare(self, sim_info: cfutil.SimulatorInfo):
        """Prepare the array for simulation.

        This method is called when the user creates the Simulator from the SimulatorSetup. This means that the sim_info is fixed, and the information can be used in the Array.

        For dynamic arrays, this precomputes ``time_all`` (signed sample indices) and
        ``pos_all`` (positions at each entry) for the whole simulation including the
        ``sim_buffer`` warmup window at negative time. During the warmup window the
        array is stationary at ``current_pos(0)``.

        Parameters
        ----------
        sim_info : SimulatorInfo
            The simulation info object
        """
        if not self.dynamic:
            return

        step = sim_info.array_update_freq
        # smallest multiple of step that is <= -sim_buffer
        start = -((sim_info.sim_buffer + step - 1) // step) * step
        stop = sim_info.tot_samples + sim_info.sim_buffer
        self.time_all = np.arange(start, stop, step)
        self._array_update_freq = step
        self._time_zero_idx = int(np.searchsorted(self.time_all, 0))

        positions = [
            self.trajectory.current_pos(0 if t < 0 else int(t)) for t in self.time_all
        ]
        self.pos_all = np.stack(positions, axis=0)

        self.pos = self.pos_all[self._time_zero_idx]

    def pos_idx_at(self, glob_idx: int) -> int:
        """Return the row in ``pos_all`` / ``rir_all`` corresponding to ``glob_idx``.

        Only valid for dynamic arrays after ``prepare`` has been called.
        """
        return self._time_zero_idx + glob_idx // self._array_update_freq

    def set_groups(self, group_idxs):
        """Sort each object of the array into groups.

        Parameters
        ----------
        group_idxs : list of lists or list of 1D nd.arrays
            Each inner list is the indices of the array elements
            belonging to one group.
        """
        self.num_groups = len(group_idxs)
        self.group_idxs = group_idxs

    def plot(self, ax, sim_info):
        """Plot the position of the array elements on the given axis.

        Parameters
        ----------
        ax : matplotlib axis
            The axis to plot on
        sim_info : SimulatorInfo
            The simulation info object, containing metadata about the simulation.
        """
        if self.dynamic:
            self.trajectory.plot(ax, self.plot_symbol, self.name, sim_info.tot_samples)
        else:
            ax.plot(
                self.pos[:, 0],
                self.pos[:, 1],
                self.plot_symbol,
                label=self.name,
                alpha=0.8,
            )

    def update(self, glob_idx: int):
        """Update the current position and other properties of the array if it is dynamic.

        Parameters
        ----------
        glob_idx : int
            The global time index of the current sample

        Returns
        -------
        changed : bool
            True if the array was updated, False otherwise
        """
        if self.dynamic:
            if glob_idx % self._array_update_freq != 0:
                return False
            self.pos = self.pos_all[self.pos_idx_at(glob_idx)]
            return True
        return False


class MicArray(Array):
    """Array class for microphones.

    Parameters
    ----------
    name : str
        name of the array, which will also be the name of the
        signal associated with the array
    pos : ndarray of shape (num_mics, spatial_dim)
        position of the microphones
    """

    is_mic = True
    plot_symbol = "x"

    def __init__(self, name, pos, **kwargs):
        super().__init__(name, pos, **kwargs)


class RegionArray(MicArray):
    """Class for representing a continuous region with an microphone array.

    Requires a Region object to be used as representation of the region. The Region object
    has a equally_spaced_points method, which is used to generate the positions of the microphones
    unless the pos argument is supplied. In that case, the signals generated will be
    identical to using a MicArray() with region.equally_spaced_points() as positions.

    Attributes
    ----------
    region : Region
        The region object used to represent the region
    region_segments : list of Region
        If the region is a CombinedRegion, this attribute will contain the individual regions
        that were combined to create the region.
    pos : ndarray of shape (num_mics, spatial_dim)
        The positions of the microphones that are used in the simulation
    """

    def __init__(self, name, region, pos=None, **kwargs):
        if isinstance(region, (list, tuple)):
            if len(region) > 1:
                self.region_segments = region
                region = reg.CombinedRegion(region)
            else:
                self.region_segments = region
                region = region[0]
        else:
            self.region_segments = [region]

        if pos is None:
            pos = [r.equally_spaced_points() for r in self.region_segments]
        super().__init__(name, pos, **kwargs)
        self.region = region
        self.metadata["region shape"] = self.region.__class__.__name__

    def plot(self, ax, sim_info):
        """Plot the region and the microphone positions.

        Parameters
        ----------
        ax : matplotlib axis
            The axis to plot on
        sim_info : SimulatorInfo
            The simulation info object, containing metadata about the simulation.
        """
        self.region.plot(ax, self.name)


class ControllableSourceArray(Array):
    """Array for sources controllable by a processor.

    Parameters
    ----------
    name : str
        name of the array, which will also be the name of the
        signal associated with the array
    pos : ndarray of shape (num_sources, spatial_dim)
        position of the sources

    """

    is_source = True
    plot_symbol = "o"

    def __init__(self, name, pos, **kwargs):
        super().__init__(name, pos, **kwargs)


class FreeSourceArray(Array):
    """Array for free sound sources, that cannot be adaptively controlled.

    Parameters
    ----------
    name : str
        name of the array, which will also be the name of the
        signal associated with the array
    pos : ndarray of shape (num_sources, spatial_dim)
        position of the sources
    source : Source
        The source object used to generate the signal
        see aspsim.signal.source module for more info. More sources are also available
        in the aspsim.room.sourcescollection module.

    """

    is_source = True
    plot_symbol = "s"

    def __init__(self, name, pos, source, **kwargs):
        super().__init__(name, pos, **kwargs)
        self.set_source(source)

        self.metadata["source info"] = self.source.metadata

    def get_samples(self, num_samples: int):
        """Get signal samples from the source.

        Parameters
        ----------
        num_samples : int
            The number of samples to be generated

        Returns
        -------
        samples : ndarray of shape (num_sources, num_samples)
            The generated signal samples for each source in the array
        """
        return self.source.get_samples(num_samples)

    def set_source(self, source):
        """Set the source object for the array.

        Parameters
        ----------
        source : Source
            The source object used to generate the signal.
            See aspsim.signal.source module for more info.
            More sources are also available in the aspsim.room.sourcescollection module.
        """
        assert source.num_channels == self.num
        self.source = source
