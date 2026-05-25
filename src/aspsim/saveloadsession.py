"""Helpers for saving and loading simulation sessions."""

import json
import shutil

import aspsim.array as ar
import aspsim.configutil as configutil
import aspsim.fileutilities as futil


def save_session(session_folder, sim_info, arrays, sim_metadata=None, extraprefix=""):
    """Save a simulation session to a new folder.

    Parameters
    ----------
    session_folder : pathlib.Path
        Folder where sessions are stored.
    sim_info : SimulatorInfo
        Simulation configuration to save.
    arrays : ArrayCollection
        Array collection to save.
    sim_metadata : dict, optional
        Additional metadata to add to the session.
    extraprefix : str, optional
        Extra prefix for the session folder name.
    """
    session_path = futil.get_unique_folder_name(
        "session_" + extraprefix, session_folder
    )

    session_path.mkdir()
    arrays.save_to_file(session_path)
    sim_info.save_to_file(session_path)
    if sim_metadata is not None:
        add_to_sim_metadata(session_path, sim_metadata)


def load_session(sessions_path, new_folder_path, chosen_sim_info, chosen_arrays):
    """Load a session matching the chosen configuration and arrays.

    Parameters
    ----------
    sessions_path : pathlib.Path
        Folder where all sessions reside.
    new_folder_path : pathlib.Path
        Target folder for copied metadata.
    chosen_sim_info : SimInfo
        Requested simulation configuration.
    chosen_arrays : ArrayCollection
        Requested arrays.

    Returns
    -------
    ArrayCollection
        Loaded array collection.
    """
    session_to_load = search_for_matching_session(
        sessions_path, chosen_sim_info, chosen_arrays
    )
    print("Loaded Session: ", str(session_to_load))
    loaded_arrays = ar.load_arrays(session_to_load)

    for fs_array in chosen_arrays.free_sources():
        loaded_arrays[fs_array.name].source = fs_array.source

    return loaded_arrays


def load_from_path(session_path_to_load, new_folder_path=None):
    """Load a session from a specific path.

    Parameters
    ----------
    session_path_to_load : pathlib.Path
        Path to the session folder.
    new_folder_path : pathlib.Path, optional
        Folder where metadata should be copied.

    Returns
    -------
    loaded_sim_info : SimulatorInfo
        Loaded simulation configuration.
    loaded_arrays : ArrayCollection
        Loaded array collection.
    """
    loaded_arrays = ar.load_arrays(session_path_to_load)
    loaded_sim_info = configutil.load_from_file(session_path_to_load)

    if new_folder_path is not None:
        copy_sim_metadata(session_path_to_load, new_folder_path)
        loaded_sim_info.save_to_file(new_folder_path)
    return loaded_sim_info, loaded_arrays


def copy_sim_metadata(from_folder, to_folder):
    """Copy simulation metadata JSON between folders.

    Parameters
    ----------
    from_folder : pathlib.Path
        Source folder.
    to_folder : pathlib.Path
        Destination folder.
    """
    shutil.copy(
        from_folder.joinpath("metadata_sim.json"),
        to_folder.joinpath("metadata_sim.json"),
    )


class MatchingSessionNotFoundError(ValueError):
    """Raised when no matching session is found."""

    pass


def search_for_matching_session(sessions_path, chosen_sim_info, chosen_arrays):
    """Find a session matching the chosen configuration and arrays.

    Parameters
    ----------
    sessions_path : pathlib.Path
        Folder where all sessions reside.
    chosen_sim_info : SimulatorInfo
        Requested simulation configuration.
    chosen_arrays : ArrayCollection
        Requested arrays.

    Returns
    -------
    pathlib.Path
        Path to the matching session.

    Raises
    ------
    MatchingSessionNotFoundError
        If no matching session is found.
    """
    for dir_path in sessions_path.iterdir():
        if dir_path.is_dir():
            loaded_sim_info = configutil.load_from_file(dir_path)
            loaded_arrays = ar.load_arrays(dir_path)

            if configutil.equal_audio(
                chosen_sim_info, loaded_sim_info, chosen_arrays.path_type
            ) and ar.prototype_equals(chosen_arrays, loaded_arrays):
                return dir_path
    raise MatchingSessionNotFoundError("No matching saved sessions")


def add_to_sim_metadata(folder_path, dict_to_add):
    """Merge metadata into the simulation metadata file.

    Parameters
    ----------
    folder_path : pathlib.Path
        Folder containing the metadata file.
    dict_to_add : dict
        Metadata to add or update.
    """
    try:
        with open(folder_path.joinpath("metadata_sim.json"), "r") as f:
            old_data = json.load(f)
            tot_data = {**old_data, **dict_to_add}
    except FileNotFoundError:
        tot_data = dict_to_add
    with open(folder_path.joinpath("metadata_sim.json"), "w") as f:
        json.dump(tot_data, f, indent=4)


def write_processor_metadata(processors, folder_path):
    """Write processor metadata to disk.

    Parameters
    ----------
    processors : iterable
        Processors providing metadata.
    folder_path : pathlib.Path or None
        Target folder to write metadata into.
    """
    if folder_path is None:
        return

    file_name = "metadata_processor.json"
    tot_metadata = {}
    for proc in processors:
        tot_metadata[proc.name] = proc.metadata
    with open(folder_path.joinpath(file_name), "w") as f:
        json.dump(tot_metadata, f, indent=4)
