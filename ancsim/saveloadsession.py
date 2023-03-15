import shutil
import json

import ancsim.fileutilities as futil
import ancsim.configutil as configutil
import ancsim.array as ar



def save_session(session_folder, sim_info, arrays, sim_metadata=None, extraprefix=""):
    session_path = futil.get_unique_folder_name("session_" + extraprefix, session_folder)

    session_path.mkdir()
    arrays.save_to_file(session_path)
    sim_info.save_to_file(session_path)
    if sim_metadata is not None:
        add_to_sim_metadata(session_path, sim_metadata)

def load_session(sessions_path, new_folder_path, chosen_sim_info, chosen_arrays):
    """sessionsPath refers to the folder where all sessions reside 
        This function will load a session matching the chosenConfig 
        and chosenArrays if it exists"""
    session_to_load = search_for_matching_session(sessions_path, chosen_sim_info, chosen_arrays)
    print("Loaded Session: ", str(session_to_load))
    loaded_arrays = ar.load_arrays(session_to_load)

    for fs_array in chosen_arrays.free_sources():
        loaded_arrays[fs_array.name].source = fs_array.source

    return loaded_arrays
    
def load_from_path(session_path_to_load, new_folder_path=None):
    loaded_arrays = ar.load_arrays(session_path_to_load)
    loaded_sim_info = configutil.load_from_file(session_path_to_load)

    if new_folder_path is not None:
        copy_sim_metadata(session_path_to_load, new_folder_path)
        loaded_sim_info.save_to_file(new_folder_path)
    return loaded_sim_info, loaded_arrays

def copy_sim_metadata(from_folder, to_folder):
    shutil.copy(
        from_folder.joinpath("metadata_sim.json"), to_folder.joinpath("metadata_sim.json")
    )


class MatchingSessionNotFoundError(ValueError): pass

def search_for_matching_session(sessions_path, chosen_sim_info, chosen_arrays):
    for dir_path in sessions_path.iterdir():
        if dir_path.is_dir():
            loaded_sim_info = configutil.load_from_file(dir_path)
            loaded_arrays = ar.load_arrays(dir_path)

            if configutil.equal_audio(chosen_sim_info, loaded_sim_info, chosen_arrays.path_type) and \
                ar.ArrayCollection.prototype_equals(chosen_arrays, loaded_arrays):
                return dir_path
    raise MatchingSessionNotFoundError("No matching saved sessions")



def add_to_sim_metadata(folder_path, dict_to_add):
    try:
        with open(folder_path.joinpath("metadata_sim.json"), "r") as f:
            old_data = json.load(f)
            tot_data = {**old_data, **dict_to_add}
    except FileNotFoundError:
        tot_data = dict_to_add
    with open(folder_path.joinpath("metadata_sim.json"), "w") as f:
        json.dump(tot_data, f, indent=4)


def write_processor_metadata(processors, folder_path):
    if folder_path is None:
        return
        
    file_name = "metadata_processor.json"
    tot_metadata = {}
    for proc in processors:
        tot_metadata[proc.name] = proc.metadata
    with open(folder_path.joinpath(file_name), "w") as f:
        json.dump(tot_metadata, f, indent=4)