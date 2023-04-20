import datetime




def get_time_string(detailed=False):
    tm = datetime.datetime.now()
    time_str = (
        str(tm.year)
        + "_"
        + str(tm.month).zfill(2)
        + "_"
        + str(tm.day).zfill(2)
        + "_"
        + str(tm.hour).zfill(2)
        + "_"
        + str(tm.minute).zfill(2)
    )  # + "_"+\
    # str(tm.second).zfill(2)
    if detailed:
        time_str += "_" + str(tm.second).zfill(2)
        time_str += "_" + str(tm.microsecond).zfill(2)
    return time_str


def get_unique_folder_name(prefix, parent_folder, detailed_naming=False):
    file_name = prefix + get_time_string(detailed=detailed_naming)
    file_name += "_0"
    folder_name = parent_folder.joinpath(file_name)
    if folder_name.exists():
        idx = 1
        folder_name_len = len(folder_name.name) - 2
        while folder_name.exists():
            new_name = folder_name.name[:folder_name_len] + "_" + str(idx)
            folder_name = folder_name.parent.joinpath(new_name)
            idx += 1
    # folderName += "/"
    return folder_name


def get_multiple_unique_folder_names(prefix, parent_folder, num_names):
    start_path = get_unique_folder_name(prefix, parent_folder)
    sub_folder_name = start_path.parts[-1]
    base_folder = start_path.parent

    start_idx = int(sub_folder_name.split("_")[-1])
    start_idx_len = len(sub_folder_name.split("_")[-1])
    base_name = sub_folder_name[:-start_idx_len]

    folder_names = []
    for i in range(num_names):
        folder_names.append(base_folder.joinpath(base_name + str(i + start_idx)))

    return folder_names


def get_highest_numbered_file(folder, prefix, suffix):
    highest_file_idx = -1
    for file_path in folder.iterdir():
        if file_path.name.startswith(prefix) and file_path.name.endswith(suffix):
            summary_idx = file_path.name[len(prefix) : len(file_path.name) - len(suffix)]
            try:
                summary_idx = int(summary_idx)
                if summary_idx > highest_file_idx:
                    highest_file_idx = summary_idx
            except ValueError:
                print("Warning: check prefix and suffix")

    if highest_file_idx == -1:
        return None
    else:
        fname = prefix + str(highest_file_idx) + suffix
        return folder.joinpath(fname)


def find_all_earlier_files(
    folder, name, current_idx, name_includes_idx=True, error_if_future_files_exist=True
):
    if name_includes_idx:
        name = name[: -len(str(current_idx))]
    else:
        name = name + "_"

    earlier_files = []
    for f in folder.iterdir():
        if f.stem.startswith(name) and f.stem[len(name):].isdigit():
            f_idx = int(f.stem[len(name) :])
            if f_idx > current_idx:
                if error_if_future_files_exist:
                    raise ValueError
                else:
                    continue
            elif f_idx == current_idx:
                continue
            earlier_files.append(f)
    return earlier_files


def find_index_in_name(name):
    idx = []
    for ch in reversed(name):
        if ch.isdigit():
            idx.append(ch)
        else:
            break
    if len(idx) == 0:
        return None
    idx = int("".join(idx[::-1]))
    assert name.endswith(str(idx))
    return idx