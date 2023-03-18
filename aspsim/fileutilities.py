import datetime




def get_time_string(detailed=False):
    tm = datetime.datetime.now()
    timestr = (
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
        timestr += "_" + str(tm.second).zfill(2)
        timestr += "_" + str(tm.microsecond).zfill(2)
    return timestr


def get_unique_folder_name(prefix, parent_folder, detailed_naming=False):
    fileName = prefix + get_time_string(detailed=detailed_naming)
    fileName += "_0"
    folderName = parent_folder.joinpath(fileName)
    if folderName.exists():
        idx = 1
        folderNameLen = len(folderName.name) - 2
        while folderName.exists():
            newName = folderName.name[:folderNameLen] + "_" + str(idx)
            folderName = folderName.parent.joinpath(newName)
            idx += 1
    # folderName += "/"
    return folderName


def get_multiple_unique_folder_names(prefix, parent_folder, num_names):
    startPath = get_unique_folder_name(prefix, parent_folder)
    subFolderName = startPath.parts[-1]
    baseFolder = startPath.parent

    startIdx = int(subFolderName.split("_")[-1])
    startIdxLen = len(subFolderName.split("_")[-1])
    baseName = subFolderName[:-startIdxLen]

    folderNames = []
    for i in range(num_names):
        folderNames.append(baseFolder.joinpath(baseName + str(i + startIdx)))

    return folderNames





def indices_same_in_all_folders(folder_name, prefix, suffix, excluded_folders=[]):
    firstSubFolder = True
    prevGoodIndices = []
    for subFolderName in folder_name.iterdir():
        if subFolderName in excluded_folders:
            continue
        if folder_name.joinpath(subFolderName).is_dir():
            goodIndices = []
            for filePath in folder_name.joinpath(subFolderName).iterdir():
                filename = filePath.name
                if filename.startswith(prefix) and filename.endswith(suffix):
                    summaryIdx = filename[len(prefix) : len(filename) - len(suffix)]
                    try:
                        summaryIdx = int(summaryIdx)
                        if firstSubFolder or summaryIdx in prevGoodIndices:
                            goodIndices.append(summaryIdx)
                    except ValueError:
                        pass
            prevGoodIndices = goodIndices
            firstSubFolder = False
    return goodIndices


def get_highest_numbered_file(folder, prefix, suffix):
    highestFileIdx = -1
    for filePath in folder.iterdir():
        if filePath.name.startswith(prefix) and filePath.name.endswith(suffix):
            summaryIdx = filePath.name[len(prefix) : len(filePath.name) - len(suffix)]
            try:
                summaryIdx = int(summaryIdx)
                if summaryIdx > highestFileIdx:
                    highestFileIdx = summaryIdx
            except ValueError:
                print("Warning: check prefix and suffix")

    if highestFileIdx == -1:
        return None
    else:
        fname = prefix + str(highestFileIdx) + suffix
        return folder.joinpath(fname)


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


def find_all_earlier_files(
    folder, name, current_idx, name_includes_idx=True, error_if_future_files_exist=True
):
    if name_includes_idx:
        name = name[: -len(str(current_idx))]
    else:
        name = name + "_"

    earlierFiles = []
    for f in folder.iterdir():
        if f.stem.startswith(name) and f.stem[len(name):].isdigit():
            fIdx = int(f.stem[len(name) :])
            if fIdx > current_idx:
                if error_if_future_files_exist:
                    raise ValueError
                else:
                    continue
            elif fIdx == current_idx:
                continue
            earlierFiles.append(f)
    return earlierFiles



def to_num(val):
    constructors = [int, float, str]
    for c in constructors:
        try:
            val = c(val)
            return val
        except ValueError:
            pass



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