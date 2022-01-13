
def indicesSameInAllFolders(folderName, prefix, suffix, excludedFolders=[]):
    firstSubFolder = True
    prevGoodIndices = []
    for subFolderName in folderName.iterdir():
        if subFolderName in excludedFolders:
            continue
        if folderName.joinpath(subFolderName).is_dir():
            goodIndices = []
            for filePath in folderName.joinpath(subFolderName).iterdir():
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


def getHighestNumberedFile(folder, prefix, suffix):
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


def findAllEarlierFiles(
    folder, name, currentIdx, nameIncludesIdx=True, errorIfFutureFilesExist=True
):
    if nameIncludesIdx:
        name = name[: -len(str(currentIdx))]
    else:
        name = name + "_"

    earlierFiles = []
    for f in folder.iterdir():
        if f.stem.startswith(name) and f.stem[len(name):].isdigit():
            fIdx = int(f.stem[len(name) :])
            if fIdx > currentIdx:
                if errorIfFutureFilesExist:
                    raise ValueError
                else:
                    continue
            elif fIdx == currentIdx:
                continue
            earlierFiles.append(f)
    return earlierFiles



def toNum(val):
    constructors = [int, float, str]
    for c in constructors:
        try:
            val = c(val)
            return val
        except ValueError:
            pass
