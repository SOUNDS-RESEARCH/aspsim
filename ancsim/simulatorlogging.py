from pathlib import Path
import json
import ancsim.utilities as util


def getFolderName(config, folderForPlots, generateSubFolder=True, safeNaming=False):
    packageDir = Path(__file__).parent
    if config["PLOTOUTPUT"] != "none":
        if generateSubFolder:
            # folderName = createFigFolder(packageDir.parent.joinpath(folderForPlots), safeNaming)
            folderName = createFigFolder(folderForPlots, safeNaming)
        else:
            folderName = folderForPlots
            if not folderName.exists():
                folderName.mkdir()
    else:
        folderName = ""
    return folderName


def createFigFolder(parentFolder=None, safeNaming=False):
    if parentFolder is None:
        parentFolder = ""
    folderName = util.getUniqueFolderName("figs_", parentFolder, safeNaming)
    folderName.mkdir()
    return folderName


def addToSimMetadata(folderPath, dictToAdd):
    try:
        with open(folderPath.joinpath("simmetadata.json"), "r") as f:
            oldData = json.load(f)
            totData = {**oldData, **dictToAdd}
    except FileNotFoundError:
        totData = dictToAdd
    with open(folderPath.joinpath("simmetadata.json"), "w") as f:
        json.dump(totData, f, indent=4)


def writeFilterMetadata(filters, folderPath):
    fileName = "filtermetadata.json"
    totMetadata = {}
    for filt in filters:
        totMetadata[filt.name] = filt.metadata
    with open(folderPath.joinpath(fileName), "w") as f:
        json.dump(totMetadata, f, indent=4)
