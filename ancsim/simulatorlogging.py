from pathlib import Path
import json
import ancsim.utilities as util


def addToSimMetadata(folderPath, dictToAdd):
    try:
        with open(folderPath.joinpath("metadata_sim.json"), "r") as f:
            oldData = json.load(f)
            totData = {**oldData, **dictToAdd}
    except FileNotFoundError:
        totData = dictToAdd
    with open(folderPath.joinpath("metadata_sim.json"), "w") as f:
        json.dump(totData, f, indent=4)


def writeFilterMetadata(filters, folderPath):
    fileName = "metadata_processor.json"
    totMetadata = {}
    for filt in filters:
        totMetadata[filt.name] = filt.metadata
    with open(folderPath.joinpath(fileName), "w") as f:
        json.dump(totMetadata, f, indent=4)
