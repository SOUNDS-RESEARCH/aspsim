import os
import ancsim.experiment.multiexperimentplots as mep
import ancsim.experiment.multiexperimentutils as meu
import ancsim.experiment.soundfieldplot as sfp



def generateFullExpData(multiExpFolder):
    #[x for x in p.iterdir() if x.is_dir()]
    for singleSetFolder in multiExpFolder.iterdir():
        if singleSetFolder.is_dir():
            generateSingleExpData(multiExpFolder.joinpath(singleSetFolder))

def generateSingleExpData(singleSetFolder):
    meu.extractSettings(singleSetFolder)
    meu.extractSummaries(singleSetFolder, "latest")
    meu.extractAllSummaries(singleSetFolder)
    meu.extractConfigs(singleSetFolder)
    meu.extractFilterParameters(singleSetFolder)
    
    summary, settings, config, filtParams = meu.openData(singleSetFolder)
    # entries = meu.findChangingEntries(settings)
    # for entry in entries:
    #     mep.plotSingleEntryMetrics(entry, summary, settings, singleSetFolder)
    entries = meu.findChangingEntries(config)
    for entry in entries:
        mep.plotSingleEntryMetrics(entry, summary, config, singleSetFolder)
        
    filtEntries = meu.findFilterChangingEntries(filtParams)
    for filtName, entries in filtEntries.items():
        for entry in entries:
            mep.plotSingleEntryMetrics(entry, summary, filtParams[filtName], singleSetFolder)

    if "NOISEFREQ" in entries:
        mep.reductionByFrequency(singleSetFolder, "latest")

    #sfp.generateSoundfieldForFolder(singleSetFolder)

if __name__ == "__main__":
    generateSingleExpData("multi_experiments/full_exp_2020_04_22_00_42_0/single_set_2020_04_22_00_42_0/")