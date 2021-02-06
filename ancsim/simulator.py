import numpy as np

import ancsim.utilities as util
import ancsim.configutil as configutil

from ancsim.simulatorsetup import setupIR
from ancsim.simulatorlogging import addToSimMetadata, writeFilterMetadata

import ancsim.saveloadsession as sess
import ancsim.experiment.plotscripts as psc
import ancsim.experiment.multiexperimentutils as meu
from ancsim.adaptivefilter.diagnostics import PlotDiagnosticDispatcher


class Simulator:
    def __init__(
        self,
        folderForPlots,
        sessionFolder=None,
    ):
        self.filters = []
        self.freeSources = []
        self.ctrlSources = []
        self.mics = []

        self.config = configutil.getBaseConfig()

        self.folderPath = createFigFolder(self.config, folderForPlots, False)
        self.sessionFolder = sessionFolder

        print("Figure folder: ", self.folderPath.name)

    def setConfig(config):
        self.config = config

    def addFreeSource(self, name, pos):
        self.freeSources.append(name)
        self.pos[name] = pos

    def addControllableSource(self, name, pos):
        self.ctrlSources.append(name)
        self.pos[name]
    
    def addMics(self, name, pos):
        self.mics.append(name)
        self.pos[name] = pos

    def addProcessor(self, processor):
        try:
            self.filters.extend(processor)
        except TypeError:
            self.filters.append(processor)
        
    def loadSession(self, sessionPath):
        self.config, self.pos, self.arrays, ,  = sess.loadFromPath(sessionPath, self.folderPath)

    def prepare(self):
        if self.config["AUTOLOADSESSION"]:
            self.sourceFilters, self.speakerFilters = sess.loadSession(
                self.sessionFolder, self.folderPath, self.config
            )
        else:
            self.sourceFilters, self.speakerFilters, irMetadata = setupIR(
                self.pos, config
            )
            addToSimMetadata(self.folderPath, irMetadata)

        # LOGGING AND DIAGNOSTICS
        sess.saveConfig(self.folderPath, self.config)
        plotAnyPos(self.pos, self.folderPath, self.config)
        

    def runSimulation(self):
        self.config = configPreprocessing(self.config, len(self.filters))

        setUniqueFilterNames(self.filters)
        writeFilterMetadata(self.filters, self.folderPath)

        self.plotDispatcher = PlotDiagnosticDispatcher(
            self.filters, self.config["PLOTOUTPUT"]
        )


        print("Filling buffers...")
        noises = self._fillBuffers()
        for filt in self.filters:
            filt.prepare()

        print("SIM START")
        n_tot = 0
        bufferIdx = -1
        noises = self._updateNoises(n_tot, noises)
        noiseIndices = [self.config["SIMBUFFER"] for _ in range(len(self.filters))]
        while n_tot < self.config["ENDTIMESTEP"]-self.config["LARGESTBLOCKSIZE"]:
            bufferIdx += 1
            for n in range(
                self.config["SIMBUFFER"],
                self.config["SIMBUFFER"]
                + min(self.config["SIMCHUNKSIZE"], self.config["ENDTIMESTEP"] - n_tot),
            ):
                if n_tot % 1000 == 0:
                    print("Timestep: ", n_tot)
                if self.config["ENDTIMESTEP"] - self.config["LARGESTBLOCKSIZE"] < n_tot:
                    break

                if (
                    np.max(
                        [
                            nIdx + bSize
                            for nIdx, bSize in zip(
                                noiseIndices, self.config["BLOCKSIZE"]
                            )
                        ]
                    )
                    >= self.config["SIMBUFFER"] + self.config["SIMCHUNKSIZE"]
                ):
                    noises = self._updateNoises(n_tot, noises)
                    noiseIndices = [
                        i - self.config["SIMCHUNKSIZE"] for i in noiseIndices
                    ]

                for i, (filt, bSize, nIdx) in enumerate(
                    zip(self.filters, self.config["BLOCKSIZE"], noiseIndices)
                ):
                    if n_tot % bSize == 0:
                        # filt.forwardPass(bSize, *[noiseAtPoints[:,nIdx:nIdx+bSize] for noiseAtPoints in noises])
                        filt.process(bSize,
                            {
                                pointName: noiseAtPoints[:, nIdx : nIdx + bSize]
                                for pointName, noiseAtPoints in noises.items()
                            },
                        )
                        # filt.forwardPass(
                        #     bSize,
                        #     {
                        #         pointName: noiseAtPoints[:, nIdx : nIdx + bSize]
                        #         for pointName, noiseAtPoints in noises.items()
                        #     },
                        # )
                        # filt.updateFilter()
                        noiseIndices[i] += bSize

                if n_tot % self.config["SIMCHUNKSIZE"] == 0 and n_tot > 0:
                    if self.config["PLOTOUTPUT"] != "none":
                        self.plotDispatcher.dispatch(
                            self.filters, n_tot, bufferIdx, self.folderPath
                        )
                    if self.config["SAVERAWDATA"] and (
                        bufferIdx % self.config["SAVERAWDATAFREQUENCY"] == 0
                        or bufferIdx - 1 == 0
                    ):
                        sess.saveRawData(self.filters, n_tot, self.folderPath)

                n_tot += 1

    def _updateNoises(self, timeIdx, noises):
        noise = self.noiseSource.getSamples(self.config["SIMCHUNKSIZE"])
        noises = {
            filtName: np.concatenate(
                (noises[filtName][:, -self.config["SIMBUFFER"] :], sf.process(noise)),
                axis=-1,
            )
            for filtName, sf in self.sourceFilters.items()
        }
        return noises

    def _fillBuffers(self):
        noise = self.noiseSource.getSamples(self.config["SIMBUFFER"])
        noises = {
            filtName: sf.process(noise) for filtName, sf in self.sourceFilters.items()
        }

        maxSecPathLength = np.max(
            [speakerFilt.shape[-1] for _, speakerFilt, in self.speakerFilters.items()]
        )
        fillStartIdx = self.config["LARGESTBLOCKSIZE"] + np.max(
            (self.config["FILTLENGTH"] + self.config["KERNFILTLEN"], maxSecPathLength)
        )
        fillNumBlocks = [
            (self.config["SIMBUFFER"] - fillStartIdx) // bs
            for bs in self.config["BLOCKSIZE"]
        ]

        startIdxs = []
        for filtIdx, blockSize in enumerate(self.config["BLOCKSIZE"]):
            startIdxs.append([])
            for i in range(fillNumBlocks[filtIdx]):
                startIdxs[filtIdx].append(
                    self.config["SIMBUFFER"]
                    - ((fillNumBlocks[filtIdx] - i) * blockSize)
                )

        for filtIdx, (filt, blockSize) in enumerate(
            zip(self.filters, self.config["BLOCKSIZE"])
        ):
            filt.idx = startIdxs[filtIdx][0]
            for i in startIdxs[filtIdx]:
                filt.forwardPass(
                    blockSize,
                    {
                        pointName: noiseAtPoints[:, i : i + blockSize]
                        for pointName, noiseAtPoints in noises.items()
                    },
                )

        return noises


def setUniqueFilterNames(filters):
    names = []
    for filt in filters:
        newName = filt.name
        i = 1
        while newName in names:
            newName = filt.name + " " + str(i)
            i += 1
        names.append(newName)
        filt.name = newName


def plotAnyPos(pos, folderPath, config):
    print("Setup Positions")
    if config["SPATIALDIMS"] == 3:
        if config["ARRAYSHAPES"] == "circle":
            psc.plotPos3dDisc(pos, folderPath, config, config["PLOTOUTPUT"])
        elif config["ARRAYSHAPES"] in ("cuboid", "rectangle", "doublerectangle", "smaller_rectangle"):
            if config["REVERB"] == "ism":
                psc.plotPos3dRect(pos, folderPath, config, config["ROOMSIZE"], config["ROOMCENTER"], printMethod=config["PLOTOUTPUT"])
            else:
                psc.plotPos3dRect(
                    pos, folderPath, config, printMethod=config["PLOTOUTPUT"]
                )
        else:
            psc.plotPos(pos, folderPath,config, printMethod=config["PLOTOUTPUT"])
    elif config["SPATIALDIMS"] == 2:
        if config["ARRAYSHAPES"] == "circle":
            if config["REFDIRECTLYOBTAINED"]:
                raise NotImplementedError
            else:
                psc.plotPos2dDisc(pos, folderPath, config, config["PLOTOUTPUT"])
        else:
            raise NotImplementedError
    else:
        raise ValueError
