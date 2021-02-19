import numpy as np

import ancsim.utilities as util
import ancsim.configutil as configutil

from ancsim.array import ArrayCollection
from ancsim.simulatorlogging import addToSimMetadata, writeFilterMetadata

import ancsim.saveloadsession as sess
import ancsim.experiment.plotscripts as psc
import ancsim.experiment.multiexperimentutils as meu
import ancsim.soundfield.setuparrays as preset
import ancsim.soundfield.roomimpulseresponse as rir



class Simulator:
    def __init__(
        self,
        baseFolderPath,
        sessionFolder=None,
    ):
        self.processors = []
        self.arrays = ArrayCollection()
        self.propagationPaths = {}
        #self.propPathTypes = {}
        #self.reverbExceptions = ((),)

        self.setConfig(configutil.getBaseConfig())

        self.baseFolderPath = baseFolderPath
        self.sessionFolder = sessionFolder

    def setConfig(self, config):
        self.config = configutil.processConfig(config)
        
        for mic, src in self.arrays.paths():
            self.propagationPaths[src][mic] = self.config["REVERB"]

    def addArrays(self, arrayCollection):
        for ar in arrayCollection:
            self.arrays.add_array(ar)
            if ar.is_source:
                self.propagationPaths[ar.name] = {}

    def addArray(self, array):
        self.arrays.add_array(array)
        if array.is_source():
            self.propagationPaths[array.name] = {}

    def addFreeSource(self, name, pos):
        arr = Array(name, ArrayType.FREESOURCE, pos)
        self.addArray(arr)
        self.propagationPaths[name] = {}

    def addControllableSource(self, name, pos):
        arr = Array(name, ArrayType.CTRLSOURCE, pos)
        self.addArray(arr)
        self.propagationPaths[name] = {}
    
    def addMics(self, name, pos):
        arr = Array(name, ArrayType.MIC, pos)
        self.addArray(arr)

    def setSource(self, name, source):
        self.arrays[name].setSource(source)

    def addProcessor(self, processor):
        try:
            self.processors.extend(processor)
        except TypeError:
            self.processors.append(processor)
        
    def loadFromPath(self, sessionPath):
        self.config, self.arrays = sess.loadFromPath(sessionPath, self.folderPath)

    # def loadSession(self, sessionPath=None, config=None):
    #     if config is not None:
    #         return sess.loadSession(sessionPath, self.folderPath, config)
    #     else:
    #         return sess.loadSession(sessionPath, self.folderPath)

    def usePreset(self, presetName, **kwargs):
        presetFunctions = {
            "cuboid" : preset.getPositionsCuboid3d,
            "cylinder" : preset.getPositionsCylinder3d,
        }

        arrays, chosenPropPaths = presetFunctions[presetName](self.config, **kwargs)
        self.addArrays(arrays)
        for src, micSet in chosenPropPaths.items():
            for mic, pathVal in micSet.items():
                self.propagationPaths[src][mic] = pathVal

    
    def prepareNewSession(self):
        irMetadata = self.setupIR(ArrayType.FREESOURCE, self.reverbExceptions)
        #self.propFiltersCtrl, irMetadataCtrl = self.setupIR(ArrayType.CTRLSOURCE, self.reverbExceptions)
        addToSimMetadata(self.folderPath, irMetadata)

        if self.config["AUTOSAVELOAD"]:
            sess.saveSession(self.arrays, self.propFiltersFree, self.propFiltersCtrl)

    def prepare(self):
        assert not self.arrays.empty()

        self.folderPath = self.createFigFolder(self.baseFolderPath)
        print(f"Figure folder: {self.folderPath}")

        if self.config["AUTOSAVELOAD"] and self.sessionFolder is not None:
            sess.loadSession(self.sessionFolder, self.folderPath)
        print(self.propagationPaths)
        self.setupIR()

        #self.noiseSource = setupSource(config)

        # LOGGING AND DIAGNOSTICS
        sess.saveConfig(self.folderPath, self.config)
        #psc.plotArrays(self.arrays)
        #plotAnyPos(self.pos, self.folderPath, self.config)
        

    def runSimulation(self):
        self.config = configPreprocessing(self.config, len(self.processors))

        setUniqueFilterNames(self.processors)
        writeFilterMetadata(self.processors, self.folderPath)

        self.plotDispatcher = PlotDiagnosticDispatcher(
            self.processors, self.config["PLOTOUTPUT"]
        )


        print("Filling buffers...")
        noises = self._fillBuffers()
        for filt in self.processors:
            filt.prepare()

        print("SIM START")
        n_tot = 0
        bufferIdx = -1
        noises = self._updateNoises(n_tot, noises)
        noiseIndices = [self.config["SIMBUFFER"] for _ in range(len(self.processors))]
        while n_tot < self.config["ENDTIMESTEP"]-np.max(self.config["BLOCKSIZE"]):
            bufferIdx += 1
            for n in range(
                self.config["SIMBUFFER"],
                self.config["SIMBUFFER"]
                + min(self.config["SIMCHUNKSIZE"], self.config["ENDTIMESTEP"] - n_tot),
            ):
                if n_tot % 1000 == 0:
                    print("Timestep: ", n_tot)
                if self.config["ENDTIMESTEP"] - np.max(self.config["BLOCKSIZE"]) < n_tot:
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
                    zip(self.processors, self.config["BLOCKSIZE"], noiseIndices)
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
                            self.processors, n_tot, bufferIdx, self.folderPath
                        )
                    if self.config["SAVERAWDATA"] and (
                        bufferIdx % self.config["SAVERAWDATAFREQUENCY"] == 0
                        or bufferIdx - 1 == 0
                    ):
                        sess.saveRawData(self.processors, n_tot, self.folderPath)

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
        fillStartIdx = np.max(self.config["BLOCKSIZE"]) + np.max(
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
            zip(self.processors, self.config["BLOCKSIZE"])
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

    def setupIR(self):
        """reverbExpeptions is a tuple of tuples, where each inner tuple is
            formatted as (sourceName, micName, REVERB-PARAMETER), where the options
            for reverb-parameter are the same as for config['REVERB']"""
        print("Computing Room IR...")
        metadata = {}
        propagationFilters = {}

        for src, mic in self.arrays.paths():
            reverb = self.config["REVERB"]
            if mic.name in self.propagationPaths[src.name]:
                if isinstance(self.propagationPaths[src.name][mic.name], np.ndarray):
                    assert self.propagationPaths[src.name][mic.name].shape[0] == self.arrays[src.name].num
                    assert self.propagationPaths[src.name][mic.name].shape[1] == self.arrays[mic.name].num
                    assert self.propagationPaths[src.name][mic.name].ndim == 3
                    continue
                elif isinstance(self.propagationPaths[src.name][mic.name], str):
                    reverb = self.propagationPaths[src.name][mic.name]

            #reverb = self.config["REVERB"]
            # for (sName, mName, chosenReverb) in reverbExceptions:
            #     if src.name==sName and mic.name==mName:
            #         reverb = chosenReverb
            print(f"{src.name}->{mic.name} has propagation type: {reverb}")

            if reverb == "none": 
                self.propagationPaths[src.name][mic.name] = np.zeros((src.num, mic.num, 1))

            elif reverb == "identity":
                self.propagationPaths[src.name][mic.name] = np.ones((src.num,mic.num, 1))
            
            elif reverb == "ism":
                if self.config["SPATIALDIMS"] == 3:
                    self.propagationPaths[src.name][mic.name], metadata[src.name+"->"+mic.name+" ISM"] = rir.irRoomImageSource3d(
                                                        src.pos, mic.pos, self.config["ROOMSIZE"], self.config["ROOMCENTER"], 
                                                        self.config["MAXROOMIRLENGTH"], self.config["RT60"], self.config["SAMPLERATE"],
                                                        calculateMetadata=True)
                else:
                    raise ValueError
            elif reverb == "freespace":
                if self.config["SPATIALDIMS"] == 3:
                    self.propagationPaths[src.name][mic.name] = rir.irPointSource3d(
                    src.pos, mic.pos, self.config["SAMPLERATE"], self.config["C"])
                elif self.config["SPATIALDIMS"] == 2:
                    self.propagationPaths[src.name][mic.name] = rir.irPointSource2d(
                        src.pos, mic.pos, self.config["SAMPLERATE"], self.config["C"]
                    )
                else:
                    raise ValueError

        # for srcName, irSet in propagationFilters.items():
        #     for micName, ir in irSet.items():
        #         propagationFilters[srcName][micName] = FilterSum_IntBuffer(ir=ir)

        return metadata

    def usePresetPositions(self, config, presetName):
        print("Setup Positions")
        if config["SPATIALDIMS"] == 3:
            if presetName == "circle":
                if config["SPATIALDIMS"] == 3:
                    posMic, posSrc = setup.getPositionsCylinder3d(config)
                elif config["SPATIALDIMS"] == 2:
                    posMic, posSrc  = setup.getPositionsDisc2d(config)
            elif presetName == "cuboid":
                    posMic, posSrc = setup.getPositionsCuboid3d(config)
            elif presetName == "rectangle":
                    posMic, posSrc = setup.getPositionsRectangle3d(config)
            elif presetName == "doublerectangle":
                    posMic, posSrc = setup.getPositionsDoubleRectangle3d(config)
        elif config["SPATIALDIMS"] == 1:
            posMic, posSrc = setup.getPositionsLine1d(config)
        else:
            raise ValueError

        for arrayName, arrayPos in posMic.items():
            self.addMics(arrayName, arrayPos)
        return pos


    def createFigFolder(self, folderForPlots, generateSubFolder=True, safeNaming=False):
        if self.config["PLOTOUTPUT"] != "none":
            if generateSubFolder:
                folderName = util.getUniqueFolderName("figs_", folderForPlots, safeNaming)
                folderName.mkdir()
            else:
                folderName = folderForPlots
                if not folderName.exists():
                    folderName.mkdir()
        else:
            folderName = ""
        return folderName


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


