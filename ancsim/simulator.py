import numpy as np

import ancsim.utilities as util
import ancsim.configutil as configutil

from ancsim.array import ArrayCollection, Array
from ancsim.adaptivefilter.base import ProcessorWrapper
from ancsim.simulatorlogging import addToSimMetadata, writeFilterMetadata

import ancsim.saveloadsession as sess
import ancsim.experiment.plotscripts as psc
import ancsim.experiment.multiexperimentutils as meu
import ancsim.soundfield.presets as preset
import ancsim.soundfield.roomimpulseresponse as rir
import ancsim.adaptivefilter.diagnostics as diag


class SimulatorSetup:
    def __init__():
        pass




class Simulator:
    def __init__(
        self,
        baseFolderPath,
        sessionFolder=None,
    ):
        self.processors = []
        self.arrays = ArrayCollection()

        #self.freeSrcProp = FreeSourcePropagator()

        self.setConfig(configutil.getDefaultConfig())

        self.baseFolderPath = baseFolderPath
        self.sessionFolder = sessionFolder

        self.rng = np.random.default_rng(1)

    def setConfig(self, config):
        configutil.checkConfig(config)
        self.config = config

    def addArrays(self, arrayCollection):
        for ar in arrayCollection:
            self.arrays.add_array(ar)
            #if ar.is_source:
            #    self.arrays.paths[ar.name] = {}

    def addArray(self, array):
        self.arrays.add_array(array)

    def addFreeSource(self, name, pos):
        arr = Array(name, ArrayType.FREESOURCE, pos)
        self.addArray(arr)

    def addControllableSource(self, name, pos):
        arr = Array(name, ArrayType.CTRLSOURCE, pos)
        self.addArray(arr)
    
    def addMics(self, name, pos):
        arr = Array(name, ArrayType.MIC, pos)
        self.addArray(arr)

    def setPath(self, srcName, micName, path):
        self.arrays.set_prop_path(path, srcName, micName)

    def setSource(self, name, source):
        self.arrays[name].setSource(source)

    def addProcessor(self, processor):
        try:
            self.processors.extend(processor)
        except TypeError:
            self.processors.append(processor)
        
    def loadFromPath(self, sessionPath):
        self.folderPath = self.createFigFolder(self.baseFolderPath)
        loadedConfig, loadedArrays = sess.loadFromPath(sessionPath, self.folderPath)
        self.setConfig(loadedConfig)
        self.arrays = loadedArrays
        #self.freeSrcProp.prepare(self.config, self.arrays)

    # def loadSession(self, sessionPath=None, config=None):
    #     if config is not None:
    #         return sess.loadSession(sessionPath, self.folderPath, config)
    #     else:
    #         return sess.loadSession(sessionPath, self.folderPath)

    def usePreset(self, presetName, **kwargs):
        presetFunctions = {
            "cuboid" : preset.getPositionsCuboid3d,
            "cylinder" : preset.getPositionsCylinder3d,
            "audio_processing" : preset.audioProcessing,
            "signal_estimation" : preset.signalEstimation,
            "anc_mpc" : preset.ancMultiPoint,
        }

        arrays, chosenPropPaths = presetFunctions[presetName](self.config, **kwargs)
        self.addArrays(arrays)
        self.arrays.set_path_types(chosenPropPaths)
    
    # def prepareNewSession(self):
    #     """currently not in use"""
    #     irMetadata = self.setupIR(ArrayType.FREESOURCE, self.reverbExceptions)
    #     #self.propFiltersCtrl, irMetadataCtrl = self.setupIR(ArrayType.CTRLSOURCE, self.reverbExceptions)
    #     addToSimMetadata(self.folderPath, irMetadata)

    #     if self.config["auto_save_load"]:
    #         sess.saveSession(self.arrays, self.propFiltersFree, self.propFiltersCtrl)

    # def createNewSession(self):
    #     irMetadata = self.setupIR()

    #     if self.config["auto_save_load"] and self.sessionFolder is not None:
    #         sess.saveSession(self.sessionFolder, self.config, self.arrays, simMetadata=irMetadata)
            
            

    def prepare(self):
        assert not self.arrays.empty()
        self.sim_info = configutil.SimulatorInfo(self.config)
        self.arrays.set_default_path_type(self.sim_info.reverb)
        
        self.folderPath = self.createFigFolder(self.baseFolderPath)
        print(f"Figure folder: {self.folderPath}")

        if self.sim_info.auto_save_load and self.sessionFolder is not None:
            try:
                self.arrays = sess.loadSession(self.sessionFolder, self.folderPath, self.config, self.arrays)
            except sess.MatchingSessionNotFoundError:
                print("No matching session found")
                irMetadata = self.setupIR()
                sess.saveSession(self.sessionFolder, self.config, self.arrays, simMetadata=irMetadata)
                sess.addToSimMetadata(self.folderPath, irMetadata)
        else:
            self.setupIR()

        # LOGGING AND DIAGNOSTICS
        sess.saveConfig(self.folderPath, self.config)
        self.arrays.plot(self.folderPath, self.sim_info.plot_output)
        

    def _setupSimulation(self):

        setUniqueFilterNames(self.processors)
        writeFilterMetadata(self.processors, self.folderPath)

        self.plotDispatcher = diag.PlotDiagnosticDispatcher(
            self.processors, self.sim_info.plot_output
        )

        self.processors = [ProcessorWrapper(pr, self.arrays) for pr in self.processors]

        for filt in self.processors:
            print("preparing")
            filt.prepare()

    def runSimulation(self):
        self._setupSimulation()

        blockSizes = [proc.blockSize for proc in self.processors]
        maxBlockSize = np.max(blockSizes)

        print("SIM START")
        n_tot = 1
        bufferIdx = -1
        #noises = self._updateNoises(n_tot, noises)
        #noiseIndices = [self.config["sim_buffer"] for _ in range(len(self.processors))]
        while n_tot < self.sim_info.tot_samples - maxBlockSize:
            bufferIdx += 1
            for n in range(
                self.sim_info.sim_buffer,
                self.sim_info.sim_buffer
                + min(self.sim_info.sim_chunk_size, self.sim_info.tot_samples - n_tot),
            ):
                # Write progress
                if n_tot % 1000 == 0:
                    print("Timestep: ", n_tot)#, end="\r")

                #Break if simulation is finished
                if self.sim_info.tot_samples - maxBlockSize < n_tot:
                    break

                #Generate and propagate noise from free sources
                # if n_tot % np.max(self.config["BLOCKSIZE"]) == 0:
                #     self.freeSrcProp.process(np.max(self.config["BLOCKSIZE"]))

                #Generate and propagate noise from controllable sources
                for i, proc in enumerate(self.processors):
                    if n_tot % proc.blockSize == 0:
                        #noise = {mic.name : self.freeSrcProp.sig[mic.name] for mic in self.arrays.mics()}
                        proc.propagate(n_tot)
                        proc.process(n_tot)
                        #print("n_tot", n_tot)
                        #print("proc idx", proc.processor.idx)
                        #noiseIndices[i] += proc.blockSize

                # Generate plots and diagnostics
                if n_tot % self.sim_info.sim_chunk_size == 0 and n_tot > 0:
                    if self.sim_info.plot_output != "none":
                        self.plotDispatcher.dispatch(
                            [p.processor for p in self.processors], n_tot, bufferIdx, self.folderPath
                        )
                    if self.sim_info.save_raw_data and (
                        bufferIdx % self.sim_info.save_raw_data_freq == 0
                        or bufferIdx - 1 == 0
                    ):
                        sess.saveRawData(self.processors, n_tot, self.folderPath)

                n_tot += 1
        print(n_tot)


    # def _updateNoises(self):
    #     for src in self.arrays.of_type(ArrayType.FREESOURCE):
    #         for mic in self.arrays.mics():
    #             self.freeSrcSig[src.name][mic.name] = {
    #                 = src.getSamples(self.config["sim_chunk_size"])
    #             }
             

    # def _updateNoises(self, timeIdx, noises):
    #     noise = self.freeSource.getSamples(self.config["sim_chunk_size"])
    #     noises = {
    #         filtName: np.concatenate(
    #             (noises[filtName][:, -self.config["sim_buffer"] :], sf.process(noise)),
    #             axis=-1,
    #         )
    #         for filtName, sf in self.sourceFilters.items()
    #     }
    #     return noises

    # def _fillBuffers(self):
    #     noise = self.noiseSource.getSamples(self.config["sim_buffer"])
    #     noises = {
    #         filtName: sf.process(noise) for filtName, sf in self.sourceFilters.items()
    #     }

    #     maxSecPathLength = np.max(
    #         [speakerFilt.shape[-1] for _, speakerFilt, in self.speakerFilters.items()]
    #     )
    #     fillStartIdx = np.max(self.config["BLOCKSIZE"]) + np.max(
    #         (self.config["FILTLENGTH"] + self.config["KERNFILTLEN"], maxSecPathLength)
    #     )
    #     fillNumBlocks = [
    #         (self.config["sim_buffer"] - fillStartIdx) // bs
    #         for bs in self.config["BLOCKSIZE"]
    #     ]

    #     startIdxs = []
    #     for filtIdx, blockSize in enumerate(self.config["BLOCKSIZE"]):
    #         startIdxs.append([])
    #         for i in range(fillNumBlocks[filtIdx]):
    #             startIdxs[filtIdx].append(
    #                 self.config["sim_buffer"]
    #                 - ((fillNumBlocks[filtIdx] - i) * blockSize)
    #             )

    #     for filtIdx, (filt, blockSize) in enumerate(
    #         zip(self.processors, self.config["BLOCKSIZE"])
    #     ):
    #         filt.idx = startIdxs[filtIdx][0]
    #         for i in startIdxs[filtIdx]:
    #             filt.forwardPass(
    #                 blockSize,
    #                 {
    #                     pointName: noiseAtPoints[:, i : i + blockSize]
    #                     for pointName, noiseAtPoints in noises.items()
    #                 },
    #             )

    #     return noises

    def setupIR(self):
        """reverbExpeptions is a tuple of tuples, where each inner tuple is
            formatted as (sourceName, micName, reverb-PARAMETER), where the options
            for reverb-parameter are the same as for config['reverb']"""
        print("Computing Room IR...")
        metadata = {}

        for src, mic in self.arrays.mic_src_combos():
            reverb = self.arrays.path_type[src.name][mic.name]
            #if mic.name in self.arrays.paths[src.name]:
                #if isinstance(self.arrays.paths[src.name][mic.name], np.ndarray):
                #    continue
                #elif isinstance(self.arrays.paths[src.name][mic.name], str):
                #    reverb = self.arrays.paths[src.name][mic.name]

            print(f"{src.name}->{mic.name} has propagation type: {reverb}")

            if reverb == "none": 
                self.arrays.paths[src.name][mic.name] = np.zeros((src.num, mic.num, 1))
            elif reverb == "identity":
                self.arrays.paths[src.name][mic.name] = np.ones((src.num,mic.num, 1))
            elif reverb == "isolated":
                assert src.num == mic.num
                self.arrays.paths[src.name][mic.name] = np.eye(src.num, mic.num)[...,None]
            elif reverb == "random":
                self.arrays.paths[src.name][mic.name] = self.rng.normal(0, 1, size=(src.num, mic.num, self.config["max_room_ir_length"]))
            elif reverb == "ism":
                if self.config["spatial_dims"] == 3:
                    self.arrays.paths[src.name][mic.name], metadata[src.name+"->"+mic.name+" ISM"] = rir.irRoomImageSource3d(
                                                        src.pos, mic.pos, self.config["room_size"], self.config["room_center"], 
                                                        self.config["max_room_ir_length"], self.config["rt60"], self.config["samplerate"],
                                                        calculateMetadata=True)
                else:
                    raise ValueError
            elif reverb == "freespace":
                if self.config["spatial_dims"] == 3:
                    self.arrays.paths[src.name][mic.name] = rir.irPointSource3d(
                    src.pos, mic.pos, self.config["samplerate"], self.config["c"])
                elif self.config["spatial_dims"] == 2:
                    self.arrays.paths[src.name][mic.name] = rir.irPointSource2d(
                        src.pos, mic.pos, self.config["samplerate"], self.config["c"]
                    )
                else:
                    raise ValueError
            elif reverb == "modified":
                pass
            else:
                raise ValueError

        # for srcName, irSet in propagationFilters.items():
        #     for micName, ir in irSet.items():
        #         propagationFilters[srcName][micName] = FilterSum_IntBuffer(ir=ir)

        return metadata

    # def usePresetPositions(self, config, presetName):
    #     print("Setup Positions")
    #     if config["spatial_dims"] == 3:
    #         if presetName == "circle":
    #             if config["spatial_dims"] == 3:
    #                 posMic, posSrc = setup.getPositionsCylinder3d(config)
    #             elif config["spatial_dims"] == 2:
    #                 posMic, posSrc  = setup.getPositionsDisc2d(config)
    #         elif presetName == "cuboid":
    #                 posMic, posSrc = setup.getPositionsCuboid3d(config)
    #         elif presetName == "rectangle":
    #                 posMic, posSrc = setup.getPositionsRectangle3d(config)
    #         elif presetName == "doublerectangle":
    #                 posMic, posSrc = setup.getPositionsDoubleRectangle3d(config)
    #     elif config["spatial_dims"] == 1:
    #         posMic, posSrc = setup.getPositionsLine1d(config)
    #     else:
    #         raise ValueError

    #     for arrayName, arrayPos in posMic.items():
    #         self.addMics(arrayName, arrayPos)
    #     return pos


    def createFigFolder(self, folderForPlots, generateSubFolder=True, safeNaming=False):
        if self.config["plot_output"] != "none":
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


# def plotAnyPos(pos, folderPath, config):
#     print("Setup Positions")
#     if config["spatial_dims"] == 3:
#         if config["ARRAYSHAPES"] == "circle":
#             psc.plotPos3dDisc(pos, folderPath, config, config["plot_output"])
#         elif config["ARRAYSHAPES"] in ("cuboid", "rectangle", "doublerectangle", "smaller_rectangle"):
#             if config["reverb"] == "ism":
#                 psc.plotPos3dRect(pos, folderPath, config, config["room_size"], config["room_center"], printMethod=config["plot_output"])
#             else:
#                 psc.plotPos3dRect(
#                     pos, folderPath, config, printMethod=config["plot_output"]
#                 )
#         else:
#             psc.plotPos(pos, folderPath,config, printMethod=config["plot_output"])
#     elif config["spatial_dims"] == 2:
#         if config["ARRAYSHAPES"] == "circle":
#             if config["REFDIRECTLYOBTAINED"]:
#                 raise NotImplementedError
#             else:
#                 psc.plotPos2dDisc(pos, folderPath, config, config["plot_output"])
#         else:
#             raise NotImplementedError
#     else:
#         raise ValueError



#class Buffer():
#    def __init__(self):
#        self.data = np.zeros(())



