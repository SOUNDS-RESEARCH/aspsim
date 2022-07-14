import numpy as np
import json
from pathlib import Path
import copy

import ancsim.utilities as util
import ancsim.configutil as configutil

from ancsim.array import ArrayCollection, MicArray, ControllableSourceArray, FreeSourceArray
from ancsim.adaptivefilter.base import ProcessorWrapper, FreeSourceHandler
from ancsim.simulatorlogging import addToSimMetadata, writeFilterMetadata

import ancsim.saveloadsession as sess
import ancsim.experiment.plotscripts as psc
import ancsim.presets as preset
import ancsim.diagnostics.core as diacore


class SimulatorSetup:
    def __init__(
        self,
        baseFolderPath=None,
        sessionFolder=None
        ):
        self.arrays = ArrayCollection()

        self.config = configutil.getDefaultConfig()

        self.baseFolderPath = baseFolderPath
        self.sessionFolder = sessionFolder

        self.rng = np.random.default_rng(1)

    def addArrays(self, arrayCollection):
        for ar in arrayCollection:
            self.arrays.add_array(ar)
        self.arrays.set_path_types(arrayCollection.path_type)
        pass
            #if ar.is_source:
            #    self.arrays.paths[ar.name] = {}

    def addArray(self, array):
        self.arrays.add_array(array)

    def addFreeSource(self, name, pos, source):
        arr = FreeSourceArray(name, pos, source)
        self.addArray(arr)

    def addControllableSource(self, name, pos):
        arr = ControllableSourceArray(name, pos)
        self.addArray(arr)
    
    def addMics(self, name, pos):
        arr = MicArray(name,pos)
        self.addArray(arr)

    def setPath(self, srcName, micName, path):
        self.arrays.set_prop_path(path, srcName, micName)

    def setSource(self, name, source):
        self.arrays[name].setSource(source)

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
            "debug" : preset.debug,
        }

        arrays, chosenPropPaths = presetFunctions[presetName](self.config, **kwargs)
        self.addArrays(arrays)
        self.arrays.set_path_types(chosenPropPaths)
        

    def createSimulator(self):
        assert not self.arrays.empty()
        sim_info = configutil.SimulatorInfo(self.config)
        finished_arrays = copy.deepcopy(self.arrays)
        finished_arrays.set_default_path_type(sim_info.reverb)
        
        folderPath = self.createFigFolder(self.baseFolderPath)
        print(f"Figure folder: {folderPath}")

        if self.config["auto_save_load"] and self.sessionFolder is not None:
            try:
                finished_arrays = sess.loadSession(self.sessionFolder, folderPath, self.config, finished_arrays)
            except sess.MatchingSessionNotFoundError:
                print("No matching session found")
                irMetadata = finished_arrays.setupIR(sim_info)
                sess.saveSession(self.sessionFolder, self.config, finished_arrays, simMetadata=irMetadata)
                #sess.addToSimMetadata(folderPath, irMetadata)     
        else:
            finished_arrays.setupIR(sim_info)

        

        # LOGGING AND DIAGNOSTICS
        sess.saveConfig(folderPath, self.config)
        finished_arrays.plot(folderPath, self.config["plot_output"])
        finished_arrays.save_metadata(folderPath)
        return Simulator(sim_info, finished_arrays, folderPath)

    def createFigFolder(self, folderForPlots, generateSubFolder=True, safeNaming=False):
        if self.config["plot_output"] == "none" or folderForPlots is None:
            return None

        if generateSubFolder:
            folderName = util.getUniqueFolderName("figs_", folderForPlots, safeNaming)
            folderName.mkdir()
        else:
            folderName = folderForPlots
            if not folderName.exists():
                folderName.mkdir()
        return folderName



class Simulator:
    def __init__(
        self,
        sim_info,
        arrays,
        folderPath,
    ):
        
        self.sim_info = sim_info
        self.arrays = arrays
        self.folderPath = folderPath
        
        self.processors = []
        self.rng = np.random.default_rng(1)

    def addProcessor(self, processor):
        try:
            self.processors.extend(processor)
        except TypeError:
            self.processors.append(processor)
            
    def _setupSimulation(self):
        setUniqueFilterNames(self.processors)
        writeFilterMetadata(self.processors, self.folderPath)

        self.plot_exporter = diacore.DiagnosticExporter(
            self.sim_info, self.processors
        )

        self.free_src_handler = FreeSourceHandler(self.sim_info, self.arrays)
        self.processors = [ProcessorWrapper(pr, self.arrays) for pr in self.processors]
        
        self.free_src_handler.prepare()
        for proc in self.processors:
            self.free_src_handler.copy_sig_to_proc(proc, 0, self.sim_info.sim_buffer)
            proc.prepare()
            

    def runSimulation(self):
        self._setupSimulation()

        assert len(self.processors) > 0
        blockSizes = [proc.blockSize for proc in self.processors]
        maxBlockSize = np.max(blockSizes)

        print("SIM START")
        self.n_tot = 1
        bufferIdx = 0
        while self.n_tot < self.sim_info.tot_samples+maxBlockSize:
            for n in range(
                self.sim_info.sim_buffer,
                self.sim_info.sim_buffer
                + min(self.sim_info.sim_chunk_size, self.sim_info.tot_samples + maxBlockSize - self.n_tot),
            ):

                #Generate and propagate noise from controllable sources
                for i, proc in enumerate(self.processors):
                    if self.n_tot % proc.blockSize == 0:
                        proc.process(self.n_tot)
                        self.free_src_handler.copy_sig_to_proc(proc, self.n_tot, proc.blockSize)
                        proc.propagate(self.n_tot)
 
                self.arrays.update(self.n_tot)

                # Generate plots and diagnostics
                #if self.plot_exporter.ready_for_export([p.processor for p in self.processors]):
                if self.sim_info.plot_output != "none":
                    self.plot_exporter.dispatch(
                        [p.processor for p in self.processors], self.n_tot, self.folderPath
                    )

                # Write progress
                if self.n_tot % 1000 == 0:
                    print("Timestep: ", self.n_tot)#, end="\r")

                self.n_tot += 1
            bufferIdx += 1
        if self.sim_info.plot_output != "none":
            self.plot_exporter.dispatch(
                [p.processor for p in self.processors], self.n_tot, self.folderPath
            )

        print(self.n_tot)


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



