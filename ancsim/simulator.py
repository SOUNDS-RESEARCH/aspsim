import numpy as np
import json
from pathlib import Path
import copy

import ancsim.utilities as util
import ancsim.configutil as configutil

from ancsim.array import ArrayCollection, MicArray, ControllableSourceArray, FreeSourceArray
from ancsim.processor import ProcessorWrapper, FreeSourceHandler

import ancsim.saveloadsession as sess
import ancsim.diagnostics.plotscripts as psc
import ancsim.presets as preset
import ancsim.diagnostics.core as diacore


class SimulatorSetup:
    def __init__(
        self,
        base_folder_path=None,
        session_folder=None,
        config = None,
        ):
        self.arrays = ArrayCollection()

        if config is not None:
            self.config = config
        else:
            self.config = configutil.get_default_config()

        self.baseFolderPath = base_folder_path
        self.sessionFolder = session_folder

        self.rng = np.random.default_rng(1)

    def add_arrays(self, array_collection):
        for ar in array_collection:
            self.arrays.add_array(ar)
            #if ar.is_source:
            #    self.arrays.paths[ar.name] = {}

    def add_array(self, array):
        self.arrays.add_array(array)

    def add_free_source(self, name, pos, source):
        arr = FreeSourceArray(name, pos, source)
        self.add_array(arr)

    def add_controllable_source(self, name, pos):
        arr = ControllableSourceArray(name, pos)
        self.add_array(arr)
    
    def add_mics(self, name, pos):
        arr = MicArray(name,pos)
        self.add_array(arr)

    def set_path(self, srcName, micName, path):
        self.arrays.set_prop_path(path, srcName, micName)

    def set_source(self, name, source):
        self.arrays[name].setSource(source)

    def load_from_path(self, sessionPath):
        self.folderPath = self._create_fig_folder(self.baseFolderPath)
        loaded_config, loaded_arrays = sess.loadFromPath(sessionPath, self.folderPath)
        self.setConfig(loaded_config)
        self.arrays = loaded_arrays
        #self.freeSrcProp.prepare(self.config, self.arrays)

    # def loadSession(self, sessionPath=None, config=None):
    #     if config is not None:
    #         return sess.loadSession(sessionPath, self.folderPath, config)
    #     else:
    #         return sess.loadSession(sessionPath, self.folderPath)

    def use_preset(self, preset_name, **kwargs):
        preset_functions = {
            "cuboid" : preset.getPositionsCuboid3d,
            "cylinder" : preset.getPositionsCylinder3d,
            "audio_processing" : preset.audioProcessing,
            "signal_estimation" : preset.signalEstimation,
            "anc_mpc" : preset.ancMultiPoint,
            "debug" : preset.debug,
        }

        arrays, chosen_prop_paths = preset_functions[preset_name](self.config, **kwargs)
        self.add_arrays(arrays)
        self.arrays.set_path_types(chosen_prop_paths)
        

    def create_simulator(self):
        assert not self.arrays.empty()
        sim_info = configutil.SimulatorInfo(self.config)
        finished_arrays = copy.deepcopy(self.arrays)
        finished_arrays.set_default_path_type(sim_info.reverb)
        
        folderPath = self._create_fig_folder(self.baseFolderPath)
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

    def _create_fig_folder(self, folderForPlots, generateSubFolder=True, safeNaming=False):
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
            
    def _setup_simulation(self):
        set_unique_processor_names(self.processors)
        sess.writeFilterMetadata(self.processors, self.folderPath)

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
        self._setup_simulation()

        assert len(self.processors) > 0
        blockSizes = [proc.blockSize for proc in self.processors]
        max_block_size = np.max(blockSizes)

        print("SIM START")
        self.n_tot = 1
        buffer_idx = 0
        while self.n_tot < self.sim_info.tot_samples+max_block_size:
            for n in range(
                self.sim_info.sim_buffer,
                self.sim_info.sim_buffer
                + min(self.sim_info.sim_chunk_size, self.sim_info.tot_samples + max_block_size - self.n_tot),
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
            buffer_idx += 1
        if self.sim_info.plot_output != "none":
            self.plot_exporter.dispatch(
                [p.processor for p in self.processors], self.n_tot, self.folderPath
            )

        print(self.n_tot)



def set_unique_processor_names(filters):
    names = []
    for filt in filters:
        newName = filt.name
        i = 1
        while newName in names:
            newName = filt.name + " " + str(i)
            i += 1
        names.append(newName)
        filt.name = newName



