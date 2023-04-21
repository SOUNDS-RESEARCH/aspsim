import numpy as np
import copy

import aspsim.fileutilities as futil
import aspsim.configutil as configutil

import aspsim.array as ar
from aspsim.processor import ProcessorWrapper, FreeSourceHandler

import aspsim.saveloadsession as sess
import aspsim.presets as preset
import aspsim.diagnostics.core as diacore


class SimulatorSetup:
    def __init__(
        self,
        base_fig_path=None,
        session_folder=None,
        config_path = None,
        rng = None, 
        ):
        """
        Parameters
        ----------
        base_fig_path : str or Path from pathlib
            If this is supplied, the simulator will create a new
            subfolder in that directory and fill it with plots and metadata
        session_folder : str or Path from pathlib
            If the option auto_save_load in the config is True, the simulator will 
            look in the session_folder for sessions to load, and will save the current
            session there. 
        config_path : str or Path from pathlib
            Supply if you want to load the config parameters from a (yaml) file. 
            Otherwise the default will be loaded, which can be changed inside your
            Python code. 
        rng : numpy Generator object
            recommended to obtain your rng object from np.random.default_rng(seed)
        
        """
        self.arrays = ar.ArrayCollection()

        if config_path is None:
            self.sim_info = configutil.load_default_config()
        else:
            self.sim_info = configutil.load_from_file(config_path)

        self.base_fig_path = base_fig_path
        self.session_folder = session_folder

        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

    def add_arrays(self, array_collection):
        for ar in array_collection:
            self.arrays.add_array(ar)
        self.arrays.set_path_types(array_collection.path_type)

            #if ar.is_source:
            #    self.arrays.paths[ar.name] = {}

    def add_array(self, array):
        self.arrays.add_array(array)

    def add_free_source(self, name, pos, source):
        arr = ar.FreeSourceArray(name, pos, source)
        self.add_array(arr)

    def add_controllable_source(self, name, pos):
        arr = ar.ControllableSourceArray(name, pos)
        self.add_array(arr)
    
    def add_mics(self, name, pos):
        arr = ar.MicArray(name,pos)
        self.add_array(arr)

    def set_path(self, src_name, mic_name, path):
        self.arrays.set_prop_path(path, src_name, mic_name)

    def set_source(self, name, source):
        self.arrays[name].set_source(source)

    def load_from_path(self, session_path):
        self.folder_path = self._create_fig_folder(self.base_fig_path)
        self.sim_info, self.arrays = sess.load_from_path(session_path, self.folder_path)
        #self.freeSrcProp.prepare(self.config, self.arrays)
        #self.setConfig(loaded_config)

    def use_preset(self, preset_name, **kwargs):
        preset_functions = {
            "audio_processing" : preset.audio_processing,
            "signal_estimation" : preset.signal_estimation,
            "anc_mpc" : preset.anc_multi_point,
            "debug" : preset.debug,
        }

        arrays, chosen_prop_paths = preset_functions[preset_name](self.sim_info, **kwargs)
        self.add_arrays(arrays)
        self.arrays.set_path_types(chosen_prop_paths)
        

    def create_simulator(self):
        assert not self.arrays.empty()
        finished_arrays = copy.deepcopy(self.arrays)
        finished_arrays.set_default_path_type(self.sim_info.reverb)
        
        folder_path = self._create_fig_folder(self.base_fig_path)
        print(f"Figure folder: {folder_path}")

        if self.sim_info.auto_save_load and self.session_folder is not None:
            try:
                finished_arrays = sess.load_session(self.session_folder, folder_path, self.sim_info, finished_arrays)
            except sess.MatchingSessionNotFoundError:
                print("No matching session found")
                irMetadata = finished_arrays.setup_ir(self.sim_info)
                sess.save_session(self.session_folder, self.sim_info, finished_arrays, sim_metadata=irMetadata)   
        else:
            finished_arrays.setup_ir(self.sim_info)

        # LOGGING AND DIAGNOSTICS
        self.sim_info.save_to_file(folder_path)
        finished_arrays.plot(self.sim_info, folder_path, self.sim_info.plot_output)
        finished_arrays.save_metadata(folder_path)
        return Simulator(self.sim_info, finished_arrays, folder_path)

    def _create_fig_folder(self, folder_for_plots, gen_subfolder=True, safe_naming=False):
        if self.sim_info.plot_output == "none" or folder_for_plots is None:
            return None

        if gen_subfolder:
            folder_name = futil.get_unique_folder_name("figs_", folder_for_plots, safe_naming)
            folder_name.mkdir()
        else:
            folder_name = folder_for_plots
            if not folder_name.exists():
                folder_name.mkdir()
        return folder_name



class Simulator:
    def __init__(
        self,
        sim_info,
        arrays,
        folder_path,
        rng = None, 
    ):
        
        self.sim_info = sim_info
        self.arrays = arrays
        self.folder_path = folder_path
        
        self.processors = []

        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

    def add_processor(self, processor):
        try:
            self.processors.extend(processor)
        except TypeError:
            self.processors.append(processor)
            
    def _setup_simulation(self):
        set_unique_processor_names(self.processors)
        sess.write_processor_metadata(self.processors, self.folder_path)

        self.plot_exporter = diacore.DiagnosticExporter(
            self.sim_info, self.processors
        )

        self.free_src_handler = FreeSourceHandler(self.sim_info, self.arrays)
        self.processors = [ProcessorWrapper(pr, self.arrays) for pr in self.processors]
        
        self.free_src_handler.prepare()
        for proc in self.processors:
            self.free_src_handler.copy_sig_to_proc(proc, 0, self.sim_info.sim_buffer)
            proc.prepare()
            

    def run_simulation(self):
        self._setup_simulation()

        assert len(self.processors) > 0
        block_sizes = [proc.block_size for proc in self.processors]
        max_block_size = np.max(block_sizes)

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
                    if self.n_tot % proc.block_size == 0:
                        proc.process(self.n_tot)
                        self.free_src_handler.copy_sig_to_proc(proc, self.n_tot, proc.block_size)
                        proc.propagate(self.n_tot)
 
                self.arrays.update(self.n_tot)

                # Generate plots and diagnostics
                #if self.plot_exporter.ready_for_export([p.processor for p in self.processors]):
                if self.sim_info.plot_output != "none":
                    self.plot_exporter.dispatch(
                        [p.processor for p in self.processors], self.n_tot, self.folder_path
                    )

                # Write progress
                if self.n_tot % 1000 == 0:
                    print(f"Timestep: {self.n_tot}")#, end="\r")

                self.n_tot += 1
            buffer_idx += 1
        if self.sim_info.plot_output != "none":
            self.plot_exporter.dispatch(
                [p.processor for p in self.processors], self.n_tot, self.folder_path
            )

        print(self.n_tot)



def set_unique_processor_names(processors):
    names = []
    for proc in processors:
        new_name = proc.name
        i = 1
        while new_name in names:
            new_name = f"{proc.name} {i}"
            i += 1
        names.append(new_name)
        proc.name = new_name



