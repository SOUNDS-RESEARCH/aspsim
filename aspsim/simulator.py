import numpy as np
import copy

import aspsim.fileutilities as futil
import aspsim.configutil as configutil
import aspsim.array as ar
import aspsim.saveloadsession as sess
import aspsim.diagnostics.core as diacore
import aspsim.counter as counter

import aspcore.filterclasses as fc


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

        self.arrays = ar.ArrayCollection()
        #self.diag = diacore.DiagnosticHandler(self.sim_info) 

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
        return Simulator(self.sim_info, finished_arrays, folder_path, rng=self.rng)

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
        self.diag = diacore.Logger(self.sim_info)
        #self.diag = diacore.DiagnosticHandler(self.sim_info) 
        
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

    def _prepare_simulation(self):
        if len(self.processors) > 1:
            raise NotImplementedError
        
        set_unique_processor_names(self.processors)
        sess.write_processor_metadata(self.processors, self.folder_path)

        
        #self.plot_exporter = diacore.DiagnosticExporter(
        #    self.sim_info, self.diag
        #)

        self.sig = Signals(self.sim_info, self.arrays)
        self.propagator = Propagator(self.sim_info, self.arrays, self.sig)

        self.diag.prepare()
        self.propagator.prepare()
        for proc in self.processors:
            proc.sig = self.sig
            proc.prepare()
            

    def run_simulation(self):
        self._prepare_simulation()

        # the else value (which happens when there is no processor) can be changed to just about anything. Could be set to 1, or 
        # set to a high value that can be used to speed up. 
        max_block_size = np.max([proc.block_size for proc in self.processors]) if self.processors else self.sim_info.sim_buffer // 4  

        print("SIM START")
        self.n_tot = 0
        while self.n_tot < self.sim_info.tot_samples+max_block_size:
            self.arrays.update(self.n_tot)

            for proc in self.processors:
                if self.n_tot % proc.block_size == proc.block_size-1:
                    proc.process(proc.block_size)
            self.propagator.propagate(1)
            self.diag_moved_from_propagate(max_block_size)

            self.diag.dispatch(self.folder_path)

            # Write progress
            if self.n_tot % 1000 == 0:
                print(f"Timestep: {self.n_tot}")#, end="\r")

            self.sig.idx += 1
            self.n_tot += 1
        self.diag.dispatch(self.folder_path)

        print(self.n_tot)

    def diag_moved_from_propagate(self, max_block_size):
        last_block = self.last_block_on_buffer(max_block_size)
        self.diag.save_data(self.processors, self.sig, self.sig.idx, self.n_tot, last_block)
        if last_block:
            self.sig._reset_signals()

    def last_block_on_buffer(self, max_block_size):
        return self.sig.idx+max_block_size >= self.sim_info.sim_chunk_size+self.sim_info.sim_buffer

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





class Signals():
    """
    A simple wrapper around a dictionary to hold all signals
    """
    def __init__(self, sim_info, arrays):
        self.sim_info = sim_info
        self.signals = {}
        self.idx = 0
        self.idx_tot = 0

        for array in arrays:
            self.create_signal(array.name, array.num)
        if self.sim_info.save_source_contributions:
            for src, mic in arrays.mic_src_combos():
                self.create_signal(src.name+"~"+mic.name, mic.num)
      
    def __getitem__(self, key):
        return self.signals[key]

    def __contains__(self, key):
        return key in self.signals

    def items(self):
        yield self.signals.items()
    
    def values(self):
        yield self.signals.values()

    def keys(self):
        yield self.signals.keys()

    def create_signal(self, name, dim):
        """
        Inserts a new signal of shape (*dim, simbuffer+simchunksize)

        Parameters
        ----------
        name : str 
            name of the signal, in the same way a dictionary entry has a name
            signal is accessed as signals[name]
        dim : int, list or tuple
            dimensionality of the signal. signal will have the shape (*dim, simbuffer+simchunksize)
            where the last axis represents time
        """
        if isinstance(dim, int):
            dim = (dim,)
        assert name not in self.signals
        self.signals[name] = np.zeros(dim + (self.sim_info.sim_buffer + self.sim_info.sim_chunk_size,))

    def _reset_signals(self):
        for name, sig in self.signals.items():
            self.signals[name] = np.concatenate(
                (
                    sig[..., -self.sim_info.sim_buffer :],
                    np.zeros(sig.shape[:-1] + (self.sim_info.sim_chunk_size,)),
                ),
                axis=-1,
            )
        self.idx -= self.sim_info.sim_chunk_size


class Propagator():
    def __init__(self, sim_info, arrays, sig):
        self.sim_info = sim_info
        self.arrays = arrays
        self.sig = sig

        self.path_filters = {}
        for src, mic, path in arrays.iter_paths():
            if src.name not in self.path_filters:
                self.path_filters[src.name] = {}
            self.path_filters[src.name][mic.name] = fc.create_filter(ir=path, sum_over_input=True, dynamic=(src.dynamic or mic.dynamic))

    def prepare(self):
        # Does not take movement into account. Currently just propagates from the stationary RIRs
        # associated with the initial position

        for src in self.arrays.free_sources():
            self.sig[src.name][..., :self.sim_info.sim_buffer] = src.get_samples(self.sim_info.sim_buffer)

        for src, mic in self.arrays.mic_src_combos():
            propagated_signal = self.path_filters[src.name][mic.name].process(
                    self.sig[src.name][:,:self.sim_info.sim_buffer])
            self.sig[mic.name][:,:self.sim_info.sim_buffer] += propagated_signal
            if self.sim_info.save_source_contributions:
                self.sig[f"{src.name}~{mic.name}"][:,:self.sim_info.sim_buffer] = propagated_signal
        self.sig.idx = self.sim_info.sim_buffer


    def propagate(self, num_samples):
        """ propagates audio from all sources.
            The mic_signals are calculated for the indices
            self.sig.idx to self.sig.idx+num_samples"""

        i = self.sig.idx

        for src in self.arrays.free_sources():
            self.sig[src.name][..., i:i+num_samples] = src.get_samples(num_samples)

        for src, mic in self.arrays.mic_src_combos():
            if src.dynamic or mic.dynamic:
                self.path_filters[src.name][mic.name].ir = self.arrays.paths[src.name][mic.name]
        
        for src, mic in self.arrays.mic_src_combos():
            propagated_signal = self.path_filters[src.name][mic.name].process(
                    self.sig[src.name][...,i:i+num_samples])
            self.sig[mic.name][...,i:i+num_samples] += propagated_signal
            if self.sim_info.save_source_contributions:
                self.sig[src.name+"~"+mic.name][...,i:i+num_samples] = propagated_signal
        #self.sig.idx += num_samples