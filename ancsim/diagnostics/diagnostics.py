import numpy as np
from functools import partial
import itertools as it
import operator as op
import copy

import ancsim.array as ar
import ancsim.utilities as util
import ancsim.experiment.plotscripts as psc
import ancsim.signal.filterclasses as fc
import ancsim.diagnostics.diagnosticplots as dplot
import ancsim.diagnostics.diagnosticsummary as dsum


import ancsim.diagnostics.core as diacore


class RecordFilter(diacore.InstantDiagnostic):
    """
        Remember to include .ir in the property name 
        if the property is a filter object
    """
    def __init__ (
        self, 
        prop_name, 
        sim_info, 
        block_size, 
        save_at = None, 
        export_func = "plot",
        keep_only_last_export = False,
        ):
        super().__init__(sim_info, block_size, save_at, export_func, keep_only_last_export)
        self.prop_name = prop_name
        self.get_prop = op.attrgetter(prop_name)
        self.prop = None

    def save(self, processor, chunkInterval, globInterval):
        self.prop = copy.deepcopy(self.get_prop(processor))

    def get_output(self):
        return self.prop


class RecordSignal(diacore.SignalDiagnostic):
    def __init__(self, 
        sig_name, 
        sim_info,
        block_size,
        export_at=None,
        save_at = None, 
        export_func="plot"):
        super().__init__(sim_info, block_size, export_at, save_at, export_func)
        self.sig_name = sig_name
        self.signal = np.full((sim_info.tot_samples), np.nan)
        
    def save(self, processor, chunkInterval, globInterval):
        if processor.sig[self.sig_name].shape[0] > 1:
            raise NotImplementedError

        self.signal[globInterval[0]:globInterval[1]] = \
            processor.sig[self.sig_name][0,chunkInterval[0]:chunkInterval[1]]

    def get_output(self):
        return self.signal



class SignalPower(diacore.SignalDiagnostic):
    def __init__(self, sig_name,
                        sim_info, 
                        block_size, 
                        export_at=None):
        super().__init__(sim_info, block_size, export_at)
        self.sig_name = sig_name
        self.power = np.full((sim_info.tot_samples), np.nan)
    
    def save(self, processor, chunkInterval, globInterval):
        self.power[globInterval[0]:globInterval[1]] = np.mean(np.abs(
            processor.sig[sig_name][:,chunkInterval[0]:chunkInterval[1]])**2, axis=0)

    def get_output(self):
        return self.power
