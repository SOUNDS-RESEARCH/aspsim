import numpy as np
import json
import scipy.signal as spsig

import ancsim.diagnostics.diagnosticplots as dplt
import ancsim.diagnostics.core as diacore

def add_to_summary(diagName, summaryValues, timeIdx, folderPath):
    fullPath = folderPath.joinpath("summary_" + str(timeIdx) + ".json")
    try:
        with open(fullPath, "r") as f:
            summary = json.load(f)
            summary[diagName] = summaryValues
            # totData = {**oldData, **dictToAdd}
    except FileNotFoundError:
        summary = {}
        summary[diagName] = summaryValues
    with open(fullPath, "w") as f:
        json.dump(summary, f, indent=4)


def mean_near_time_idx(outputs, timeIdx):
    summaryValues = {}
    numToAverage = 3000

    for filtName, output in outputs.items():
        # val = diagnostic.getOutput()[timeIdx-numToAverage:timeIdx]
        val = output[...,timeIdx - numToAverage : timeIdx]
        filterArray = np.logical_not(np.isnan(val))
        summaryValues[filtName] = np.mean(val[filterArray])
    return summaryValues


def last_value():
    raise NotImplementedError






class SummaryDiagnostic(diacore.Diagnostic):
    export_functions = {
        "npz" : dplt.savenpz,
        "text" : dplt.txt,
        "spectrum" : dplt.spectrum_plot,
    }
    def __init__ (
        self,
        sim_info,
        block_size,
        save_at,
        export_func = "text",
        keep_only_last_export = False,
        export_kwargs = None,
        preprocess = None,
        ):
        """
        save_at should be a tuple (start_sample, end_sample)
            it will use the samples between start_samle (inclusive) and end_sample (exclusive)
        
        """
        if isinstance(save_at, diacore.IntervalCounter):
            raise NotImplementedError
        else:
            export_at = [save_at[1]]
            save_at = diacore.IntervalCounter(((save_at[0], save_at[1]),))

        super().__init__(sim_info, block_size, export_at, save_at, export_func, keep_only_last_export, export_kwargs, preprocess)






class SignalPowerRatioSummary(SummaryDiagnostic):
    def __init__(self, 
        numerator_name,
        denom_name,
        sim_info, 
        block_size, 
        save_range,
        numerator_channels=slice(None),
        denom_channels = slice(None),
        **kwargs
        ):
        self.save_range = save_range
        self.num_samples = save_range[1] - save_range[0]
        super().__init__(sim_info, block_size, save_range, **kwargs)
        self.numerator_name = numerator_name
        self.denom_name = denom_name
        self.num_power = 0
        self.denom_power = 0

        self.numerator_channels = numerator_channels
        self.denom_channels = denom_channels

        self.plot_data["title"] = f"Ratio of power: {self.numerator_name} / {self.denom_name}. Samples: {self.save_range}"
        
    def save(self, processor, chunkInterval, globInterval):
        self.num_power += np.sum(np.mean(np.abs(processor.sig[self.numerator_name][self.numerator_channels, chunkInterval[0]:chunkInterval[1]])**2, axis=0)) / self.num_samples
        self.denom_power += np.sum(np.mean(np.abs(processor.sig[self.denom_name][self.denom_channels, chunkInterval[0]:chunkInterval[1]])**2,axis=0)) / self.num_samples

        #self.power_ratio[globInterval[0]:globInterval[1]] = num / denom

    def get_output(self):
        return self.num_power / self.denom_power


class SignalPowerSummary(SummaryDiagnostic):
    def __init__(self, 
        sig_name,
        sim_info, 
        block_size, 
        save_range,
        sig_channels=slice(None),
        **kwargs
        ):
        """
        Will output the wrong value if it is exported in the middle of the save_range
        """
        self.save_range = save_range
        self.num_samples = save_range[1] - save_range[0]
        super().__init__(sim_info, block_size, save_range, **kwargs)
        self.sig_name = sig_name
        self.power = 0

        self.sig_channels = sig_channels

        #self.plot_data["title"] = f"Power of {self.sig_name}. Samples: {self.save_range}"
        
    def save(self, processor, chunkInterval, globInterval):
        self.power += np.sum(np.mean(np.abs(processor.sig[self.sig_name][self.sig_channels, chunkInterval[0]:chunkInterval[1]])**2, axis=0)) / self.num_samples

        #self.power_ratio[globInterval[0]:globInterval[1]] = num / denom

    def get_output(self):
        return self.power



class SignalPowerSpectrum(SummaryDiagnostic):
    def __init__(self, 
        sig_name,
        sim_info, 
        block_size, 
        save_range,
        num_channels,
        sig_channels=slice(None),
        **kwargs
        ):
        """
        Will output the wrong value if it is exported in the middle of the save_range
        """
        self.save_range = save_range
        self.num_samples = save_range[1] - save_range[0]
        super().__init__(sim_info, block_size, save_range, export_func = "spectrum", **kwargs)
        self.sig_name = sig_name
        self.power = 0

        self.samplerate = self.sim_info.samplerate
        self.num_channels = num_channels
        self.sig_channels = sig_channels
        self.power = np.full((self.num_channels, self.num_samples), fill_value=np.nan)

        self.sample_counter = 0

        #self.plot_data["title"] = f"Power of {self.sig_name}. Samples: {self.save_range}"
        
    def save(self, processor, chunkInterval, globInterval):
        num_samples = chunkInterval[1] - chunkInterval[0]
        self.power[:,self.sample_counter:self.sample_counter+num_samples] = np.abs(processor.sig[self.sig_name][self.sig_channels, chunkInterval[0]:chunkInterval[1]])**2

        self.sample_counter += num_samples
        #self.power_ratio[globInterval[0]:globInterval[1]] = num / denom

    def get_output(self):
        f, spec = spsig.welch(self.power, self.samplerate, nperseg=512, scaling="spectrum", axis=-1)
        spec = np.mean(spec, axis=0)
        return spec
