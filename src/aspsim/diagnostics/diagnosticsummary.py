"""Summary diagnostics and helpers."""

import json

import numpy as np
import scipy.signal as spsig

import aspsim.diagnostics.core as diacore
import aspsim.diagnostics.plot as dplt


def add_to_summary(diagName, summaryValues, timeIdx, folderPath):
    """Add summary values to a JSON file."""
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
    """Compute mean values near a time index."""
    summaryValues = {}
    numToAverage = 3000

    for filtName, output in outputs.items():
        # val = diagnostic.getOutput()[timeIdx-numToAverage:timeIdx]
        val = output[..., timeIdx - numToAverage : timeIdx]
        filterArray = np.logical_not(np.isnan(val))
        summaryValues[filtName] = np.mean(val[filterArray])
    return summaryValues


def last_value():
    """Raise NotImplementedError for a last-value summary function."""
    raise NotImplementedError


class SummaryDiagnostic(diacore.Diagnostic):
    """Base class for summary diagnostics."""

    export_functions = {
        "npz": dplt.savenpz,
        "text": dplt.txt,
        "spectrum": dplt.spectrum_plot,
    }

    def __init__(
        self,
        sim_info,
        block_size,
        save_at,
        export_func="text",
        keep_only_last_export=False,
        export_kwargs=None,
        preprocess=None,
    ):
        """Initialize a summary diagnostic.

        Will use the samples between start_sample (inclusive) and end_sample (exclusive).

        Parameters
        ----------
        save_at : tuple
            The range as (start_sample, end_sample).
        """
        if isinstance(save_at, diacore.IntervalCounter):
            raise NotImplementedError
        else:
            export_at = [save_at[1]]
            save_at = diacore.IntervalCounter(((save_at[0], save_at[1]),))

        super().__init__(
            sim_info,
            export_at,
            save_at,
            export_func,
            keep_only_last_export,
            export_kwargs,
            preprocess,
        )


class SignalPowerRatioSummary(SummaryDiagnostic):
    """Summarize a ratio of signal powers."""

    def __init__(
        self,
        numerator_name,
        denom_name,
        sim_info,
        block_size,
        save_range,
        numerator_channels=slice(None),
        denom_channels=slice(None),
        **kwargs,
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

        self.plot_data["title"] = (
            f"Ratio of power: {self.numerator_name} / {self.denom_name}. Samples: {self.save_range}"
        )

    def save(self, processor, sig, chunk_interval, glob_interval):
        """Accumulate power ratio statistics for the current chunk."""
        self.num_power += (
            np.sum(
                np.mean(
                    np.abs(
                        processor.sig[self.numerator_name][
                            self.numerator_channels,
                            chunk_interval[0] : chunk_interval[1],
                        ]
                    )
                    ** 2,
                    axis=0,
                )
            )
            / self.num_samples
        )
        self.denom_power += (
            np.sum(
                np.mean(
                    np.abs(
                        processor.sig[self.denom_name][
                            self.denom_channels, chunk_interval[0] : chunk_interval[1]
                        ]
                    )
                    ** 2,
                    axis=0,
                )
            )
            / self.num_samples
        )

        # self.power_ratio[globInterval[0]:globInterval[1]] = num / denom

    def get_output(self):
        """Return the final power ratio."""
        return self.num_power / self.denom_power


class SignalPowerSummary(SummaryDiagnostic):
    """Summarize signal power across a range."""

    def __init__(
        self,
        sig_name,
        sim_info,
        block_size,
        save_range,
        sig_channels=slice(None),
        **kwargs,
    ):
        """Initialize power summary for a signal.

        Notes
        -----
        Exporting in the middle of the save_range yields an incorrect value.
        """
        self.save_range = save_range
        self.num_samples = save_range[1] - save_range[0]
        super().__init__(sim_info, block_size, save_range, **kwargs)
        self.sig_name = sig_name
        self.power = 0

        self.sig_channels = sig_channels

        # self.plot_data["title"] = f"Power of {self.sig_name}. Samples: {self.save_range}"

    def save(self, processor, sig, chunk_interval, glob_interval):
        """Accumulate power for the current chunk."""
        self.power += (
            np.sum(
                np.mean(
                    np.abs(
                        processor.sig[self.sig_name][
                            self.sig_channels, chunk_interval[0] : chunk_interval[1]
                        ]
                    )
                    ** 2,
                    axis=0,
                )
            )
            / self.num_samples
        )

        # self.power_ratio[globInterval[0]:globInterval[1]] = num / denom

    def get_output(self):
        """Return the accumulated power."""
        return self.power


class SignalPowerSpectrum(SummaryDiagnostic):
    """Summarize signal power spectrum across a range."""

    def __init__(
        self,
        sig_name,
        sim_info,
        block_size,
        save_range,
        num_channels,
        sig_channels=slice(None),
        **kwargs,
    ):
        """Initialize power spectrum summary for a signal.

        Notes
        -----
        Exporting in the middle of the save_range yields an incorrect value.
        """
        self.save_range = save_range
        self.num_samples = save_range[1] - save_range[0]
        super().__init__(
            sim_info, block_size, save_range, export_func="spectrum", **kwargs
        )
        self.sig_name = sig_name

        self.samplerate = self.sim_info.samplerate
        self.num_channels = num_channels
        self.sig_channels = sig_channels
        self.power = np.full((self.num_channels, self.num_samples), fill_value=np.nan)

        self.sample_counter = 0

        # self.plot_data["title"] = f"Power of {self.sig_name}. Samples: {self.save_range}"

    def save(self, processor, sig, chunk_interval, glob_interval):
        """Accumulate spectrum data for the current chunk."""
        num_samples = chunk_interval[1] - chunk_interval[0]
        self.power[:, self.sample_counter : self.sample_counter + num_samples] = (
            np.abs(
                processor.sig[self.sig_name][
                    self.sig_channels, chunk_interval[0] : chunk_interval[1]
                ]
            )
            ** 2
        )

        self.sample_counter += num_samples
        # self.power_ratio[globInterval[0]:globInterval[1]] = num / denom

    def get_output(self):
        """Return the Welch power spectrum."""
        f, spec = spsig.welch(
            self.power, self.samplerate, nperseg=512, scaling="spectrum", axis=-1
        )
        spec = np.mean(spec, axis=0)
        return spec
