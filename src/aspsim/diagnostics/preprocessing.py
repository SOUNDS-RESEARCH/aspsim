"""Preprocessing helpers for diagnostics signals."""

import numpy as np
import scipy.signal as spsig


# =================== SCALING ===================
def linear(signal):
    """Return the signal unchanged."""
    return signal


def db_power(signal):
    """Convert power values to decibels."""
    return 10 * np.log10(signal)


def db_amplitude(signal):
    """Convert amplitude values to decibels."""
    return 20 * np.log10(signal)


def natural_log(signal):
    """Apply the natural logarithm to the signal."""
    return np.log(signal)


def clip(low_lim, high_lim):
    """Return a function that clips the signal to limits."""
    def clip_internal(signal):
        return np.clip(signal, low_lim, high_lim)

    return clip_internal


def smooth(smooth_len):
    """Return a function that smooths the signal."""
    ir = np.ones((1, smooth_len)) / smooth_len

    def smooth_internal(signal):
        return spsig.oaconvolve(signal, ir, mode="full", axes=-1)[:, : signal.shape[-1]]

    return smooth_internal
