import numpy as np
import scipy.signal as spsig


# =================== SCALING ===================
def linear(signal):
    return signal


def db_power(signal):
    return 10 * np.log10(signal)


def db_amplitude(signal):
    return 20 * np.log10(signal)


def natural_log(signal):
    return np.log(signal)


def clip(low_lim, high_lim):
    def clip_internal(signal):
        return np.clip(signal, low_lim, high_lim)

    return clip_internal


def smooth(smooth_len):
    ir = np.ones((1, smooth_len)) / smooth_len

    def smooth_internal(signal):
        return spsig.oaconvolve(signal, ir, mode="full", axes=-1)[:, : signal.shape[-1]]

    return smooth_internal
