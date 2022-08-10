import numpy as np
import scipy.signal as spsig



# =================== SCALING ===================
def linear(signal):
    return signal

def db_power(signal):
    return 10*np.log10(signal)

def db_amplitude(signal):
    return 20*np.log10(signal)

def natural_log(signal):
    return np.log(signal)

# def preset_scaling(scaling):
#     if scaling == "linear":
#         return linear
#     elif scaling == "db_power":
#         return db_power
#     elif scaling == "db_amplitude":
#         return db_amplitude
#     elif scaling == "natural_log":
#         return natural_log
#     else:
#         raise ValueError("Invalid scaling string")

# def scale_data(signal, scaling):
#     if callable(scaling):
#         return scaling(signal)
#     elif isinstance(scaling, str):
#         scaling_func = preset_scaling(scaling)
#         return scaling_func(signal)
#     else:
#         raise ValueError("Invalid scaling type")


def clip(low_lim, high_lim):
    def clip_internal(signal):
        return np.clip(signal, low_lim, high_lim)
    return clip_internal


def smooth(smooth_len):
    ir = np.ones((1, smooth_len)) / smooth_len
    def smooth_internal(signal):
        return spsig.oaconvolve(signal, ir, mode="full", axes=-1)[:,:signal.shape[-1]]
    return smooth_internal

