import numpy as np


def fft_transpose(time_sig, n=None, add_empty_dim=False):
    freq_signal = np.fft.fft(time_sig, n=n, axis=-1)

    new_axis_order = np.concatenate(
        ([time_sig.ndim - 1], np.arange(time_sig.ndim - 1))
    )
    freq_signal = np.transpose(freq_signal, new_axis_order)
    if add_empty_dim:
        freq_signal = np.expand_dims(freq_signal, -1)
    return freq_signal


def ifft_transpose(freq_signal, remove_empty_dim=False):
    if remove_empty_dim:
        freq_signal = np.squeeze(freq_signal, axis=-1)
    time_signal = np.fft.ifft(freq_signal, axis=0)
    new_axis_order = np.concatenate((np.arange(1, freq_signal.ndim), [0]))
    time_signal = np.transpose(time_signal, new_axis_order)
    return time_signal


def correlate_sum_tt(time_filter, time_signal):
    """signal is two blocks long signal
    filter is one block FIR unpadded
    last dimension is time, the dimension to FFT over
    next to last dimension is summed over. Broadcasting applies"""
    assert 2 * time_filter.shape[-1] == time_signal.shape[-1]
    # assert(timeFilter.shape[-2] == timeSignal.shape[-2])
    freq_filter = fft_transpose(
        np.concatenate((np.zeros_like(time_filter), time_filter), axis=-1),
        add_empty_dim=False,
    )
    return correlate_sum_ft(freq_filter, time_signal)


def correlate_sum_ft(freq_filter, time_signal):
    """signal is two blocks long signal
    filter is FFT of one block FIR filter, padded with zeros BEFORE the ir
    last dimension of filter, next to last of signal is summed over"""
    freq_signal = fft_transpose(time_signal, add_empty_dim=False)
    return correlate_sum_ff(freq_filter, freq_signal)

def correlateSumTF(time_filter, freq_signal):
    """"""
    freq_filter = fft_transpose(
        np.concatenate((np.zeros_like(time_filter), time_filter), axis=-1),
        add_empty_dim=False,
    )
    return correlate_sum_ff(freq_filter, freq_signal)


def correlate_sum_ff(freq_filter, freq_signal):
    """signal is FFT of two blocks worth of signal
    filter is FFT of one block FIR filter, padded with zeros BEFORE the ir
    first dimension is frequencies, the dimension to IFFT over
    last dimension is summed over"""
    assert freq_filter.shape[0] == freq_signal.shape[0]
    assert freq_filter.shape[0] % 2 == 0
    output_len = freq_filter.shape[0] // 2
    filtered_signal = ifft_transpose(np.sum(freq_filter * freq_signal.conj(), axis=-1))
    return np.real(filtered_signal[..., :output_len])


def correlate_euclidian_tt(time_filter, time_signal):
    """signal is two blocks long signal
    filter is one block FIR unpadded
    Convolves every channel of signal with every channel of the filter
    Outputs (filter.shape[:-1], signal.shape[:-1], numSamples)"""
    assert 2 * time_filter.shape[-1] == time_signal.shape[-1]
    freq_filter = fft_transpose(
        np.concatenate((np.zeros_like(time_filter), time_filter), axis=-1),
        add_empty_dim=False,
    )
    return correlate_euclidian_ft(freq_filter, time_signal)


def correlate_euclidian_ft(freq_filter, time_signal):
    """signal is two blocks long signal
    filter is FFT of one block FIR filter, padded with zeros BEFORE the ir"""
    freq_signal = fft_transpose(time_signal, add_empty_dim=False)
    return correlate_euclidian_ff(freq_filter, freq_signal)

def correlate_euclidian_tf(time_filter, freq_signal):
    """"""
    freq_filter = fft_transpose(
        np.concatenate((np.zeros_like(time_filter), time_filter), axis=-1),
        add_empty_dim=False,
    )
    return correlate_euclidian_ff(freq_filter, freq_signal)


def correlate_euclidian_ff(freq_filter, freq_signal):
    """signal is two blocks long signal
    filter is FFT of one block FIR filter, padded with zeros BEFORE the ir
    last dimension of filter, next to last of signal is summed over"""
    assert freq_filter.shape[0] % 2 == 0
    output_len = freq_filter.shape[0] // 2

    filtered_signal = (
        freq_filter.reshape(freq_filter.shape + (1,) * (freq_signal.ndim - 1))
        * freq_signal.reshape(
            freq_signal.shape[0:1] + (1,) * (freq_filter.ndim - 1) + freq_signal.shape[1:]
        ).conj()
    )
    filtered_signal = ifft_transpose(filtered_signal)
    return np.real(filtered_signal[..., :output_len])


def convolve_sum(freq_filter, time_signal):
    """The freq domain filter should correspond to a time domain filter
    padded with zeros after its impulse response
    The signal should be 2 blocks worth of signal."""
    assert freq_filter.shape[-1] == time_signal.shape[-2]
    assert freq_filter.shape[0] == time_signal.shape[-1]
    assert freq_filter.shape[0] % 2 == 0
    output_len = freq_filter.shape[0] // 2

    filtered_signal = freq_filter @ fft_transpose(time_signal, add_empty_dim=True)
    filtered_signal = ifft_transpose(filtered_signal, remove_empty_dim=True)
    return np.real(filtered_signal[..., output_len:])


def convolve_euclidian_ff(freq_filter, freq_signal):
    """Convolves every channel of input with every channel of the filter
    Outputs (filter.shape, signal.shape, numSamples)
    Signals must be properly padded"""
    assert freq_filter.shape[0] % 2 == 0
    output_len = freq_filter.shape[0] // 2

    filtered_signal = freq_filter.reshape(
        freq_filter.shape + (1,) * (freq_signal.ndim - 1)
    ) * freq_signal.reshape(
        freq_signal.shape[0:1] + (1,) * (freq_filter.ndim - 1) + freq_signal.shape[1:]
    )
    filtered_signal = ifft_transpose(filtered_signal)
    return np.real(filtered_signal[..., output_len:])


def convolve_euclidian_ft(freq_filter, time_signal):
    """Convolves every channel of input with every channel of the filter
    Outputs (filter.shape[:-1], signal.shape[:-1], numSamples)"""
    assert freq_filter.shape[0] == time_signal.shape[-1]

    freq_signal = fft_transpose(time_signal, add_empty_dim=False)
    return convolve_euclidian_ff(freq_filter, freq_signal)
