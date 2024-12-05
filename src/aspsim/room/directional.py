import numpy as np
import scipy.signal as spsig

import aspcore.filterdesign as fd



def differential_cardioid_microphone(signal_main, signal_secondary, microphone_distance, filter_order, c, samplerate, filter_below=None):
    """Constructs a cardioid microphone response from two omnidirectional microphones
    
    The two omnidirectional microphones must be separated by a distance of microphone_distance.
    The strongest response is in the direction of the main microphone from the secondary, and the null is in the opposite direction.

    Parameters
    ----------
    signal_main : ndarray of shape (num_mics, num_samples)
        the signal from the main microphone
    signal_secondary : ndarray of shape (num_mics, num_samples)
        the signal from the secondary microphone
    microphone_distance : float
        the distance between the microphones
    filter_order : int
        the order of the filter. The filter will have 2*filter_order + 1 taps. 
    c : float
        the speed of sound. 
    samplerate : int
        the sampling rate of the signals. 
    filter_below : float or None
        if not None, filters out the frequencies below this frequency, given in Hertz. The differential microphone
        is problematic at low frequencies, and this can be used to avoid this.

    Returns
    -------
    cardioid_signal : ndarray of shape (num_mics, num_samples)
        the signal from the cardioid microphone

    References
    ----------
    [benestyStudy2013] J. Benesty and J. Chen, Study and design of differential microphone arrays, vol. 6. in Springer Topics in Signal Processing, vol. 6. Springer, 2013.
    """
    num_freq = 4*filter_order # we need more or equal frequency samples compared to the ir_len
    ir_len = 2*filter_order + 1
    #wave_num = ft.get_real_wavenum(num_freq, samplerate, c)

    def freq_response_main_mic(f):
        wave_num = 2 * np.pi * f / c
        wave_num[wave_num == 0] = 1e-10 # To avoid division by zero warnings. Not strictly necessary.
        both_mic_factor = 1 / (1 - np.exp(-1j * 2 * wave_num * microphone_distance))
        both_mic_factor[0] = 0
        mic1_factor = np.conj(both_mic_factor)

        if filter_below is not None:
            num_freqs_to_filter = np.sum(f <= filter_below)
            scaling = np.linspace(0, 1, num_freqs_to_filter) ** 4
            mic1_factor[:num_freqs_to_filter] *= scaling
        return mic1_factor
    
    def freq_response_secondary_mic(f):
        wave_num = 2 * np.pi * f / c
        wave_num[wave_num == 0] = 1e-10 # To avoid division by zero warnings. Not strictly necessary.

        both_mic_factor = 1 / (1 - np.exp(-1j * 2 * wave_num * microphone_distance))
        both_mic_factor[0] = 0
        mic2_factor = -np.conj(both_mic_factor * np.exp(-1j * wave_num * microphone_distance))

        if filter_below is not None:
            num_freqs_to_filter = np.sum(f <= filter_below)
            scaling = np.linspace(0, 1, num_freqs_to_filter) ** 4
            mic2_factor[:num_freqs_to_filter] *= scaling
        return mic2_factor
    
    mic1_ir = fd.fir_from_frequency_function(freq_response_main_mic, ir_len, samplerate, window=None)
    mic2_ir = fd.fir_from_frequency_function(freq_response_secondary_mic, ir_len, samplerate, window=None)

    #mic1_factor = ft.insert_negative_frequencies(mic1_factor, even = True)
    #mic2_factor = ft.insert_negative_frequencies(mic2_factor, even = True)

    #mic1_ir, _ = fir_from_freqs_window(mic1_factor, ir_len, two_sided=True, window="hamming")
    #mic2_ir, _ = fir_from_freqs_window(mic2_factor, ir_len, two_sided=True, window="hamming")

    signal_main_filtered = spsig.fftconvolve(mic1_ir[None,:], signal_main, axes=-1, mode="full")
    signal_secondary_filtered = spsig.fftconvolve(mic2_ir[None,:], signal_secondary, axes=-1, mode="full")

    cardioid_signal = signal_main_filtered + signal_secondary_filtered
    cardioid_signal = cardioid_signal[...,filter_order:filter_order+signal_main.shape[-1]]
    return cardioid_signal





# def differential_cardioid_microphone(signal_main, signal_secondary, microphone_distance, filter_order, c, samplerate, filter_below=None):
#     """Constructs a cardioid microphone response from two omnidirectional microphones
    
#     The two omnidirectional microphones must be separated by a distance of microphone_distance.
#     The strongest response is in the direction of the main microphone from the secondary, and the null is in the opposite direction.

#     Parameters
#     ----------
#     signal_main : ndarray of shape (num_mics, num_samples)
#         the signal from the main microphone
#     signal_secondary : ndarray of shape (num_mics, num_samples)
#         the signal from the secondary microphone
#     microphone_distance : float
#         the distance between the microphones
#     filter_order : int
#         the order of the filter. The filter will have 2*filter_order + 1 taps. 
#     c : float
#         the speed of sound. 
#     samplerate : int
#         the sampling rate of the signals. 
#     filter_below : float or None
#         if not None, filters out the frequencies below this frequency, given in Hertz. The differential microphone
#         is problematic at low frequencies, and this can be used to avoid this.

#     Returns
#     -------
#     cardioid_signal : ndarray of shape (num_mics, num_samples)
#         the signal from the cardioid microphone

#     References
#     ----------
#     [benestyStudy2013] J. Benesty and J. Chen, Study and design of differential microphone arrays, vol. 6. in Springer Topics in Signal Processing, vol. 6. Springer, 2013.
#     """
#     num_freq = 4*filter_order # we need more or equal frequency samples compared to the ir_len
#     ir_len = 2*filter_order + 1
#     wave_num = ft.get_real_wavenum(num_freq, samplerate, c)

#     #def freq_response_main_mic(f):
#     both_mic_factor = 1 / (1 - np.exp(-1j * 2 * wave_num * microphone_distance))
#     both_mic_factor[0] = 0
#     mic1_factor = np.conj(both_mic_factor)
#     mic2_factor = -np.conj(both_mic_factor * np.exp(-1j * wave_num * microphone_distance))

#     if filter_below is not None:
#         #def freq_drop_function(freq):
#         #    return 1 - 1 / (1 + (freq / filter_below) ** 4)
        
#         freqs = ft.get_real_freqs(num_freq, samplerate)
#         num_freqs_to_filter = np.sum(freqs <= filter_below)
#         #freqs_to_filter = freqs[freqs_to_filter_mask]
#         scaling = np.linspace(0, 1, num_freqs_to_filter)
#         scaling = scaling ** 4
#         mic1_factor[:num_freqs_to_filter] *= scaling
#         mic2_factor[:num_freqs_to_filter] *= scaling

#     mic1_factor = ft.insert_negative_frequencies(mic1_factor, even = True)
#     mic2_factor = ft.insert_negative_frequencies(mic2_factor, even = True)

#     mic1_ir, _ = fir_from_freqs_window(mic1_factor, ir_len, two_sided=True, window="hamming")
#     mic2_ir, _ = fir_from_freqs_window(mic2_factor, ir_len, two_sided=True, window="hamming")

#     signal_main_filtered = signal.fftconvolve(mic1_ir[None,:], signal_main, axes=-1, mode="full")
#     signal_secondary_filtered = signal.fftconvolve(mic2_ir[None,:], signal_secondary, axes=-1, mode="full")

#     cardioid_signal = signal_main_filtered + signal_secondary_filtered
#     cardioid_signal = cardioid_signal[...,filter_order:filter_order+signal_main.shape[-1]]
#     return cardioid_signal