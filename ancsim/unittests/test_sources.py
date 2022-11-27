import pytest
import hypothesis as hyp
import hypothesis.strategies as st

import numpy as np
import scipy.signal.windows as win
import scipy.signal as spsig

import ancsim.signal.sources as sources
import ancsim.signal.sourcescollection as sourcescol

import aspcol.correlation as cr

# @pytest.mark.parametrize("src", [sources.SineSource(1,100,8000)])
# def test_source_blocks(src, ):


def test_sine_blocks():
    numSamples = 1000
    numBlocks = 50
    block_size = int(numSamples / numBlocks)
    freq = 100
    sr = 8000

    np.random.seed(0)
    src = sources.SineSource(1, 1, freq, sr)
    ref = src.get_samples(numSamples)

    np.random.seed(0)
    source = sources.SineSource(1, 1, freq, sr)
    out = np.zeros(numSamples)
    for i in range(0, numSamples, block_size):
        out[i : i + block_size] = source.get_samples(block_size)

    assert np.mean(out - ref) < 1e-6


def test_noisesrc_blocks():
    N = 10000
    numBlocks = 100
    block_size = int(N / numBlocks)
    freqs = (100, 400)
    sr = 8000

    np.random.seed(0)
    source = sources.BandlimitedNoiseSource(1, 1, freqs, sr)
    sig1 = source.get_samples(N)
    # sig1 = util.pow2db(np.abs(np.fft.rfft(sig1))**2)

    np.random.seed(0)
    source2 = sources.BandlimitedNoiseSource(1, 1, freqs, sr)
    sig2 = np.zeros((1, N))
    for i in range(N // 100):
        sig2[:, i * block_size : (i + 1) * block_size] = source2.get_samples(block_size)

    assert np.sum(np.abs(sig1 - sig2)) < 1e-6


def test_multisine_blocks():
    numSamples = 10000
    numBlocks = 100
    block_size = int(numSamples / numBlocks)
    freqs = np.linspace(100, 1000, 10)
    sr = 8000

    rng = np.random.default_rng(1)
    source1 = sources.MultiSineSource(1, 1, freqs, sr, rng=rng)
    sig1 = source1.get_samples(numSamples)

    rng = np.random.default_rng(1)
    source2 = sources.MultiSineSource(1, 1, freqs, sr, rng=rng)
    sig2 = np.zeros((1, numSamples))
    for i in range(numBlocks):
        sig2[:, i * block_size : (i + 1) * block_size] = source2.get_samples(block_size)

    assert np.sum(np.abs(sig1 - sig2)) < 1e-6



@hyp.given(num_samples = st.integers(min_value=64,max_value=1024),
            num_channels = st.integers(min_value=1, max_value=5),
            block_size = st.integers(min_value=1, max_value = 64),
            seq_len = st.integers(min_value = 1, max_value=1500))
def test_sequence_source_blocks(num_samples, num_channels, block_size, seq_len):
    num_samples -= num_samples % block_size
    num_blocks = num_samples // block_size

    rng = np.random.default_rng(1)
    seq = rng.normal(0, 1, (num_channels,seq_len))

    src1 = sources.Sequence(seq, end_mode="repeat")
    sig1 = src1.get_samples(num_samples)

    src2 = sources.Sequence(seq, end_mode="repeat")
    sig2 = np.zeros((num_channels, num_samples))

    for i in range(num_blocks):
        sig2[:, i * block_size : (i + 1) * block_size] = src2.get_samples(block_size)
    assert sig1.shape == (num_channels, num_samples)
    assert sig2.shape == (num_channels, num_samples)
    assert np.mean(np.abs(sig1 - sig2)) < 1e-10

@hyp.given(num_channels = st.integers(min_value=1, max_value=5),
            num_repeats = st.integers(min_value=2, max_value = 8),
            seq_len = st.integers(min_value = 1, max_value=1500))
def test_sequence_source_repeat(num_channels, num_repeats, seq_len):
    rng = np.random.default_rng(1)
    seq = rng.normal(0, 1, (num_channels,seq_len))

    src = sources.Sequence(seq, end_mode="repeat")
    sig = src.get_samples(seq_len*num_repeats)

    for r in range(num_repeats):
        assert np.allclose(seq, sig[:, r*seq_len:(r+1)*seq_len])



def test_gold_sequence_remembering_state():
    src = sources.GoldSequenceSource(1,1, 11)
    N = 100
    sig1 = src.get_samples(N)
    sig2 = src.get_samples(N)
    sig = np.concatenate((sig1, sig2), axis=-1)

    src = sources.GoldSequenceSource(1,1,11)
    sigCompare = src.get_samples(2 * N)

    assert np.allclose(sigCompare, sig)





#@hyp.settings(deadline=None)
#@hyp.given(bs = st.integers(min_value=1, max_value=5))
@hyp.settings(deadline=None)
@hyp.given(bs = st.integers(min_value=1, max_value=5),
            num_samples = st.integers(min_value=10, max_value=100),
            num_channels = st.integers(min_value=1, max_value=3))

def test_pulse_train(bs, num_samples, num_channels):
    #import matplotlib.pyplot as plt
    rng = np.random.default_rng()
    period = rng.integers(low=1, high=10, size=num_channels)
    dly = rng.integers(low=0, high=30, size=num_channels)
    amp = rng.standard_normal(size=num_channels)

    pulse_src = sources.PulseTrain(num_channels, amp, period, dly)
    sig = np.zeros((num_channels,num_samples))
    for i in range(num_samples // bs):
        sig[:,i*bs:(i+1)*bs] = pulse_src.get_samples(bs)
    sig[:,(i+1)*bs:] = pulse_src.get_samples(num_samples-(i+1)*bs)

    for j in range(num_channels):
        for i in range(num_samples):
            if (i - dly[j]) % period[j] == 0:
                assert sig[j,i] == amp[j]
            else:
                assert sig[j,i] == 0
    #plt.plot(sig.T)
    #plt.show()





def test_autocorr_of_AutocorrSource():
    #autocorr = np.array([[2,0.5,0.5,0.2,0.1]], dtype=float)
    autocorr = np.zeros((1, 25))
    autocorr[0,0] = 1
    autocorr[0,4] = 0.5
    src = sources.AutocorrSource(1, autocorr)
    sig = src.get_samples(int(1e6))
    ac = cr.Autocorrelation(1, 5*autocorr.shape[-1], autocorr.shape[0])
    ac.update(sig)
    print(ac.corr.state)
    pass # todo: calculate the power of the white noise. The relative sizes of the resulting autocorr is correct



# def test_bandlim_noise():
#     # import matplotlib.pyplot as plt
#     sr = 8000
#     freqLim = (200, 250)
#     src = sources.BandlimitedNoiseSource(1, freqLim, sr)
#     signal = src.get_samples(10000)
#     f, p = spsig.welch(signal, sr, nperseg=2048, window="hamming")
    # p = np.abs(np.fft.fft(np.squeeze(signal)))**2
    # plt.plot(10*np.log10(p))
    # plt.plot(f, 10*np.log10(np.squeeze(p)))
    # plt.show()


# def test_gold_sequence_autocorrelation():
#     import matplotlib.pyplot as plt

#     src = sources.GoldSequenceSource(11)
#     N = 10000
#     sig = np.squeeze(src.get_samples(N))
#     corr = np.correlate(sig, sig, mode="full")
#     plt.plot(corr)
    # plt.show()


# def test_MLS_autocorrelation():
#     import matplotlib.pyplot as plt

#     src = sources.MLS(11, [11, 2, 0])
#     N = 1000
#     sig = np.squeeze(src.get_samples(N))
#     corr = np.correlate(sig, sig, mode="full")
#     plt.plot(corr)
    # plt.show()


# def test_preferred_pair_autocorrelation():
#     import matplotlib.pyplot as plt

#     src = sources.MLS(11, [11, 2, 0])
#     src2 = sources.MLS(11, [11, 8, 5, 2, 0])
#     N = 1000
#     sig = np.squeeze(src.get_samples(N))
#     sig2 = np.squeeze(src2.get_samples(N))
#     corr = np.correlate(sig, sig2, mode="full")
#     plt.plot(corr)
    # plt.show()


# def test_gold_autocorrelation_2():
#     import matplotlib.pyplot as plt

#     src = sources.MLS(8, [8, 6, 5, 3, 0], state=[0, 0, 0, 0, 0, 0, 0, 1])
#     src2 = sources.MLS(8, [8, 6, 5, 2, 0], state=[0, 0, 0, 0, 0, 0, 0, 1])
#     N = 256
#     sig = np.squeeze(src.get_samples(N))
#     sig2 = np.squeeze(src2.get_samples(N))
#     corr = np.correlate(sig, sig2, mode="full")
#     plt.plot(corr)
#     # plt.show()



@pytest.fixture
def chirpSetup():
    samplesToSweep = 3999
    minFreq = 100
    maxFreq = 3000
    sr = 8000
    amp = 1
    src = sources.LinearChirpSource(amp, (minFreq, maxFreq), samplesToSweep, sr)
    return src, sr, minFreq, maxFreq, samplesToSweep


def test_chirp_blocks():
    numSamples = 10000
    numBlocks = 100
    block_size = int(numSamples / numBlocks)
    freqs = (100, 400)
    numToSweep = 1000
    sr = 8000

    np.random.seed(0)
    source = sources.LinearChirpSource(1, 1, freqs, numToSweep, sr)
    noise = source.get_samples(numSamples)

    np.random.seed(0)
    source2 = sources.LinearChirpSource(1, 1, freqs, numToSweep, sr)
    noise2 = np.zeros(numSamples)
    for i in range(numBlocks):
        noise2[i * block_size : (i + 1) * block_size] = source2.get_samples(block_size)

    assert np.sum(np.abs(noise - noise2)) < 1e-6

def test_chirp_min_frequency_property(chirpSetup):
    src, sr, minFreq, maxFreq, samplesToSweep = chirpSetup
    signal = src.get_samples(samplesToSweep * 10)
    assert src.freq - minFreq < 1e-6


def test_chirp_max_frequency_property(chirpSetup):
    src, sr, minFreq, maxFreq, samplesToSweep = chirpSetup
    signal = src.get_samples(samplesToSweep * 9)
    assert src.freq - maxFreq < 1e-6


def test_chirp_min_frequency_fft(chirpSetup):
    src, sr, minFreq, maxFreq, samplesToSweep = chirpSetup
    block_size = 128
    fftsize = 2 ** 14
    src.get_samples(int(samplesToSweep * 20 - block_size / 2))
    signal = src.get_samples(block_size)
    freq = get_freq_of_sine(signal, sr, block_size, fftsize)

    freqBinDif = sr / fftsize
    blockFreqChange = (block_size / (samplesToSweep * 2)) * (maxFreq - minFreq)
    assert freq - minFreq < blockFreqChange


def test_chirp_max_frequency_fft(chirpSetup):
    src, sr, minFreq, maxFreq, samplesToSweep = chirpSetup
    block_size = 128
    fftsize = 2 ** 14
    src.get_samples(int(samplesToSweep * 19 - block_size / 2))
    signal = src.get_samples(block_size)
    freq = get_freq_of_sine(signal, sr, block_size, fftsize)

    freqBinDif = sr / fftsize
    blockFreqChange = (block_size / (samplesToSweep * 2)) * (maxFreq - minFreq)
    assert freq - maxFreq < blockFreqChange


def get_freq_of_sine(signal, sr, block_size, fftsize):
    signal *= win.hamming(block_size)
    freqs = np.fft.rfft(signal, n=fftsize)
    maxBin = np.argmax(freqs)
    freq = sr * maxBin / (2 * fftsize)
    return freq
