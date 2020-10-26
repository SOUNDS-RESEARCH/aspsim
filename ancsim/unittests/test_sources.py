import pytest
import ancsim.signal.sources as sources

import numpy as np
import scipy.signal.windows as win
import scipy.signal as spsig


@pytest.fixture
def chirpSetup():
    samplesToSweep = 3999
    minFreq = 100
    maxFreq = 3000
    sr = 8000
    amp = 1
    src = sources.LinearChirpSource(amp, (minFreq, maxFreq), samplesToSweep, sr)
    return src, sr, minFreq, maxFreq, samplesToSweep


def test_chirp_min_frequency_property(chirpSetup):
    src, sr, minFreq, maxFreq, samplesToSweep = chirpSetup
    signal = src.getSamples(samplesToSweep * 10)
    assert src.freq - minFreq < 1e-6


def test_chirp_max_frequency_property(chirpSetup):
    src, sr, minFreq, maxFreq, samplesToSweep = chirpSetup
    signal = src.getSamples(samplesToSweep * 9)
    assert src.freq - maxFreq < 1e-6


def test_chirp_min_frequency_fft(chirpSetup):
    src, sr, minFreq, maxFreq, samplesToSweep = chirpSetup
    blockSize = 128
    fftsize = 2 ** 14
    src.getSamples(int(samplesToSweep * 20 - blockSize / 2))
    signal = src.getSamples(blockSize)
    freq = get_freq_of_sine(signal, sr, blockSize, fftsize)

    freqBinDif = sr / fftsize
    blockFreqChange = (blockSize / (samplesToSweep * 2)) * (maxFreq - minFreq)
    assert freq - minFreq < blockFreqChange


def test_chirp_max_frequency_fft(chirpSetup):
    src, sr, minFreq, maxFreq, samplesToSweep = chirpSetup
    blockSize = 128
    fftsize = 2 ** 14
    src.getSamples(int(samplesToSweep * 19 - blockSize / 2))
    signal = src.getSamples(blockSize)
    freq = get_freq_of_sine(signal, sr, blockSize, fftsize)

    freqBinDif = sr / fftsize
    blockFreqChange = (blockSize / (samplesToSweep * 2)) * (maxFreq - minFreq)
    assert freq - maxFreq < blockFreqChange


def get_freq_of_sine(signal, sr, blockSize, fftsize):
    signal *= win.hamming(blockSize)
    freqs = np.fft.rfft(signal, n=fftsize)
    maxBin = np.argmax(freqs)
    freq = sr * maxBin / (2 * fftsize)
    return freq


# @pytest.mark.parametrize("src", [sources.SineSource(1,100,8000)])
# def test_source_blocks(src, ):


def test_sine_blocks():
    numSamples = 1000
    numBlocks = 50
    blockSize = int(numSamples / numBlocks)
    freq = 100
    sr = 8000

    np.random.seed(0)
    src = sources.SineSource(1, freq, sr)
    ref = src.getSamples(numSamples)

    np.random.seed(0)
    source = sources.SineSource(1, freq, sr)
    out = np.zeros(numSamples)
    for i in range(0, numSamples, blockSize):
        out[i : i + blockSize] = source.getSamples(blockSize)

    assert np.mean(out - ref) < 1e-6


def test_noisesrc_blocks():
    N = 10000
    numBlocks = 100
    blockSize = int(N / numBlocks)
    freqs = (100, 400)
    sr = 8000

    np.random.seed(0)
    source = sources.BandlimitedNoiseSource(1, freqs, sr)
    sig1 = source.getSamples(N)
    # sig1 = util.pow2db(np.abs(np.fft.rfft(sig1))**2)

    np.random.seed(0)
    source2 = sources.BandlimitedNoiseSource(1, freqs, sr)
    sig2 = np.zeros((1, N))
    for i in range(N // 100):
        sig2[:, i * blockSize : (i + 1) * blockSize] = source2.getSamples(blockSize)
    # sig2 = util.pow2db(np.abs(np.fft.rfft(sig2))**2)

    assert np.sum(np.abs(sig1 - sig2)) < 1e-6
    # plt.plot(np.linspace(0,s.SAMPLERATE/2, N/2+1), np.squeeze(sig2), alpha=0.7)
    # plt.plot(np.linspace(0,s.SAMPLERATE/2, N/2+1), np.squeeze(sig1), alpha = 0.7)
    # plt.show()


def test_chirp_blocks():
    numSamples = 10000
    numBlocks = 100
    blockSize = int(numSamples / numBlocks)
    freqs = (100, 400)
    numToSweep = 1000
    sr = 8000

    np.random.seed(0)
    source = sources.LinearChirpSource(1, freqs, numToSweep, sr)
    noise = source.getSamples(numSamples)

    np.random.seed(0)
    source2 = sources.LinearChirpSource(1, freqs, numToSweep, sr)
    noise2 = np.zeros(numSamples)
    for i in range(numBlocks):
        noise2[i * blockSize : (i + 1) * blockSize] = source2.getSamples(blockSize)

    assert np.sum(np.abs(noise - noise2)) < 1e-6

    # for i in range(100):
    #    f = np.abs(np.fft.rfft(noise[i*100:(i+1)*100]))
    #    freq = np.argmax(f) * s.SAMPLERATE / 100
    #    print(freq)


def test_multisine_blocks():
    numSamples = 10000
    numBlocks = 100
    blockSize = int(numSamples / numBlocks)
    freqs = np.linspace(100, 1000, 10)
    sr = 8000

    np.random.seed(0)
    source1 = sources.MultiSineSource(1, freqs, sr)
    sig1 = source1.getSamples(numSamples)

    np.random.seed(0)
    source2 = sources.MultiSineSource(1, freqs, sr)
    sig2 = np.zeros((1, numSamples))
    for i in range(numBlocks):
        sig2[:, i * blockSize : (i + 1) * blockSize] = source2.getSamples(blockSize)

    assert np.sum(np.abs(sig1 - sig2)) < 1e-6


def test_bandlim_noise():
    # import matplotlib.pyplot as plt
    sr = 8000
    freqLim = (200, 250)
    src = sources.BandlimitedNoiseSource(1, freqLim, sr)
    signal = src.getSamples(10000)
    f, p = spsig.welch(signal, sr, nperseg=2048, window="hamming")
    # p = np.abs(np.fft.fft(np.squeeze(signal)))**2
    # plt.plot(10*np.log10(p))
    # plt.plot(f, 10*np.log10(np.squeeze(p)))
    # plt.show()


def test_gold_sequence_remembering_state():
    src = sources.GoldSequenceSource(11)
    N = 100
    sig1 = src.getSamples(N)
    sig2 = src.getSamples(N)
    sig = np.concatenate((sig1, sig2), axis=-1)

    src = sources.GoldSequenceSource(11)
    sigCompare = src.getSamples(2 * N)

    assert np.allclose(sigCompare, sig)


def test_gold_sequence_autocorrelation():
    import matplotlib.pyplot as plt

    src = sources.GoldSequenceSource(11)
    N = 10000
    sig = np.squeeze(src.getSamples(N))
    corr = np.correlate(sig, sig, mode="full")
    plt.plot(corr)
    # plt.show()


def test_MLS_autocorrelation():
    import matplotlib.pyplot as plt

    src = sources.MLS(11, [11, 2, 0])
    N = 1000
    sig = np.squeeze(src.getSamples(N))
    corr = np.correlate(sig, sig, mode="full")
    plt.plot(corr)
    # plt.show()


def test_preferred_pair_autocorrelation():
    import matplotlib.pyplot as plt

    src = sources.MLS(11, [11, 2, 0])
    src2 = sources.MLS(11, [11, 8, 5, 2, 0])
    N = 1000
    sig = np.squeeze(src.getSamples(N))
    sig2 = np.squeeze(src2.getSamples(N))
    corr = np.correlate(sig, sig2, mode="full")
    plt.plot(corr)
    # plt.show()


def test_gold_autocorrelation_2():
    import matplotlib.pyplot as plt

    src = sources.MLS(8, [8, 6, 5, 3, 0], state=[0, 0, 0, 0, 0, 0, 0, 1])
    src2 = sources.MLS(8, [8, 6, 5, 2, 0], state=[0, 0, 0, 0, 0, 0, 0, 1])
    N = 256
    sig = np.squeeze(src.getSamples(N))
    sig2 = np.squeeze(src2.getSamples(N))
    corr = np.correlate(sig, sig2, mode="full")
    plt.plot(corr)
    # plt.show()
