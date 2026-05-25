"""Signal source definitions used by simulations."""

from abc import ABC, abstractmethod

import aspcore.correlation as cr
import aspcore.filter as fc
import aspcore.utilities as coreutil
import numpy as np
import scipy.linalg as splin
import scipy.signal as spsig
import soundfile as sf

import aspsim.utilities as util


class Source(ABC):
    """Abstract base class for signal sources."""

    def __init__(self, num_channels, rng=None):
        self.num_channels = num_channels

        if rng is None:
            self.rng = np.random.default_rng(123456)
        else:
            self.rng = rng

        self.metadata = {}
        self.metadata["number of channels"] = self.num_channels
        # self.metadata["power"] = self.power

    @abstractmethod
    def get_samples(self, num_samples):
        """Return a block of samples."""
        return np.zeros((self.num_channels, num_samples))


class SourceArray:
    """Collection of multiple sources of the same type."""

    def __init__(self, source_type, num_sources, amplitude, *args):
        if isinstance(amplitude, (int, float)):
            amplitude = [amplitude for _ in range(num_sources)]

        self.sources = [source_type(amplitude[i], *args) for i in range(num_sources)]

    def get_samples(self, num_samples):
        """Return samples from all sources."""
        output = np.concatenate(
            [src.get_samples(num_samples) for src in self.sources], axis=0
        )
        return output


class Sequence(Source):
    """Play a finite or repeating sequence."""

    def __init__(self, audio, amp_factor=1, end_mode="repeat"):
        """Play the supplied sequence.

        Parameters
        ----------
        audio is np.ndarray of audio samples to be played.
                    Shape is (num_channels, num_samples)
        end_mode : can be any of {'repeat', 'zeros', 'raise'}
            with 'repeat' the sequence starts over again indefinitely
            with 'zeros' the sequence plays once and then only 0s
            with 'raise' the source will raise an exception if get_samples()
            is called after the sequence is finished
        """
        assert isinstance(audio, np.ndarray)
        if audio.ndim == 1:
            audio = audio[None, :]
        elif audio.ndim != 2:
            raise ValueError
        super().__init__(audio.shape[0])

        self.audio = audio
        self.amp_factor = amp_factor
        self.end_mode = end_mode
        self.tot_samples = audio.shape[1]
        self.current_sample = 0

    def get_samples(self, num_samples):
        """Return the next samples from the sequence."""
        sig = np.zeros((self.num_channels, num_samples))
        if self.end_mode == "repeat":
            block_lengths = util.calc_block_sizes(
                num_samples, self.current_sample, self.tot_samples
            )
            i = 0
            for block_len in block_lengths:
                sig[:, i : i + block_len] = self.audio[
                    :, self.current_sample : self.current_sample + block_len
                ]
                self.current_sample = (
                    self.current_sample + block_len
                ) % self.tot_samples
                i += block_len
        elif self.end_mode == "zeros":
            raise NotImplementedError
        elif self.end_mode == "raise":
            if self.current_sample + num_samples >= self.tot_samples:
                raise StopIteration("End of audio signal")

            sig = self.audio[:, self.current_sample : self.current_sample + num_samples]
            self.current_sample += num_samples
            return sig
        else:
            raise ValueError("Invalid end mode")
        return sig * self.amp_factor


class WhiteNoiseSource(Source):
    """Generate white noise with a specified power."""

    def __init__(self, num_channels, power, rng=None):
        super().__init__(num_channels, rng)
        self.set_power(power)

        if isinstance(self.power, (np.ndarray)):
            self.metadata["power"] = self.power.tolist()
        else:
            self.metadata["power"] = self.power

    def get_samples(self, num_samples):
        """Return white noise samples."""
        return self.rng.normal(
            loc=0, scale=self.stdDev, size=(num_samples, self.num_channels)
        ).T

    def set_power(self, newPower):
        """Set the target noise power."""
        self.power = newPower
        self.stdDev = np.sqrt(newPower)

        if isinstance(self.power, np.ndarray):
            assert self.power.ndim == 1
            assert self.power.shape[0] == self.num_channels or self.power.shape[0] == 1


class SineSource(Source):
    """Generate a sine wave signal."""

    def __init__(self, num_channels, power, freq, samplerate, rng=None):
        super().__init__(num_channels, rng)
        self.power = power
        self.amplitude = np.sqrt(2 * self.power)
        # p = a^2 / 2
        # 2p = a^2
        # sqrt(2p) = a
        self.freq = freq
        self.samplerate = samplerate
        self.phase = self.rng.uniform(
            low=0, high=2 * np.pi, size=(self.num_channels, 1)
        )

        self.phase_per_sample = 2 * np.pi * self.freq / self.samplerate

        if isinstance(self.power, (np.ndarray)):
            self.metadata["power"] = self.power.tolist()
        else:
            self.metadata["power"] = self.power
        self.metadata["frequency"] = self.freq

    def get_samples(self, num_samples):
        """Return sine wave samples."""
        noise = self.amplitude * np.cos(
            self.phase_per_sample * np.arange(num_samples)[None, :] + self.phase
        )
        self.phase = (self.phase + num_samples * self.phase_per_sample) % (2 * np.pi)
        return noise


class Counter(Source):
    """Count upward by one per sample.

    Used primarily for debugging.
    """

    def __init__(self, num_channels, start_number=0):
        super().__init__(num_channels)
        if num_channels > 1:
            raise NotImplementedError
        self.current_number = start_number

        self.metadata["start number"] = start_number

    def get_samples(self, num_samples):
        """Return a sequence of increasing integers."""
        values = np.arange(self.current_number, self.current_number + num_samples)
        self.current_number += num_samples
        return values


class MultiSineSource(Source):
    """Generate a sum of sine waves."""

    def __init__(self, num_channels, power, freq, samplerate, rng=None):
        super().__init__(num_channels, rng)
        if num_channels > 1:
            raise NotImplementedError
        if isinstance(power, (list, tuple, np.ndarray)):
            raise NotImplementedError
        if isinstance(freq, (list, tuple)):
            freq = np.array(freq)
        assert freq.ndim == 1
        self.power = power
        self.freq = freq
        self.samplerate = samplerate
        self.numSines = len(freq)
        self.phase = self.rng.uniform(low=0, high=2 * np.pi, size=self.numSines)
        self.phasePerSample = 2 * np.pi * self.freq / self.samplerate
        self.amplitude = np.sqrt(2 * self.power / self.numSines)

        if isinstance(self.power, (np.ndarray)):
            self.metadata["power"] = self.power.tolist()
        else:
            self.metadata["power"] = self.power
        self.metadata["frequency"] = self.freq

    def get_samples(self, num_samples):
        """Return multi-sine samples."""
        noise = np.zeros((1, num_samples))
        for i in range(self.numSines):
            noise += (
                self.amplitude
                * np.cos(
                    np.arange(num_samples) * self.phasePerSample[i] + self.phase[i]
                )[None, :]
            )
            self.phase[i] = (self.phase[i] + num_samples * self.phasePerSample[i]) % (
                2 * np.pi
            )
        return noise


class BandlimitedNoiseSource(Source):
    """Generate band-limited noise."""

    def __init__(self, num_channels, power, freqLim, samplerate, rng=None):
        super().__init__(num_channels, rng)
        assert len(freqLim) == 2
        assert freqLim[0] < freqLim[1]
        self.num_channels = num_channels

        wp = [freq for freq in freqLim]
        self.filtCoef = spsig.butter(
            16, wp, btype="bandpass", output="sos", fs=samplerate
        )

        testSig = spsig.sosfilt(
            self.filtCoef, self.rng.normal(size=(self.num_channels, 10000))
        )
        self.testSigPow = np.mean(testSig**2)
        self.set_power(power)

        self.zi = spsig.sosfilt_zi(self.filtCoef)
        self.zi = np.tile(np.expand_dims(self.zi, 1), (1, self.num_channels, 1))

        if isinstance(self.power, (np.ndarray)):
            self.metadata["power"] = self.power.tolist()
        else:
            self.metadata["power"] = self.power
        self.metadata["frequency span"] = freqLim

    def get_samples(self, num_samples):
        """Return band-limited noise samples."""
        noise = self.amplitude * self.rng.normal(size=(self.num_channels, num_samples))
        filtNoise, self.zi = spsig.sosfilt(self.filtCoef, noise, zi=self.zi, axis=-1)
        return filtNoise

    def set_power(self, newPower):
        """Set the target noise power."""
        self.power = newPower
        self.amplitude = np.sqrt(newPower / self.testSigPow)

        # if isinstance(self.power, np.ndarray):
        #     assert self.power.ndim == 1
        #     assert self.power.shape[0] == self.num_channels or \
        #             self.power.shape[0] == 1


class PulseTrain(Source):
    """Generate a periodic pulse train."""

    def __init__(self, num_channels, amplitude, period_len, delay=0):
        super().__init__(num_channels)

        if isinstance(amplitude, (tuple, list, np.ndarray)):
            assert len(amplitude) == self.num_channels
            self.amplitude = np.array(amplitude)
        else:
            self.amplitude = np.ones(self.num_channels) * amplitude

        if isinstance(period_len, (tuple, list, np.ndarray)):
            assert len(period_len) == self.num_channels
            self.period_len = np.array(period_len)
        else:
            self.period_len = np.ones(self.num_channels) * period_len

        if isinstance(delay, (tuple, list, np.ndarray)):
            assert len(delay) == self.num_channels
            self.delay = np.array(delay)
        else:
            self.delay = np.ones(self.num_channels) * delay

        self.idx = np.zeros(self.num_channels)
        self.idx -= self.delay

    def get_samples(self, num_samples):
        """Return pulse train samples."""
        sig = np.zeros((self.num_channels, num_samples))

        for i in range(num_samples):
            add_pulse = np.mod(self.idx, self.period_len) == 0
            sig[add_pulse, i] = self.amplitude[add_pulse]
            self.idx += 1
        return sig


def load_audio_file(file_name, desired_sr=None, num_channels=None, verbose=False):
    """Load an audio file and optionally resample."""
    audio, audio_sr = sf.read(file_name)
    assert audio.ndim == 2
    audio = audio.T
    # raise NotImplementedError("Check which axis is audio and which is channel. \
    #                            Make support for multiple channels (should be simple)")
    if num_channels is not None:
        audio = audio[:num_channels, :]
    num_channels = audio.shape[0]

    if desired_sr is None:
        return audio

    if desired_sr != audio_sr:
        assert audio_sr > desired_sr
        if audio_sr % desired_sr == 0:
            down_factor = audio_sr // desired_sr
            up_factor = 1
        else:
            up_factor, down_factor = coreutil.simplify_ratio(desired_sr, audio_sr)
        audio_downsampled = [
            spsig.resample_poly(audio[ch, :], up=up_factor, down=down_factor, axis=-1)
            for ch in range(num_channels)
        ]
        audio = np.stack(audio_downsampled, axis=0)
        # srAudio = srAudio // downsamplingFactor
        if verbose:
            print(
                f"AudioFileSource downsampled audio file from \
                    {audio_sr} to {audio_sr // down_factor} Hz"
            )

    return audio


def ar_coeffs_from_autocorr(autocorr):
    """Return the parameter vector $a = [a_1, ..., a_p]$.

    This implements the AR system
        that implements the AR system
        y(n) = a_1 y(n-1) + a_2 y(n-2) + ... + a_p y(n-p) + s(n)

        where s(n) is a white noise process with variance awgn_power

    """
    assert autocorr.ndim <= 2
    if autocorr.ndim == 2:
        assert autocorr.shape[0] == 1
        autocorr = autocorr[0, :]
    ar_coeffs = splin.solve_toeplitz(autocorr[:-1], autocorr[1:])
    noise_power = autocorr[0] - np.sum(autocorr[1:] * ar_coeffs)
    return ar_coeffs, noise_power


def ar_coeffs_to_transfer_function(ar_coeffs):
    """Return the rational transfer function for an AR process.

    This gives rise to
        the provided AR process, if a white noise is filtered
        through the returned filter.

    According to scipy.signal.lfilter which implements the system
        a_0 y(n) = b_0 x(n) + b_1 x(n-1) + ... + b_t x(n-t)
                    -a_1 y(n-1) - a_2 y(n-2) - ... - a_p y(n-p)
    """
    denom_coeffs = np.concatenate(([1], -ar_coeffs))
    num_coeffs = np.array([1], dtype=float)
    return num_coeffs, denom_coeffs


class AutocorrSource(Source):
    """Generate a signal with a specified autocorrelation function.

    All channels are all independent.
    Could be extended in the future to accept autocorr of shape
        (num_channels, num_channels, corr_len) which could
        include cross-correlations

    """

    def __init__(self, num_channels, autocorr, rng=None):
        """Create a source with the specified autocorrelation function.

        Parameters
        ----------
        num_channels : int
            Number of channels in the generated signal.
        autocorr : array of shape (num_channels, corr_len)
            with autocorr[:,0] being lag 0.
            This corresponds to a diagonal autocorrelation, where each channel is independent, but
            not white.
        rng : np.random.Generator
        """
        super().__init__(num_channels, rng)
        assert autocorr.shape[0] == num_channels

        assert cr.is_autocorr_func(autocorr)
        num_coeffs = []
        denom_coeffs = []
        self.noise_power = np.zeros(self.num_channels)
        for ch_idx in range(self.num_channels):
            ar_coeffs, self.noise_power[ch_idx] = ar_coeffs_from_autocorr(
                autocorr[ch_idx : ch_idx + 1, :]
            )
            nc, dc = ar_coeffs_to_transfer_function(ar_coeffs)
            num_coeffs.append(nc)
            denom_coeffs.append(dc)

        self.filt = fc.IIRFilter(num_coeffs, denom_coeffs)
        self.prepare()

    def prepare(self):
        """Prime filter state with initial noise."""
        for ch_idx in range(self.num_channels):
            self.filt.process(
                self.rng.normal(
                    loc=0,
                    scale=np.sqrt(self.noise_power[ch_idx]),
                    size=(self.num_channels, self.filt.order[ch_idx]),
                )
            )

    def get_samples(self, num_samples):
        """Return samples from the autocorrelation source."""
        return self.filt.process(
            self.rng.normal(
                loc=0,
                scale=np.sqrt(self.noise_power)[:, None],
                size=(self.num_channels, num_samples),
            )
        )


class MLS(Source):
    """Generate a maximum length sequence."""

    def __init__(self, num_channels, order, polynomial, state=None):
        super().__init__(num_channels, None)
        if num_channels > 1:
            raise NotImplementedError
        self.order = order
        self.poly = polynomial
        self.state = None

        self.metadata["order"] = self.order
        self.metadata["polynomial"] = self.poly
        self.metadata["start state"] = self.state

    def get_samples(self, num_samples):
        """Return MLS samples."""
        seq, self.state = spsig.max_len_seq(
            self.order, state=self.state, length=num_samples, taps=self.poly
        )
        return seq * 2 - 1


class GoldSequenceSource(Source):
    """Generate a Gold sequence signal."""

    preferred_sequences = {
        5: [[2], [1, 2, 3]],
        6: [[5], [1, 4, 5]],
        7: [[4], [4, 5, 6]],
        8: [[1, 2, 3, 6, 7], [1, 2, 7]],
        9: [[5], [3, 5, 6]],
        10: [[2, 5, 9], [3, 4, 6, 8, 9]],
        11: [[9], [3, 6, 9]],
    }

    def __init__(self, num_channels, power, order):
        super().__init__(num_channels)
        assert num_channels <= 2**order - 1
        self.num_channels = num_channels
        self.seq_length = 2**order - 1
        self.idx = 0
        self.set_power(power)

        if isinstance(self.power, (np.ndarray)):
            self.metadata["power"] = self.power.tolist()
        else:
            self.metadata["power"] = self.power
        self.metadata["order"] = order
        self.metadata["sequence length"] = self.seq_length

        if 5 <= order <= 11:
            mls1 = MLS(1, order, self.preferred_sequences[order][0])
            mls2 = MLS(1, order, self.preferred_sequences[order][1])
            seq1 = mls1.get_samples(2**order - 1)
            seq2 = mls2.get_samples(2**order - 1)
        elif order > 11:
            assert order % 2 == 1
            mls1 = MLS(1, order, None)
            seq1 = mls1.get_samples(2**order - 1)
            k = coreutil.get_smallest_coprime(order)
            decimation_factor = int(2**k + 1)
            decimated_indices = (
                np.arange(self.seq_length) * decimation_factor
            ) % self.seq_length
            seq2 = np.copy(seq1[decimated_indices])

        self.sequences = np.zeros((num_channels, self.seq_length))
        for i in range(num_channels):
            self.sequences[i, :] = np.negative(
                seq1 * np.roll(seq2, i)
            )  # XOR with integer shifts

    def get_samples(self, num_samples):
        """Return Gold sequence samples."""
        out_signal = np.zeros((self.num_channels, num_samples))
        block_lengths = util.calc_block_sizes(num_samples, self.idx, self.seq_length)
        out_idx = 0
        for block_len in block_lengths:
            out_signal[:, out_idx : out_idx + block_len] = self.sequences[
                :, self.idx : self.idx + block_len
            ]
            out_idx += block_len
            self.idx = (self.idx + block_len) % self.seq_length
        return out_signal * self.amplitude

    def set_power(self, new_power):
        """Set the target sequence power."""
        if isinstance(new_power, (int, float)) or new_power.ndim == 0:
            self.power = np.ones((1, 1)) * new_power
        elif new_power.ndim == 1:
            self.power = new_power[:, None]
        elif new_power.ndim == 2:
            assert new_power.shape[-1] == 1
            self.power = new_power
        else:
            raise ValueError
        self.amplitude = np.sqrt(self.power)


class LinearChirpSource(Source):
    """Generate a linear chirp signal."""

    def __init__(
        self,
        num_channels,
        amplitude,
        freq_range,
        samples_to_sweep,
        samplerate,
        rng=None,
    ):
        super().__init__(num_channels, rng)
        assert len(freq_range) == 2
        self.amp = amplitude
        self.samplerate = samplerate
        self.freq_range = freq_range

        self.samples_to_sweep = samples_to_sweep
        self.delta_freq = (freq_range[1] - freq_range[0]) / samples_to_sweep
        self.freq = freq_range[0]
        self.freq_counter = 0
        self.phase = self.rng.uniform(low=0, high=2 * np.pi)

    def next_phase(self):
        """Advance the chirp phase."""
        self.phase += (self.freq / self.samplerate) + (
            self.delta_freq / (2 * self.samplerate)
        )
        self.phase = self.phase % 1
        self.freq = self.freq + self.delta_freq

    def get_samples(self, num_samples):
        """Return chirp samples."""
        noise = np.zeros((1, num_samples))
        samples_left = num_samples
        n = 0
        for blocks in range(1 + num_samples // self.samples_to_sweep):
            block_size = min(samples_left, self.samples_to_sweep - self.freq_counter)
            for _ in range(block_size):
                noise[0, n] = np.cos(2 * np.pi * self.phase)
                self.next_phase()

                n += 1
                self.freq_counter += 1

                if self.samples_to_sweep <= self.freq_counter:
                    self.delta_freq = -self.delta_freq
                    self.freq_counter = 0
            samples_left -= block_size
        noise *= self.amp
        return noise
