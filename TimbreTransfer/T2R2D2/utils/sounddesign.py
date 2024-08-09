import numpy as np
import scipy.io.wavfile as wavfile
import random
import scipy.signal as signal
import librosa

def random_line_slide(length, sample_rate=16000):
    """Generates a random line slider with noise.

    Args:
        length (float): Length of the slider in seconds.

    Returns:
        numpy.ndarray: The generated slider.
    """
    t = np.linspace(0, length, int(length * sample_rate), False)
    noise = np.random.randn(len(t)) * 0.1  # Adjust noise level

    # Generate random line
    line = np.random.rand(2) * 2 - 1
    line = np.interp(t, [0, length], line)

    # Add noise
    slider = line + noise

    return slider

def fm_synthesis(amplitude, carrier_freq, modulation_index, modulator_freq, length, sample_rate=16000):
    """Generates an FM synthesis sound.

    Args:
        carrier_freq (numpy.ndarray): The carrier frequency slider.
        modulation_index (numpy.ndarray): The modulation index slider.
        modulator_freq (numpy.ndarray): The modulator frequency slider.
        length (float): Length of the audio in seconds.

    Returns:
        numpy.ndarray: The generated audio signal.
    """
    t = np.linspace(0, length, int(length * sample_rate), False)

    carrier = amplitude * np.sin(2 * np.pi * carrier_freq * t)
    modulator = np.sin(2 * np.pi * modulator_freq * t)
    # audio = np.sin(2 * np.pi * (carrier_freq + modulation_index * modulator) * t)
    modulated_signal = carrier * np.exp(1j * modulation_index * modulator_freq)
    audio = np.real(modulated_signal)
    return audio

def generate_beep(length, sample_rate=16000, beep_frequency=2000, beep_amplitude=2.5):
    """Generates a beep sound.

    Args:
    frequency: The frequency of the beep.
    duration: The duration of the beep in seconds.
    sample_rate: The sample rate of the audio.

    Returns:
    The generated beep sound.
    """

    t = np.linspace(0, length, int(length * sample_rate), False)
    beep = beep_amplitude * np.sin(2 * np.pi * beep_frequency * t)
    envelope = np.concatenate([
        np.linspace(0, 1, int(length * sample_rate * 0.1)),  # Attack
        np.linspace(1, 0, int(length * sample_rate * 0.1))  # Decay
    ])
    envelope = np.pad(envelope, (0, len(beep) - len(envelope)), mode="constant")
    beep *= envelope
    return beep
#------------------------------------------------------------




def apply_low_pass_filter(audio, sample_rate=16000, cutoff_frequency=5000):
    """Applies a low-pass filter to the audio signal.

    Args:
        audio (numpy.ndarray): The input audio signal.
        cutoff_frequency (float): The cutoff frequency of the filter.
        sample_rate (int, optional): The sample rate of the audio signal. Defaults to 44100.

    Returns:
        numpy.ndarray: The filtered audio signal.
    """
    b, a = signal.butter(4, cutoff_frequency / (sample_rate / 2), btype="low", fs=sample_rate)
    filtered_audio = signal.filtfilt(b, a, audio)
    return filtered_audio

def apply_distortion(audio, gain=1.0):
    """Apply distortion to the audio signal.

    Args:
        audio (numpy.ndarray): The input audio signal.
        gain (float, optional): The gain of the distortion effect. Defaults to 1.0.

    Returns:
        numpy.ndarray: The distorted audio signal.
    """
    distorted_audio = np.clip(audio * gain, -1, 1)
    return distorted_audio

def apply_reverb(audio, sample_rate=16000, decay_time=1.0, reverb_time=1.0, room_size=0.5, damping=0.5, wet_level=0.5, dry_level=0.5):
    """Apply reverb to the audio signal.

    Args:
        audio (numpy.ndarray): The input audio signal.
        sample_rate (int, optional): The sample rate of the audio signal. Defaults to 44100.
        room_size (float, optional): The size of the virtual room. Defaults to 0.5.
        damping (float, optional): The damping factor. Defaults to 0.5.
        wet_level (float, optional): The wet level of the reverb effect. Defaults to 0.5.
        dry_level (float, optional): The dry level of the reverb effect. Defaults to 0.5.

    Returns:
        numpy.ndarray: The reverberated audio signal.
    """
    audio = np.array(audio, dtype=np.float32)
    reverb = np.zeros_like(audio)
    delay = int(reverb_time * sample_rate)

    # Generate reverb
    for i in range(delay, len(audio)):
        reverb[i] = audio[i] + decay_time * audio[i - delay]

    # Mix dry and wet signals
    output = (1 - wet_level) * audio + wet_level * reverb
    return output

def apply_delay(audio, sample_rate=16000, delay_time=0.5, feedback=0.5):
    """Apply delay to the audio signal.

    Args:
        audio (numpy.ndarray): The input audio signal.
        sample_rate (int, optional): The sample rate of the audio signal. Defaults to 44100.
        delay_time (float, optional): The delay time in seconds. Defaults to 0.5.
        feedback (float, optional): The feedback gain. Defaults to 0.5.

    Returns:
        numpy.ndarray: The delayed audio signal.
    """
    audio = np.array(audio, dtype=np.float32)
    delay_samples = int(delay_time * sample_rate)
    delayed_audio = np.zeros_like(audio)

    # Apply delay
    for i in range(delay_samples, len(audio)):
        delayed_audio[i] = audio[i] + feedback * audio[i - delay_samples]

    return delayed_audio


def apply_envelope(audio, length, sample_rate, attack, decay, sustain, release):
    envelope_length = attack + decay + sustain * length + release
    envelope = np.zeros(int(envelope_length * sample_rate))

    # Attack phase
    envelope[:int(attack * sample_rate)] = np.linspace(0, 1, int(attack * sample_rate))
    # Decay phase
    envelope[int(attack * sample_rate):int((attack + decay) * sample_rate)] = np.linspace(1, sustain, int(decay * sample_rate))
    # Sustain phase
    envelope[int((attack + decay) * sample_rate):int((attack + decay + sustain * length) * sample_rate)] = sustain
    # Release phase
    envelope[int((attack + decay + sustain * length) * sample_rate):] = np.linspace(sustain, 0, int(release * sample_rate))

    # Apply envelope to audio
    audio *= envelope[:len(audio)]

    return audio

