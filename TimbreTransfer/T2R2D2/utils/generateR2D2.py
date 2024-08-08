import numpy as np
import scipy.io.wavfile as wavfile
import numpy as np
import scipy.signal as signal
from sounddesign import random_line_slide, fm_synthesis, apply_low_pass_filter, apply_distortion, apply_delay, apply_reverb
#------------------------------------------------------------

# MANUAL IMPLEMENTATION 
# The implementation based on the guideline of creating R2D2 sound in Design Sound book by Andy Farnell
# The implementation is based on the following steps:
# 1. Random Line Slider with Noise
# 2. FM Synthesis
# 3. Apply Effects (if needed) - Low Pass Filter, Reverb, Distortion, etc.
# 3. Complete Patch
# 6. Write to WAV file

def generate_r2d2_sound(length=3, sample_rate=16000, amplitude_scale=0.5, cutoff_frequency=5000):
    """Generates an R2-D2-like sound using Python.

    Args:
        length (float, optional): Length of the audio in seconds. Defaults to 5.

    Returns:
        numpy.ndarray: The generated audio signal.
    """
    t = np.linspace(0, length, int(length * sample_rate), False)

    # Apply Random Line Slider
    carrier_freq = random_line_slide(length=length, sample_rate=sample_rate)  # adjusted to influence the overall pitch of the sound
    modulation_index = random_line_slide(length=length, sample_rate=sample_rate)
    modulator_freq = random_line_slide(length=length, sample_rate=sample_rate)

    # add random line slide to patch amplitude
    amplitude = random_line_slide(length, sample_rate) * amplitude_scale  # Scale amplitude

    # Apply FM synthesis
    audio = fm_synthesis(carrier_freq, modulation_index, modulator_freq, length, sample_rate)

    # Apply filtering or other effects
    # audio = low_pass_filter(audio, cutoff_frequency)

    # Normalize and convert to integer
    audio /= np.max(np.abs(audio))
    audio = np.int16(audio * 32767)

    return audio


# Generate R2-D2 sound
audio = generate_r2d2_sound()
wavfile.write("r2d2_sound.wav", 44100, audio)

