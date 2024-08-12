import os
import random
import numpy as np
import scipy.io.wavfile as wavfile
import numpy as np
import scipy.signal as signal
from sounddesign import generate_beep, random_line_slide, fm_synthesis, apply_low_pass_filter, apply_distortion, apply_delay, apply_reverb
#------------------------------------------------------------

# MANUAL IMPLEMENTATION 
# The implementation based on the guideline of creating R2D2 sound in Design Sound book by Andy Farnell
# The implementation is based on the following steps:
# 1. Random Line Slider with Noise
# 2. FM Synthesis
# 3. Apply Effects (if needed) - Low Pass Filter, Reverb, Distortion, etc.
# 3. Complete Patch
# 6. Write to WAV file

def generate_r2d2_sound(length=10, sample_rate=16000, amplitude_scale=0.5, cutoff_frequency=5000):
    """Generates an R2-D2-like sound using Python.

    Args:
        length (float, optional): Length of the audio in seconds. Defaults to 5.

    Returns:
        numpy.ndarray: The generated audio signal.
    """
    t = np.linspace(0, length, int(length * sample_rate), False)

    # Generate beep parameters
    lower_num= int(length * 10)
    upper_num = int(length * 20)
    
    num_beeps = random.randint(lower_num, upper_num)  # Adjust number of beeps as needed
    beep_frequencies = np.random.uniform(1000, 5000, num_beeps)
    beep_durations = np.random.uniform(0.05, 0.5, num_beeps)
    beep_start_times = np.random.uniform(0, length - max(beep_durations), num_beeps)

    # Apply Random Line Slider
    carrier_freq = random_line_slide(length=length, sample_rate=sample_rate)  # adjusted to influence the overall pitch of the sound
    modulation_index = random_line_slide(length=length, sample_rate=sample_rate)
    modulator_freq = random_line_slide(length=length, sample_rate=sample_rate)

    # add random line slide to patch amplitude
    amplitude = random_line_slide(length, sample_rate) * amplitude_scale  # Scale amplitude
    
    # Apply FM synthesis
    audio = fm_synthesis(amplitude, carrier_freq, modulation_index, modulator_freq, length, sample_rate)

    # Apply filtering or other effects
    # audio = low_pass_filter(audio, cutoff_frequency)
    # Add beeps
    for i in range(num_beeps):
        beep = generate_beep(beep_durations[i], sample_rate, beep_frequencies[i])
        start_index = int(beep_start_times[i] * sample_rate)
        end_index = start_index + len(beep)
        audio[start_index:end_index] += beep
    

    
    # Normalize and convert to integer
    audio /= np.max(np.abs(audio))
    audio = np.int16(audio * 32767)

    return audio


# # Generate single R2-D2 sound
# audio = generate_r2d2_sound()
# wavfile.write("r2d2_sound.wav", 16000, audio)

# create a function that generate multiple R2D2 sound in different length and save it to a folder
def generate_r2d2_dataset(num_audios=300, output_path="../datasets/r2d2"):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for i in range(num_audios):
        length = np.random.uniform(15, 45)
        audio = generate_r2d2_sound(length=length)

        wavfile.write(f"{output_path}/r2d2_{i}.wav", 16000, audio)
    print("R2D2 dataset created!")

if __name__ == "__main__":
    generate_r2d2_dataset()