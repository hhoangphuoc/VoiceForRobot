import numpy as np
import scipy.io.wavfile as wavfile
import random
import scipy.signal as signal

def generate_random_parameters(sound_type):
    parameters = {}
    
    if sound_type == "vocalization":
        parameters["frequency"] = random.uniform(1000, 5000)
        parameters["amplitude"] = random.uniform(0.3, 0.7)
        parameters["filter_cutoff"] = random.uniform(2000, 8000)
        parameters["envelope_attack"] = random.uniform(0.01, 0.05)
        parameters["envelope_decay"] = random.uniform(0.05, 0.2)
        parameters["envelope_sustain"] = random.uniform(0.3, 0.7)
        parameters["envelope_release"] = random.uniform(0.2, 0.5)
    elif sound_type == "mechanical":
        frequency = random.uniform(100, 500)
        parameters["amplitude"] = random.uniform(0.2, 0.5)
        parameters["filter_cutoff"] = random.uniform(500, 2000)
        parameters["envelope_attack"] = random.uniform(0.001, 0.02)
        parameters["envelope_decay"] = random.uniform(0.02, 0.1)
        parameters["envelope_sustain"] = random.uniform(0.1, 0.3)
        parameters["envelope_release"] = random.uniform(0.1, 0.3)
    elif sound_type == "whistle":
        parameters["frequency"] = np.random.normal(3000, 500)  # Gaussian distribution centered at 3000 Hz
        parameters["amplitude"] = np.random.uniform(0.5, 0.8)
        parameters["filter_cutoff"] = np.random.uniform(3000, 10000)
        parameters["envelope_attack"] = np.random.uniform(0.01, 0.03)
        parameters["envelope_decay"] = np.random.uniform(0.05, 0.1)
        parameters["envelope_sustain"] = np.random.uniform(0.2, 0.4)
        parameters["envelope_release"] = np.random.uniform(0.1, 0.2)
    elif sound_type == "beep":
        parameters["frequency"] = np.random.uniform(1000, 2000)
        parameters["amplitude"] = np.random.uniform(0.8, 1.0)
        parameters["filter_cutoff"] = np.random.uniform(2000, 5000)
        parameters["envelope_attack"] = np.random.uniform(0.001, 0.01)
        parameters["envelope_decay"] = np.random.uniform(0.01, 0.03)
        parameters["envelope_sustain"] = 0.0  # Short sustain for a sharp beep
        parameters["envelope_release"] = np.random.uniform(0.05, 0.1)

    else:
        raise ValueError("Invalid sound type")

    return parameters

def apply_filter(audio, sample_rate, cutoff):
    nyquist_freq = sample_rate / 2
    normal_cutoff = cutoff / nyquist_freq
    b, a = signal.butter(4, normal_cutoff, btype='low', fs=sample_rate)
    return signal.filtfilt(b, a, audio) #return filtered audio

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

def generate_r2d2_sound(sound_type, length):
    sample_rate = 44100
    t = np.linspace(0, length, int(length * sample_rate), False)

    parameters = generate_random_parameters(sound_type)

    # Generate basic waveform (e.g., sine)
    frequency = parameters["frequency"]
    amplitude = parameters["amplitude"]
    audio = amplitude * np.sin(2 * np.pi * frequency * t)

    # Apply filtering (example)
    filter_cutoff = parameters["filter_cutoff"]
    # Implement filtering using scipy
    audio = apply_filter(audio, sample_rate, filter_cutoff)

    # Apply envelope (example)
    attack = parameters["envelope_attack"]
    decay = parameters["envelope_decay"]
    sustain = parameters["envelope_sustain"]
    release = parameters["envelope_release"]
    # Implement envelope using numpy
    audio = apply_envelope(audio, length, sample_rate, attack, decay, sustain, release)

    # Apply other sound design techniques (e.g., distortion, modulation)
    # ...

    # Normalize audio
    audio /= np.max(np.abs(audio))

    # Convert to integer format
    audio_int = np.int16(audio * 32767)

    # Write to WAV file
    wavfile.write(f"r2d2_sound_{sound_type}_{random.randint(1, 10000)}.wav", sample_rate, audio_int)

