# import pd
# import random
# import time
# import sounddevice as sd
# import numpy as np

# # Connect to PureData
# pd = pd.Pd()

# # PureData patch parameters
# frequency_address = "/frequency"
# amplitude_address = "/amplitude"
# filter_cutoff_address = "/filter_cutoff"
# envelope_attack_address = "/envelope_attack"
# envelope_decay_address = "/envelope_decay"
# envelope_sustain_address = "/envelope_sustain"
# envelope_release_address = "/envelope_release"
# # ... other parameters

# # def generate_random_parameters(sound_type):
# #     parameters = {}

# #     # ... (previous code for generating random parameters)


# #     return parameters
# def generate_random_parameters(sound_type):
#     parameters = {}
    
#     if sound_type == "vocalization":
#         parameters["frequency"] = random.uniform(1000, 5000)
#         parameters["amplitude"] = random.uniform(0.3, 0.7)
#         parameters["filter_cutoff"] = random.uniform(2000, 8000)
#         parameters["envelope_attack"] = random.uniform(0.01, 0.05)
#         parameters["envelope_decay"] = random.uniform(0.05, 0.2)
#         parameters["envelope_sustain"] = random.uniform(0.3, 0.7)
#         parameters["envelope_release"] = random.uniform(0.2, 0.5)
#     elif sound_type == "mechanical":
#         frequency = random.uniform(100, 500)
#         parameters["amplitude"] = random.uniform(0.2, 0.5)
#         parameters["filter_cutoff"] = random.uniform(500, 2000)
#         parameters["envelope_attack"] = random.uniform(0.001, 0.02)
#         parameters["envelope_decay"] = random.uniform(0.02, 0.1)
#         parameters["envelope_sustain"] = random.uniform(0.1, 0.3)
#         parameters["envelope_release"] = random.uniform(0.1, 0.3)
#     elif sound_type == "whistle":
#         parameters["frequency"] = np.random.normal(3000, 500)  # Gaussian distribution centered at 3000 Hz
#         parameters["amplitude"] = np.random.uniform(0.5, 0.8)
#         parameters["filter_cutoff"] = np.random.uniform(3000, 10000)
#         parameters["envelope_attack"] = np.random.uniform(0.01, 0.03)
#         parameters["envelope_decay"] = np.random.uniform(0.05, 0.1)
#         parameters["envelope_sustain"] = np.random.uniform(0.2, 0.4)
#         parameters["envelope_release"] = np.random.uniform(0.1, 0.2)
#     elif sound_type == "beep":
#         parameters["frequency"] = np.random.uniform(1000, 2000)
#         parameters["amplitude"] = np.random.uniform(0.8, 1.0)
#         parameters["filter_cutoff"] = np.random.uniform(2000, 5000)
#         parameters["envelope_attack"] = np.random.uniform(0.001, 0.01)
#         parameters["envelope_decay"] = np.random.uniform(0.01, 0.03)
#         parameters["envelope_sustain"] = 0.0  # Short sustain for a sharp beep
#         parameters["envelope_release"] = np.random.uniform(0.05, 0.1)
#     # elif sound_type == "motor":
#     #     parameters["frequency"] = np.random.uniform(100, 300)
#     #     parameters["amplitude"] = np.random.uniform(0.2, 0.4)
#     #     parameters["filter_cutoff"] = np.random.uniform(500, 1500)
#     #     parameters["envelope_attack"] = 0.0  # Immediate start
#     #     parameters["envelope_decay"] = np.random.uniform(0.1, 0.3)
#     #     parameters["envelope_sustain"] = 1.0  # Continuous sound
#     #     parameters["envelope_release"] = np.random.uniform(0.2, 0.5)
#     # elif sound_type == "hydraulic":
#     #     parameters["frequency"] = np.random.uniform(50, 200)
#     #     parameters["amplitude"] = np.random.uniform(0.3, 0.6)
#     #     parameters["filter_cutoff"] = np.random.uniform(300, 1000)
#     #     parameters["envelope_attack"] = np.random.uniform(0.01, 0.05)
#     #     parameters["envelope_decay"] = np.random.uniform(0.1, 0.3)
#     #     parameters["envelope_sustain"] = 0.5  # Medium sustain
#     #     parameters["envelope_release"] = np.random.uniform(0.2, 0.4)
#     elif sound_type == "alarm":
#         parameters["frequency"] = np.random.uniform(1000, 3000)
#         parameters["amplitude"] = np.random.uniform(0.7, 1.0)
#         parameters["filter_cutoff"] = np.random.uniform(2000, 5000)
#         parameters["envelope_attack"] = 0.0  # Immediate start
#         parameters["envelope_decay"] = np.random.uniform(0.01, 0.03)
#         parameters["envelope_sustain"] = 1.0  # Continuous sound with variations
#         parameters["envelope_release"] = np.random.uniform(0.1, 0.2)

#     else:
#         raise ValueError("Invalid sound type")

#     return parameters

# def generate_r2d2_sound(sound_type, length):
#     parameters = generate_random_parameters(sound_type)

#     # Send parameters to PureData
#     pd.send_float(frequency_address, parameters["frequency"])
#     pd.send_float(amplitude_address, parameters["amplitude"])
#     pd.send_float(filter_cutoff_address, parameters["filter_cutoff"])
#     pd.send_float(envelope_attack_address, parameters["envelope_attack"])
#     pd.send_float(envelope_decay_address, parameters["envelope_decay"])
#     pd.send_float(envelope_sustain_address, parameters["envelope_sustain"])
#     pd.send_float(envelope_release_address, parameters["envelope_release"])
#     # ... send other parameters

#     # Trigger sound generation
#     pd.send_bang("/trigger_sound")

#     # Record audio (adjust parameters as needed)
#     myrecording = sd.rec(int(length * 44100), samplerate=44100, channels=2)
#     sd.wait()
#     soundfile.write(f"r2d2_sound_{random.randint(1, 10000)}_{sound_type}.wav", myrecording, 44100, 'float32')

# def main():
#     total_time = 5 * 60 * 60  # 5 hours in seconds
#     average_sound_length = 90  # Adjust average sound length
#     num_sounds = total_time / average_sound_length

#     sound_types = ["vocalization", "mechanical", "whistle", "beep", "alarm"]  # Add more types

#     for _ in range(int(num_sounds)):
#         sound_type = random.choice(sound_types)
#         length = random.uniform(60, 120)
#         generate_r2d2_sound(sound_type, length)

# if __name__ == "__main__":
#     main()