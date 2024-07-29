# TODO: This file contains all the configuration parameters for the model that we used for the timbre transfer project.abs

## Audio
sample_rate = 16000
# Number of spectrogram frames in a partial utterance
partials_n_frames = 160     # 1600 ms
# Number of spectrogram frames at inference
inference_n_frames = 80     #  800 ms

## Mel-spectrogram
# n_fft = 2048
n_fft = 1024
num_mels = 128
num_samples = 128 # input spect shape num_mels * num_samples

hop_length = int(0.0125*sample_rate)                    # 12.5ms - in line with Tacotron 2 paper
win_length = int(0.05*sample_rate)                   # 50ms - same reason as above
# HOP_LENGTH = 320
# WIN_LENGTH = 640 # (20 ms)

mel_fmin = 0.0
mel_fmax = int(sample_rate // 2)

fmin = 40
min_level_db = -100
ref_level_db = 20
bits = 9                            # bit depth of signal
mu_law = True                       # Recommended to suppress noise if using raw bits in hp.voc_mode below
peak_norm = False

# CLIP_VALUE_MIN = 1e-5
# CLIP_VALUE_MAX = 1e8
# N_IMG_CHANNELS = 1 #3

#----------------------------------------------------------
# Data
dataset_repetitions = 5
num_epochs = 1  # train for at least 50 epochs for good results
mel_spec_size = (128, 128)

#----------------------------------------------------------
# Model Config
# architecture
embedding_dims = 32
embedding_max_frequency = 1000.0
widths = [32, 64, 96, 128]
block_depth = 2


# optimization
batch_size = 64
ema = 0.999
learning_rate =  2e-5
weight_decay = 1e-4

# OLD PARAMS
# widths = [64, 128, 256, 512]
# has_attention = [False, False, True, True]
# block_depth = 4
# batch_size = 16


#----------------------------------------------------------
#Duration
duration_sample = 40960 #*2 if 256
duration_track = 480000

