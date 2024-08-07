import tensorflow as tf
# PARAMETERS

# Audio Config --------------------
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 320
WIN_LENGTH = 640 # (20 ms)

#-----------------------------------

# Mel Spectrogram Config
N_MEL_CHANNELS = 128
MEL_FMIN = 0.0
MEL_FMAX = int(SAMPLE_RATE // 2)
CLIP_VALUE_MIN = 1e-5
CLIP_VALUE_MAX = 1e8
N_IMG_CHANNELS = 1 #3

MEL_BASIS = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins=N_MEL_CHANNELS,
    num_spectrogram_bins=N_FFT // 2 + 1,
    sample_rate=SAMPLE_RATE,
    lower_edge_hertz=MEL_FMIN,
    upper_edge_hertz=MEL_FMAX)

#-----------------------------------


# Diffusion Model Config --------------------

# data
DATASET_REPETITIONS = 5
# num_epochs = 1  # train for at least 50 epochs for good results
NUM_EPOCHS = 50
MEL_SPEC_SIZE = (128, 128)

# KID = Kernel Inception Distance, see related section
KID_IMG_SIZE = 75
KID_DIFFUSION_STEPS = 10
PLOT_DIFFUSSION_STEPS = 20
plot_diffusion_steps = 20
# sampling
MIN_SIGNAL_RATE = 0.02
MAX_SIGNAL_RATE = 0.95

# Model Architecture
EMBEDDINGS_DIMS = 32
EMBEDDINGS_MAX_FREQ = 1000.0
EMBEDDINGS_WIDTHS = [32, 64, 96, 128]
BLOCK_DEPTH = 2

# optimization
BATCH_SIZE = 64
EMA = 0.999
LEARNING_RATE =  2e-5
WEIGHT_DECAY = 1e-4

# EMBEDDINGS_WIDTHS = [64, 128, 256, 512]
# HAS_ATTENTION = [False, False, True, True] # specified which block position has the attention layer
# BLOCK_DEPTH = 4
# BATCH_SIZE = 16

DURATION_SAMPLE = 40960 #*2 if 256
DURATION_TRACK = 480000

# New
#widths = [32,  64, 96,  128]
#block_depth = 2

N_IMG_CHANNELS = 1 #3 Numbers of channels in the image
COND_IMG_CHANNELS=2 # Number of channels in the conditioning image

# -----------------------------------

