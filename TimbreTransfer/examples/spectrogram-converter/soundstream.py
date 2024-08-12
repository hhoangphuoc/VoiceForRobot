import math
import tensorflow as tf
# import tensorflow_datasets as tfds
import tensorflow_hub as hub

module = hub.KerasLayer('https://www.kaggle.com/models/google/soundstream/TensorFlow2/mel-decoder-music/1')

SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 320
WIN_LENGTH = 640
N_MEL_CHANNELS = 128
MEL_FMIN = 0.0
MEL_FMAX = int(SAMPLE_RATE // 2)
CLIP_VALUE_MIN = 1e-5
CLIP_VALUE_MAX = 1e8

MEL_BASIS = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins=N_MEL_CHANNELS,
    num_spectrogram_bins=N_FFT // 2 + 1,
    sample_rate=SAMPLE_RATE,
    lower_edge_hertz=MEL_FMIN,
    upper_edge_hertz=MEL_FMAX)

def calculate_spectrogram(samples):
  """Calculate mel spectrogram using the parameters the model expects."""
  fft = tf.signal.stft(
      samples,
      frame_length=WIN_LENGTH,
      frame_step=HOP_LENGTH,
      fft_length=N_FFT,
      window_fn=tf.signal.hann_window,
      pad_end=True)
  fft_modulus = tf.abs(fft)

  output = tf.matmul(fft_modulus, MEL_BASIS)

  output = tf.clip_by_value(
      output,
      clip_value_min=CLIP_VALUE_MIN,
      clip_value_max=CLIP_VALUE_MAX)
  output = tf.math.log(output)
  return output