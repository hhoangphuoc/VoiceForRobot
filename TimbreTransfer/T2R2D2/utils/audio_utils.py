#-----------------------------------------
# Audio Utils
# Including functions that processing with audio files
#-----------------------------------------

import os
import tensorflow_io as tfio

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import params.audio_params as params

module = hub.KerasLayer('https://www.kaggle.com/models/google/soundstream/TensorFlow2/mel-decoder-music/1')

MEL_BASIS = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins=params.N_MEL_CHANNELS,
    num_spectrogram_bins=params.N_FFT // 2 + 1,
    sample_rate=params.SAMPLE_RATE,
    lower_edge_hertz=params.MEL_FMIN,
    upper_edge_hertz=params.MEL_FMAX)

def calculate_spectrogram(samples):
  """Calculate mel spectrogram using the parameters the model expects."""
  fft = tf.signal.stft(
      samples,
      frame_length=params.WIN_LENGTH,
      frame_step=params.HOP_LENGTH,
      fft_length=params.N_FFT,
      window_fn=tf.signal.hann_window,
      pad_end=True)
  fft_modulus = tf.abs(fft)

  output = tf.matmul(fft_modulus, MEL_BASIS)

  output = tf.clip_by_value(
      output,
      clip_value_min=params.CLIP_VALUE_MIN,
      clip_value_max=params.CLIP_VALUE_MAX)
  output = tf.math.log(output)
  return output

#-------------------------------------------------------------------------


def normalize_audio(audio):
    """
    Normalize the waveform to be between -1 and 1
    """
    audio = audio - audio.min()
    audio = audio/audio.max()
    audio = (audio*2)-1
    return audio


def read_audio(audio_path,resample=True, sample_rate=params.SAMPLE_RATE):
    audio_bin = tf.io.read_file(audio_path)
    audio, sample_rate = tf.audio.decode_wav(audio_bin)
    if resample:
        audio = tfio.audio.resample(audio, rate_in=tf.cast(sample_rate,tf.int64), rate_out=SAMPLE_RATE, name=None)
    return audio

def save_audio(audio, path, sample_rate=params.SAMPLE_RATE):
    audio = tf.cast(audio, tf.float32)
    audio = tfio.audio.resample(audio, rate_in=sample_rate, rate_out=sample_rate, name=None)
    audio = tf.cast(audio, tf.int16)
    audio = tf.audio.encode_wav(audio, sample_rate)
    tf.io.write_file(path, audio)

def plot_audio(audio):
    """
    Plot the audio in the waveform
    """
    plt.plot(audio)
    plt.show()

def plot_spectrogram(spectrogram):
    plt.imshow(spectrogram, aspect='auto', origin='lower')
    plt.colorbar()
    plt.show()

def plot_spectrogram_from_audio(audio):
    """
    Plot the spectrogram of the audio
    """
    spectrogram = calculate_spectrogram(audio)
    plot_spectrogram(spectrogram)

    