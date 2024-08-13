# TODO: THIS FILE IS USED FOR THE PREPROCESSING PHASE OF THE CHOSEN DATASEt
# So it can input to the model

# Calculate the spectrogram of the audio file
# and Reshape the spectrogram to (128, 128, 1) - 1 channel of image 128x128
# The expected output of this file is two numpy arrays corresponding to two dataset: target timbre and condition timbre, respectively
# The target timbre is the timbre that we want to convert the condition timbre to
# The condition timbre is the timbre that we want to convert to the target timbre
# The two numpy arrays have the shape (number of samples, 128, 128, 1)

import tensorflow as tf
import numpy as np
import os
import tensorflow_io as tfio
from tqdm import tqdm

import params.audio_params as params #parameters for audio processing
import params.model_params as model_params #parameters for the model
from utils.audio_utils import calculate_spectrogram, normalize_audio, read_audio #functions to process audio files
# from modules.wav2spec import Audio2Mel #class to convert audio to mel spectrogram

def preprocess_dataset(dataset_path=params.DATASET_PATH, target_timbre='r2d2', condition_timbre='vn'):
    """
    Preprocess the dataset
    :param dataset_path: The path to the dataset
    :param target_timbre: The name of target audio for transfering timbre
    :param condition_timbre: The name of condition audio for receiving timbre
    """
    # Get the list of audio files in the dataset
    target_audios = tf.io.gfile.glob(dataset_path + target_timbre + '/*.wav') #target audios of R2D2
    condition_audios = tf.io.gfile.glob(dataset_path + condition_timbre + '/*.wav') #condition audios of Violin

    # # Initialize the Audio2Mel class
    # audio2mel = Audio2Mel()
    
    # FIXME: CHECK THE SIZE OF THE OUTPUT SPECTROGRAM
    target_spectrograms = np.zeros((len(target_audios), params.MEL_SPEC_HEIGHT, params.MEL_SPEC_WIDTH, model_params.TGT_IMG_CHANNELS))
    condition_spectrograms = np.zeros((len(condition_audios), params.MEL_SPEC_HEIGHT, params.MEL_SPEC_WIDTH, model_params.TGT_IMG_CHANNELS))

    # Preprocess the target timbre
    for i, audio_file in enumerate(tqdm(target_audios, desc="Preprocessing target timbre...")):
        audio = read_audio(audio_file)
        audio = normalize_audio(audio)

        # Calculate the spectrogram
        spectrogram = calculate_spectrogram(audio) #expected original shape (128, 512)

        # Reshape the spectrogram
        target_spectrograms[i] = spectrogram.reshape(128, 128, 1)
    
    # Preprocess the condition timbre
    for i, audio_file in enumerate(tqdm(condition_audios, desc="Preprocessing condition timbre...")):
        # Read the audio file
        audio = read_audio(audio_file)
        audio = normalize_audio(audio)

        # Calculate the spectrogram
        spectrogram = calculate_spectrogram(audio)
        # Reshape the spectrogram
        condition_spectrograms[i] = spectrogram.reshape(128, 128, 1)

    return target_spectrograms, condition_spectrograms



