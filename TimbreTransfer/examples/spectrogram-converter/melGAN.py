class AudioCodec(object):
  """Base class for audio codec that encodes features and decodes to audio."""

  name: str
  n_dims: int
  sample_rate: int
  hop_size: int
  min_value: float
  max_value: float
  pad_value: float
  additional_frames_for_encoding: int = 0

  @property
  def abbrev_str(self):
    return self.name

  @property
  def frame_rate(self):
    return int(self.sample_rate // self.hop_size)

  def scale_features(self, features, output_range=(-1.0, 1.0), clip=False):
    """Linearly scale features to network outputs range."""
    min_out, max_out = output_range
    if clip:
      features = np.clip(features, self.min_value, self.max_value)
    # Scale to [0, 1].
    zero_one = (features - self.min_value) / (self.max_value - self.min_value)
    # Scale to [min_out, max_out].
    return zero_one * (max_out - min_out) + min_out

  def scale_to_features(self, outputs, input_range=(-1.0, 1.0), clip=False):
    """Invert by linearly scaling network outputs to features range."""
    min_out, max_out = input_range
    outputs = np.clip(outputs, min_out, max_out) if clip else outputs
    # Scale to [0, 1].
    zero_one = (outputs - min_out) / (max_out - min_out)
    # Scale to [self.min_value, self.max_value].
    return zero_one * (self.max_value - self.min_value) + self.min_value

  def encode(self, audio):
    """Encodes audio to features.
    Args:
      audio: A NumPy array of shape (batch_size, num_samples) representing audio.
    Returns:
      A NumPy array representing the encoded features.
    """
    raise NotImplementedError

  def decode(self, features):
    """Decodes features to audio.
    Args:
      features: A NumPy array representing the encoded features.
    Returns:
      A NumPy array of shape (batch_size, num_samples) representing audio.
    """
    raise NotImplementedError

  def to_images(self, features):
    """Maps a batch of features to images for visualization.
    Args:
      features: A NumPy array of shape (batch_size, n_frames, n_dims) representing features.
    Returns:
      A NumPy array of shape (batch_size, n_frames, n_dims) representing the features scaled to the range
        [0.0, 1.0] for visualization.
    """
    assert features.ndim == 3
    return self.scale_features(features, output_range=(0.0, 1.0))

  @property
  def context_codec(self):
    """Codec for encoding audio context."""
    return self

#------------------------------------------------------------
class MelGAN(AudioCodec):
  """Invertible Mel Spectrogram with 128 dims and 16kHz."""

  name = 'melgan'
  n_dims = N_DIMS
  sample_rate = SAMPLE_RATE
  hop_size = HOP_LENGTH
  min_value = np.log(1e-5)  # Matches MelGAN training.
  max_value = 4.0  # Largest value for most examples.
  pad_value = np.log(1e-5)  # Matches MelGAN training.
  # 16 extra frames are needed to avoid numerical errors during the mel bin
  # matmul.
  # The numerical errors are small, but enough to produce audible pops when
  # decoded by MelGan.
  additional_frames_for_encoding = 16

  def __init__(self, decode_dither_amount: float = 0.0):
    self._frame_length = 640
    self._fft_size = 1024
    self._lo_hz = 0.0
    self._decode_dither_amount = decode_dither_amount

  def encode(self, audio):
    """Compute features from audio.
    Args:
      audio: Shape [batch, n_samples].
    Returns:
      mel_spectrograms: Shape [batch, n_samples // hop_size, n_dims].
    """
    if tf.shape(audio)[0] == 0:
      # The STFT code doesn't always handle 0-length inputs correctly.
      # If we know the output is 0-length, just use this hard coded response.
      return tf.zeros((0, self.n_dims), dtype=tf.float32)
    return Audio2Mel(
        sample_rate=self.sample_rate,
        hop_length=self.hop_size,
        win_length=self._frame_length,
        n_fft=self._fft_size,
        n_mel_channels=self.n_dims,
        drop_dc=True,
        mel_fmin=self._lo_hz,
        mel_fmax=int(self.sample_rate // 2))(audio)

  def decode(self, features):
    """Decodes features to audio.

    Args:
      features: Mel spectrograms, shape [batch, n_frames, n_dims].

    Returns:
      audio: Shape [batch, n_frames * hop_size]
    """
    model = _load_model_from_cache('melgan')

    if self._decode_dither_amount > 0:
      features += (
          np.random.normal(size=features.shape) * self._decode_dither_amount)

    return model(features).numpy()  # pylint: disable=not-callable


#----------------------------------------------------------------------