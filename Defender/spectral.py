from functools import lru_cache

import librosa
import lws
import numpy as np
import tensorflow as tf

# from advoc.util import best_shape

def best_shape(t, axis=None):
  """Gets static shape if available, otherwise dynamic.

  Args:
    t: Tensor in question.
    axis: None if requesting entire shape, otherwise the axis in question.

  Returns:
    Python list containing (possibly a mixture of) ints or tf.Tensor.
  """
  if axis is None:
    ndims = t.get_shape().ndims
    if ndims is None:
      raise ValueError('Cannot run on tensor with dynamic ndims')
    dims = []
    for i in range(ndims):
      try:
        dim = int(t.get_shape()[i])
      except:
        dim = tf.shape(t)[i]
      dims.append(dim)
    return dims
  else:
    try:
      dim = int(t.get_shape()[axis])
    except:
      dim = tf.shape(t)[i]
    return dim

def stft(x, nfft, nhop, pad_end=True):
  """Performs the short-time Fourier transform on a waveform.

  Args:
    x: nd-array dtype float32 of shape [?, 1, 1].
    nfft: FFT size.
    nhop: Window size.
    pad_end: If true, pad incomplete frames at end of waveform.

  Returns:
    nd-array dtype complex128 of shape [?, (nfft // 2) + 1, 1] containing the features.
  """
  nsamps, nfeats, nch = x.shape
  if nfeats != 1:
    raise ValueError()
  if nch != 1:
    raise NotImplementedError('Can only take STFT of monaural signals')

  x = x[:, 0, 0]
  xlen = x.shape[0]

  if pad_end == True:
    num_frames = int(np.ceil(float(xlen) / nhop) + 1e-6)
    if num_frames > 0:
      last_frame_start = (num_frames - 1) * nhop
      last_frame_end = last_frame_start + nfft
      pad_amt = last_frame_end - xlen
      if pad_amt > 0:
        x = np.pad(x, [[0, pad_amt]], 'constant')

  return lws.lws(nfft, nhop, perfectrec=False).stft(x)[:, :, np.newaxis]


def lws_hann_default(nfft, nhop, dtype=tf.float32):
  """Constructs default LWS Hann window for parity between LWS/TF.

  Args:
    nfft: FFT size.
    nhop: Shift amount.
    dtype: Tensorflow datatype.

  Returns:
    Tensor dtype as specified of shape [nfft].
  """
  _hann = lws.hann(nfft, symmetric=True, use_offset=False)
  _awin = np.sqrt(_hann * 2 * nhop / nfft)
  return tf.constant(_awin, dtype=dtype)


def stft_tf(x, nfft, nhop, pad_end=True):
  """Constructs graph for short-time Fourier transform.

  Args:
    x: Tensor dtype float32 of shape [b, nsamps, 1, nch].
    nfft: FFT size.
    nhop: Shift amount.
    pad_end: If true, pad incomplete frames at end of waveform.

  Returns:
    Tensor dtype complex64 of shape [b, ntsteps, (nfft // 2) + 1, 1] containing the features.
  """
  batch_size, nsamps, nfeats, nch = x.get_shape().as_list()
  if nfeats != 1:
    raise ValueError()

  window_fn = lambda _, dtype: lws_hann_default(nfft, nhop, dtype)

  x = tf.transpose(x, [0, 3, 2, 1])
  X = tf.contrib.signal.stft(x, nfft, nhop, window_fn=window_fn, pad_end=pad_end)
  X = tf.squeeze(X, axis=2)
  X = tf.transpose(X, [0, 2, 3, 1])

  return X


@lru_cache(maxsize=4)
def create_mel_filterbank(*args, **kwargs):
  return librosa.filters.mel(*args, **kwargs)


@lru_cache(maxsize=4)
def create_inverse_mel_filterbank(*args, **kwargs):
  W = create_mel_filterbank(*args, **kwargs)
  return np.linalg.pinv(W)


# NOTE: nfft and hop are configured for fs=20480
def waveform_to_melspec(
    x,
    fs,
    nfft,
    nhop,
    mel_min=125,
    mel_max=7600,
    mel_num_bins=80,
    norm_allow_clipping=True,
    norm_min_level_db=-100,
    norm_ref_level_db=20):
  """Transforms waveform into mel spectrogram feature representation.

  References:
    - https://github.com/r9y9/wavenet_vocoder
    - https://github.com/r9y9/wavenet_vocoder/blob/master/audio.py
    - https://github.com/r9y9/wavenet_vocoder/blob/master/hparams.py

  Args:
    x: nd-array dtype float32 of shape [?, 1, 1].
    fs: Sample rate of x.
    nfft: FFT size.
    nhop: Window size.
    mel_min: Minimum frequency for mel transform.
    mel_max: Maximum frequency for mel transform.
    mel_num_bins: Number of mel bins.
    norm_allow_clipping: If False, throws error if data is clipped during norm.
    norm_min_level_db: Minimum dB level.
    norm_ref_level_db: Maximum dB level (clips between this and 0).

  Returns:
    nd-array dtype float64 of shape [?, nmels, 1] containing the features.
  """
  if x.dtype != np.float32:
    raise ValueError()

  nsamps, nfeats, nch = x.shape
  if nfeats != 1:
    raise ValueError()
  if nch != 1:
    raise NotImplementedError('Can only extract features from monaural signals')

  X = stft(x, nfft, nhop)[:, :, 0]
  X_mag = np.abs(X)

  mel_filterbank = create_mel_filterbank(
      fs, nfft, fmin=mel_min, fmax=mel_max, n_mels=mel_num_bins)
  X_mel = np.swapaxes(np.dot(mel_filterbank, X_mag.T), 0, 1)

  min_level = np.exp(norm_min_level_db / 20 * np.log(10))
  X_mel_db = 20 * np.log10(np.maximum(min_level, X_mel)) - norm_ref_level_db

  if not norm_allow_clipping:
    assert X_mel_db.max() <= 0 and X_mel_db.min() - norm_min_level_db >= 0
  X_mel_dbnorm = np.clip((X_mel_db - norm_min_level_db) / -norm_min_level_db, 0, 1)

  return X_mel_dbnorm[:, :, np.newaxis]


# NOTE: nfft and hop are configured for fs=20480
def waveform_to_melspec_tf(
    x,
    fs,
    nfft,
    nhop,
    mel_min=125,
    mel_max=7600,
    mel_num_bins=80,
    norm_allow_clipping=True,
    norm_min_level_db=-100,
    norm_ref_level_db=20):
  """Transforms batch of waveforms into mel spectrogram feature representation.

  References:
    - https://github.com/r9y9/wavenet_vocoder
    - https://github.com/r9y9/wavenet_vocoder/blob/master/audio.py
    - https://github.com/r9y9/wavenet_vocoder/blob/master/hparams.py

  Args:
    x: Tensor float32 of shape [b, nsamps, 1, nch].
    fs: Sample rate of x.
    nfft: FFT size.
    nhop: Window size.
    mel_min: Minimum frequency for mel transform.
    mel_max: Maximum frequency for mel transform.
    mel_num_bins: Number of mel bins.
    norm_allow_clipping: If False, throws error if data is clipped during norm.
    norm_min_level_db: Minimum dB level.
    norm_ref_level_db: Maximum dB level (clips between this and 0).

  Returns:
    Tensor float32 of shape [b, ntsteps, nmels, 1] containing the features.
  """
  batch_size, nsamps, one, nch = best_shape(x)
  if one != 1:
    raise ValueError()
  if x.dtype != tf.float32:
    raise ValueError()

  X = stft_tf(x, nfft, nhop)
  _, ntsteps, nfeats, _ = best_shape(X)
  X_mag = tf.abs(X)

  mel_filterbank = create_mel_filterbank(
      fs, nfft, fmin=mel_min, fmax=mel_max, n_mels=mel_num_bins)
  mel_filterbank = tf.constant(mel_filterbank, dtype=tf.float32)
  X_mag = tf.transpose(X_mag, [0, 1, 3, 2])
  X_mag = tf.reshape(X_mag, [batch_size * ntsteps * nch, nfeats])
  X_mel = tf.matmul(X_mag, tf.transpose(mel_filterbank))
  X_mel = tf.reshape(X_mel, [batch_size, ntsteps, nch, mel_num_bins])
  X_mel = tf.transpose(X_mel, [0, 1, 3, 2])

  min_level = np.exp(norm_min_level_db / 20 * np.log(10))
  min_level = tf.constant(min_level, dtype=np.float32)

  def tf_log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

  X_mel_db = 20 * tf_log10(tf.maximum(min_level, X_mel)) - norm_ref_level_db

  if not norm_allow_clipping:
    # TODO: TF assert
    #assert X_mel_db.max() <= 0 and X_mel_db.min() - norm_min_level_db >= 0
    raise NotImplementedError()
  X_mel_dbnorm = tf.clip_by_value(
      (X_mel_db - norm_min_level_db) / -norm_min_level_db, 0, 1)

  return X_mel_dbnorm


def waveform_to_tacotron2_melspec(x):
  """Transforms waveform into mel spectrogram feature representation.

  Transforms waveform into feature representation for as described in original Tacotron 2 paper. No open source implementation so cannot gaurantee correctness. Reference:
    - https://arxiv.org/pdf/1712.05884.pdf

  Args:
    x: nd-array dtype float32 of shape [?, 1, 1] at 24000Hz.

  Returns:
    nd-array dtype float64 of shape [?, 80, 1] at 80Hz.
  """
  return waveform_to_melspec(
      x,
      fs=24000,
      nfft=1200,
      nhop=300,
      norm_min_level_db=-40)


def waveform_to_r9y9_melspec(x, fs=22050):
  """Transforms waveform into unofficial mel spectrogram feature representation.

  Transforms waveform into feature representation for unofficial reimplementation of WaveNet vocoder. Unit tests guaranteeing parity with implementation. References:
    - https://github.com/r9y9/wavenet_vocoder
    - https://github.com/r9y9/wavenet_vocoder/blob/master/audio.py
    - https://github.com/r9y9/wavenet_vocoder/blob/master/hparams.py

  Args:
    x: nd-array dtype float32 of shape [?, 1, 1].
    fs: Sample rate (should be 22050 to be the same as r9y9).

  Returns:
    nd-array dtype float64 of shape [?, 80, 1] at 86.13Hz.
  """
  return waveform_to_melspec(
      x,
      fs=fs,
      nfft=1024,
      nhop=256)


def waveform_to_r9y9_melspec_tf(x, fs=22050):
  """Transforms waveform batch into unofficial mel spectrogram feature representation.

  Transforms waveform into feature representation for unofficial reimplementation of WaveNet vocoder. Unit tests guaranteeing parity with implementation. References:
    - https://github.com/r9y9/wavenet_vocoder
    - https://github.com/r9y9/wavenet_vocoder/blob/master/audio.py
    - https://github.com/r9y9/wavenet_vocoder/blob/master/hparams.py

  Args:
    x: nd-array dtype float32 of shape [?, 1, 1].
    fs: Sample rate (should be 22050 to be the same as r9y9).

  Returns:
    nd-array dtype float64 of shape [?, 80, 1] at 86.13Hz.
  """
  return waveform_to_melspec_tf(
      x,
      fs=fs,
      nfft=1024,
      nhop=256)


def magspec_to_waveform_griffin_lim(X_mag, nfft, nhop, ngl=60):
  nsamps, nbins, nch = X_mag.shape
  if nch != 1:
    raise NotImplementedError('Can only invert monaural signals')
  X_mag = X_mag[:, :, 0]

  lws_proc = lws.lws(nfft, nhop, mode='speech', perfectrec=False)
  # angles = np.exp(2j * np.pi * np.random.rand(*X_mag.shape))
  X_complex = np.abs(X_mag).astype(np.complex128)

  # x_gl = lws_proc.istft(X_complex * angles)
  x_gl = lws_proc.istft(X_complex)
  for i in range(ngl):
      angles = np.exp(1j * np.angle(lws_proc.stft(x_gl)))
      x_gl = lws_proc.istft(X_complex * angles)

  x_gl = x_gl[:, np.newaxis, np.newaxis].astype(np.float32)

  return x_gl


def magspec_to_waveform_lws(X_mag, nfft, nhop):
  nsamps, nbins, nch = X_mag.shape
  if nch != 1:
    raise NotImplementedError('Can only invert monaural signals')
  X_mag = X_mag[:, :, 0]

  lws_proc = lws.lws(nfft, nhop, mode='speech', perfectrec=False)
  X_lws = lws_proc.run_lws(X_mag)
  x_lws = lws_proc.istft(X_lws)

  x_lws = x_lws[:, np.newaxis, np.newaxis].astype(np.float32)

  return x_lws


# NOTE: nfft and hop are configured for fs=20480
def melspec_to_waveform(
    X_mel_dbnorm,
    fs,
    nfft,
    nhop,
    mel_min=125,
    mel_max=7600,
    norm_min_level_db=-100,
    norm_ref_level_db=20,
    phase_estimation='lws',
    waveform_len=None):
  """Approximately inverts mel spectrogram to waveform.

  Args:
    X_mel: nd-array dtype float64 of shape [?, mel_num_bins, num_ch].
    fs: Sample rate of waveform.
    nfft: FFT size.
    nhop: Window size.
    mel_min: Minimum frequency for mel transform.
    mel_max: Maximum frequency for mel transform.
    norm_allow_clipping: If False, throws error if data is clipped during norm.
    norm_min_level_db: Minimum dB level.
    norm_ref_level_db: Maximum dB level (clips between this and 0).
    phase_estimation: One of 'lws' (local weighted sums) or 'gl60' (Griffin-Lim)
    waveform_len: If specified, pad or clip output to this length.

  Returns:
    nd-array dtype float32 of shape [waveform_len, 1, num_ch] containing the features.
  """
  if X_mel_dbnorm.dtype != np.float64:
    raise ValueError()

  nsamps, mel_num_bins, nch = X_mel_dbnorm.shape
  if nch != 1:
    raise NotImplementedError('Can only invert monaural signals')
  X_mel_dbnorm = X_mel_dbnorm[:, :, 0]

  X_mel_db = (X_mel_dbnorm * -norm_min_level_db) + norm_min_level_db
  X_mel = np.power(10, (X_mel_db + norm_ref_level_db) / 20)

  inv_mel_filterbank = create_inverse_mel_filterbank(
      fs, nfft, fmin=mel_min, fmax=mel_max, n_mels=mel_num_bins)
  X_mag = np.dot(X_mel, inv_mel_filterbank.T)
  X_mag = np.maximum(0., X_mag)

  X_mag = X_mag[:, :, np.newaxis]
  if phase_estimation == 'lws':
    x = magspec_to_waveform_lws(X_mag, nfft, nhop)
  elif phase_estimation[:2] == 'gl':
    try:
      ngl = int(phase_estimation[2:])
    except:
      raise ValueError()
    x = magspec_to_waveform_griffin_lim(X_mag, nfft, nhop, ngl)
  else:
    raise ValueError()

  if waveform_len is not None:
    x_len = x.shape[0]
    if x_len < waveform_len:
      x = np.pad(x, [[0, waveform_len - x_len], [0, 0], [0, 0]], 'constant')
      pass
    elif x_len > waveform_len:
      x = x[:waveform_len]

  return x.astype(np.float32)


def r9y9_melspec_to_waveform(
    X_mel_dbnorm,
    fs=22050,
    phase_estimation='lws',
    waveform_len=None):
  """Approximately inverts unofficial mel spectrogram to waveform.

  Args:
    X_mel_dbnorm: nd-array dtype float64 of shape [?, 80, 1] at 86.13Hz.
    fs: Output sample rate (should be 22050 to be the same as r9y9).
    phase_estimation: One of 'lws' (local weighted sums) or 'gl60' (Griffin-Lim)
    waveform_len: If specified, pad or trim output waveform to be this long.

  Returns:
    nd-array dtype float32 of shape [?, 1, 1].
  """
  return melspec_to_waveform(
      X_mel_dbnorm,
      fs=fs,
      nfft=1024,
      nhop=256,
      phase_estimation=phase_estimation,
      waveform_len=waveform_len)
