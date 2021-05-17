import audioio
import spectral
from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
import os
import glob
import librosa
from pysndfx import AudioEffectsChain
from tqdm import tqdm
import lpc

def reduce_noise_power(y, sr=16000):
  cent = librosa.feature.spectral_centroid(y=y, sr=sr)

  threshold_h = round(np.median(cent))*1.5
  threshold_l = round(np.median(cent))*0.1

  less_noise = AudioEffectsChain().lowshelf(gain=-30.0, frequency=threshold_l, slope=0.8).highshelf(gain=-12.0, frequency=threshold_h, slope=0.5)#.limiter(gain=6.0)
  y_clean = less_noise(y)

  return y_clean  


def down_up_sample_np(inp_audio,sr=16000, down_sample_rate = 8000):
    
  downsampled_audio = librosa.resample(inp_audio, sr, down_sample_rate)
  upsampled_audio = librosa.resample(downsampled_audio, down_sample_rate, sr)
  # print(inp_audio.shape, downsampled_audio,shape,upsampled_audio.shape)

  return upsampled_audio

def run_defense_on_directory(in_dir, out_base, defender_type, defender_hp, meta_fp, model_ckpt, subseq_len):
  '''
  Takes as input audio file path as input and generates a cleaned output audio.
  '''
  def tacotron_mel_to_mag(X_mel_dbnorm, invmeltrans):
    norm_min_level_db = -100
    norm_ref_level_db = 20
    
    X_mel_db = (X_mel_dbnorm * -norm_min_level_db) + norm_min_level_db
    X_mel = np.power(10, (X_mel_db + norm_ref_level_db) / 20)
    X_mag = np.dot(X_mel, invmeltrans.T)
    return X_mag

  mel_bins = 80
  down_sample_rate = 8000
  num_bits = 4
  lpc_order = 10
  
  if "mel" in defender_type:
    mel_bins = int(defender_hp)

  if defender_type == "downsample_upsample":
    down_sample_rate = int(defender_hp)

  if defender_type == "quant":
    num_bits = int(defender_hp)
  if defender_type == "lpc":
    lpc_order = int(defender_hp)
    
  if defender_type == "mel_advoc":
    # Create Advoc Graph
    gen_graph = tf.Graph()
    with gen_graph.as_default():
      gan_saver = tf.train.import_meta_graph(meta_fp)
    gen_sess = tf.Session(graph=gen_graph)
    print("Restoring")
    gan_saver.restore(gen_sess, model_ckpt)
    gen_mag_spec = gen_graph.get_tensor_by_name('generator/decoder_1/strided_slice_1:0')
    x_mag_input = gen_graph.get_tensor_by_name('ExpandDims_1:0')

  input_fps = glob.glob(os.path.join(in_dir, '*.wav'))
  fs, _  = audioio.decode_audio(input_fps[0], fastwav = True)

  in_dir_name = in_dir.split("/")[-1]
  if len(in_dir_name) == 0:
    in_dir_name = in_dir.split("/")[-2]

  out_dir_name = in_dir_name + "_defended_{}_{}".format(defender_type, defender_hp)
  out_dir = os.path.join(out_base, out_dir_name)

  if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

  # Mel, Mag graph
  inp_audio = tf.placeholder('float32', [1, None, 1, 1])
  NFFT = 1024
  NHOP = 256
  input_mel_spec = spectral.waveform_to_melspec_tf(inp_audio, fs, NFFT, NHOP, mel_num_bins=mel_bins)[0]
  input_mag_spec = tf.abs(spectral.stft_tf(inp_audio, NFFT, NHOP))[0]


  quant_dequant_in = tf.quantization.quantize_and_dequantize(inp_audio, input_min = -1.0, input_max = 1.0, num_bits = num_bits)
  _down_up_sample_np  = lambda x: down_up_sample_np(x, down_sample_rate = down_sample_rate)
  down_up_sample_in = tf.numpy_function(_down_up_sample_np, [inp_audio[0,:,0,0]], Tout=tf.float32)


  _filter_power_np  = lambda x: reduce_noise_power(x)
  filter_power_in = tf.numpy_function(_filter_power_np, [inp_audio[0,:,0,0]], Tout=tf.float32)

  _lpc_np = lambda x: lpc.lpc_compress_decompress(x, lpc_order = lpc_order)
  lpc_in = tf.numpy_function(_lpc_np, [inp_audio[0,:,0,0]], Tout=tf.float32)

  spec_session = tf.Session()

  for idx in tqdm(range(len(input_fps))):
    in_fp = input_fps[idx]
    # print(in_fp)
    fs, in_wav_np  = audioio.decode_audio(in_fp, fastwav = True)
    
    #print("INput wav", in_wav_np.min(), in_wav_np.max())
    input_mel_spec_np, input_mag_spec_np, quant_dequant_np,down_up_sample_out,filter_out = spec_session.run([input_mel_spec,input_mag_spec, quant_dequant_in,down_up_sample_in,filter_power_in], 
      feed_dict = {
        inp_audio : [in_wav_np]
    })
    
    
    if defender_type == 'quant':
      wave = quant_dequant_np[0]
    elif defender_type == 'downsample_upsample':
      wave = down_up_sample_out
      wave = np.reshape( wave, (len(wave), 1, 1))
    elif defender_type == 'filter_power':
      wave = filter_out
      wave = np.reshape( wave, (len(wave), 1, 1))
    elif defender_type == "lpc":
      lpc_out = spec_session.run([lpc_in], 
        feed_dict = {
          inp_audio : [in_wav_np]
      })
      wave = lpc_out[0]
      wave = np.reshape( wave, (len(wave), 1, 1))
      
    else:
      # mel inversion transforms
      inv_mel_filterbank = spectral.create_inverse_mel_filterbank(
          fs, NFFT, fmin=125, fmax=7600, n_mels=mel_bins)
      
      if defender_type == 'mel_heuristic':
        # print("Mel + Heuristic Mag + Heuristic Phase")
        gen_mag = tacotron_mel_to_mag(input_mel_spec_np[:,:,0], inv_mel_filterbank)
        gen_mag = np.expand_dims(gen_mag, -1)
      
      elif defender_type == 'mag_heuristic':
        # print("Actual Mag + Heuristic Phase")
        gen_mag = input_mag_spec_np
          


      elif defender_type == 'mel_advoc':
        # print("Mel + Advoc Mag + Heuristic Phase")
        subseq_len = subseq_len
        X_mag = tacotron_mel_to_mag(input_mel_spec_np[:,:,0], inv_mel_filterbank)
        x_mag_original_length = X_mag.shape[0]
        x_mag_target_length = int(X_mag.shape[0] / subseq_len ) * subseq_len + subseq_len
        X_mag = np.pad(X_mag, ([0,x_mag_target_length - X_mag.shape[0]], [0,0]), 'constant')
        num_chunks = int(x_mag_target_length/subseq_len)
        X_mag = np.reshape(X_mag, [num_chunks, subseq_len, 513, 1])
        gen_mags = []
        for n in range(num_chunks):
          _gen = gen_sess.run([gen_mag_spec], feed_dict = {
            x_mag_input : X_mag[n:n+1]
            })[0]
          gen_mags.append(_gen[0])

        gen_mag = np.concatenate(gen_mags, axis = 0) #flattening the gen_mags list
        gen_mag = gen_mag[0:x_mag_original_length]

      wave = spectral.magspec_to_waveform_lws(gen_mag.astype('float64'), NFFT, NHOP)


    input_fn = os.path.splitext(os.path.split(in_fp)[1])[0]
    output_fn = input_fn + '.wav'
    output_fp = os.path.join(out_dir, output_fn)

    audioio.save_as_wav(output_fp, fs, wave)

  return out_dir

if __name__ == '__main__':
  parser = ArgumentParser()

  parser.add_argument('--in_dir', type=str, required=True,
      help='Filepath of input audio')
  parser.add_argument('--out_base', type=str, required=True,
      help='Filepath of output audio')
  parser.add_argument('--model_ckpt', type=str,
      help='Adversarial vocoder checkpoint')
  parser.add_argument('--meta_fp', type=str,
      help='Met graph filepath')
  parser.add_argument('--subseq_len', type=int,
      help="model subseq length")
  parser.add_argument('--defender_type', type=str,
      help="defender type", choices = ['mel_advoc', 'mel_heuristic', 'mag_heuristic', 'quant', 'downsample_upsample','filter_power', 'lpc'])
  
  parser.add_argument('--defender_hp' , type=str)
  


  parser.set_defaults(
      in_fp=None,
      out_base=None,
      model_ckpt=None,
      meta_fp=None,
      subseq_len=256,
      defender_hp = None
      )

  args = parser.parse_args()

  run_defense_on_directory(args.in_dir, args.out_base, args.defender_type, args.defender_hp, args.meta_fp, args.model_ckpt, args.subseq_len)



