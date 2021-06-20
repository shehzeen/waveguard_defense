import tensorflow as tf

import numpy as np
import tensorflow as tf
import argparse
from shutil import copyfile

import scipy.io.wavfile as wav
import time
import sys
from collections import namedtuple
import os
from os import listdir
from os.path import isfile, join
sys.path.append("DeepSpeech")
import DeepSpeech
import json
import time

try:
    import pydub
    import struct
except:
    print("pydub was not loaded, MP3 compression will not work")

from tf_logits import get_logits
# These are the tokens that we're allowed to use.
# The - token is special and corresponds to the epsilon
# value in CTC decoding, and can not occur in the phrase
toks = " abcdefghijklmnopqrstuvwxyz'-"



def _get_file_names(audio_dir):
  file_names = []
  for f in listdir(audio_dir):
    if f.endswith(".wav"):
      file_names.append(f)
    
  return file_names

def transcribe_deepspeech(audio_dir, restore_path):
    global WEIGHTS_RESTORED
    file_names = _get_file_names(audio_dir)
    
    transcription_map = {}
    sess = tf.InteractiveSession()
    all_times = []
    for i, file_name in enumerate(file_names):
        audio_path = join(audio_dir, file_name)
        _, audio = wav.read(audio_path)
        
        N = len(audio)

        if i % 10 == 0:
          # for some reason, graph keeps getting bigger at each iteration.
          # resetting graph after every 10 iterations
          sess.close()
          tf.reset_default_graph()
          sess = tf.InteractiveSession()
          tf.debugging.set_log_device_placement(True)
          new_input = tf.placeholder(tf.float32, [1, N])
          lengths = tf.placeholder(tf.int32, [1])  
          logits = get_logits(new_input, lengths) #created graph of deepspeech
          print("Logits", logits)
          saver = tf.train.Saver()
          print ("Restoring Weights")
          saver.restore(sess, restore_path)
          
          decoded_audio, _ = tf.nn.ctc_beam_search_decoder(logits, lengths, merge_repeated=False, beam_width=500)
          
          
        else:
          new_input = tf.placeholder(tf.float32, [1, N])
          lengths = tf.placeholder(tf.int32, [1])
        
          with tf.variable_scope("", reuse=True):
            logits = get_logits(new_input, lengths, reuse=True)

          decoded_audio, _ = tf.nn.ctc_beam_search_decoder(logits, lengths, merge_repeated=False, beam_width=500)
        
        length = (len(audio)-1)//320
        l = len(audio)
        tic = time.time()
        r = sess.run(decoded_audio, {new_input: [audio],
                               lengths: [length]})
        toc = time.time()
        all_times.append(toc - tic)
        transcription = "".join([toks[x] for x in r[0].values])
        transcription_map[file_name] = transcription

        print ("Trascribed {} out of {}".format(i, len(file_names)))
        print ("Transcription:", transcription)
    
    
    with open(join(audio_dir, "transcriptions.json"), 'w') as f:
      print("Saving Transcriptions:", audio_dir)
      f.write(json.dumps(transcription_map))
    
    print ("Average time", np.mean(all_times))
    return transcription_map


def transcribe_dirs(dirs, restore_path):
  '''
  Transcribes all audio files in the input directories and 
  saves transcriptions in a json file in the same directory.
  '''

  for audio_dir in dirs:
    transcribe_deepspeech(audio_dir, restore_path)
    print("-**-" * 50)
  print ("Finished transcribing")


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=None)
  parser.add_argument('--dirs', type=str, nargs='+', required=True,
      help='Filepath of original input audio')
  parser.add_argument('--restore_path', type=str, required=True,
      help="Path to the DeepSpeech checkpoint (ending in model0.4.1)")

  args = parser.parse_args()
  while len(sys.argv) > 1:
    sys.argv.pop()
    
    
  transcribe_dirs(args.dirs, args.restore_path)

