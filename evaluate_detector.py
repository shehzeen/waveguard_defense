import numpy as np
import argparse
import os
import sys
from os import listdir
from os.path import isfile, join
import editdistance
import json
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import roc_auc_score, roc_curve
import random

def _create_dataset(in_orig, in_orig_def, in_adv, in_adv_def):
  def _get_transcriptions(dir_path):
    with open(join(dir_path, "transcriptions.json")) as f:
      transcriptions = json.loads(f.read())
    return transcriptions

  org_t = _get_transcriptions(in_orig)
  org_def_t = _get_transcriptions(in_orig_def)
  adv_t = _get_transcriptions(in_adv)
  adv_def_t = _get_transcriptions(in_adv_def)

  dataset_list=[]
  for fn in org_t:
    if (fn in org_t) and (fn in org_def_t) and (fn in adv_t) and (fn in adv_def_t):
      dataset_list.append([org_t[fn],org_def_t[fn], 0])
      dataset_list.append([adv_t[fn],adv_def_t[fn], 1])
  print("Dataset list", len(dataset_list))
  return dataset_list


def create_dataset(args):
  return _create_dataset(args.in_orig, args.in_orig_def, args.in_adv, args.in_adv_def)

def calculate_edit_dist(data_list):
    
  for i in range(len(data_list)):
    undefended_transcription = data_list[i][0]
    defended_transcription = data_list[i][1]
    ed = editdistance.eval(undefended_transcription,defended_transcription )
    try:
        normalized_ed = (1.0 * ed)/max(len(undefended_transcription),len(defended_transcription))
    except:
        normalized_ed = 1.0
    data_list[i].append(normalized_ed) 

  return data_list    

def evaluate(data_list, log_dir, exp_name):

  y_true = []
  y_score = []
  
  random.shuffle(data_list)
  for i in range(len(data_list)):
    true_label = data_list[i][2]
    y_true.append(true_label * 1.0)
    y_score.append(data_list[i][3])

  auc_score = roc_auc_score(y_true, y_score)
  metrics = {
    'auc_score' : auc_score,
  }

  fpr, tpr, thresholds = roc_curve(y_true, y_score)


  plt.figure()
  lw = 2
  plt.plot(fpr, tpr, color='darkorange',
           lw=lw, label='ROC curve (area = %0.2f)' % auc_score)
  plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic example')
  plt.legend(loc="lower right")
  fn = "{}.png".format(exp_name)
  fp = os.path.join(log_dir, fn)
  plt.savefig(fp)

  results_fn = "{}_metrics.json".format(exp_name)
  results_fn = os.path.join(log_dir, results_fn)
  with open(results_fn, 'w') as f:
    f.write(json.dumps(metrics))

  print (metrics)
  return metrics

def main():
  '''
  Takes as input audio file paths and trains a detector.
  '''
  parser = argparse.ArgumentParser(description=None)
  parser.add_argument('--in_orig', type=str, required=True,
      help='Filepath of original input audio')
  parser.add_argument('--in_orig_def', type=str, required=True,
      help='Filepath of defended original input audio')
  parser.add_argument('--in_adv', type=str, required=True,
      help='Filepath of adversarial input audio')
  parser.add_argument('--in_adv_def', type=str, required=True,
      help='Filepath of adversarial defended input audio')
  parser.add_argument('--log_dir', type=str, required=False,
      help='Log Dir')


  args = parser.parse_args()
  while len(sys.argv) > 1:
    sys.argv.pop()

  transcription_dataset = create_dataset(args)
  transcription_dataset = calculate_edit_dist(transcription_dataset)
  
  if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

  results = evaluate(transcription_dataset, args.log_dir, "Test")
  

if __name__ == '__main__':
  main()