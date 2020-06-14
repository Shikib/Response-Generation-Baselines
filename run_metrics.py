import argparse

from collections import Counter

parser = argparse.ArgumentParser(description='Topical-Chat Evaluation Script')

parser.add_argument('--data_path', type=str, default='processed_output/')
parser.add_argument('--save_path', type=str, default='save/')

args = parser.parse_args()

def f1(true, pred):
  common = Counter(true) & Counter(pred)
  num_same = sum(common.values())
  if num_same == 0:
    return 0
  precision = num_same/len(pred)
  recall = num_same/len(true)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1

def unigram_div(true, pred):
  if len(pred) == 0:
    return 0
  return len(set(pred))/len(pred)

def bigram_div(true, pred):
  bigrams = []
  for i in range(len(pred)-1):
    bigrams.append(' '.join(pred[i:i+2]))
  if len(bigrams) == 0:
    return 0
  return len(set(bigrams))/len(bigrams)

def eval(gt, out, f):
  #return sum([f(true,'. i the , that a to it is of'.split()) for true,pred in zip(gt,out)])/len(gt)
  return sum([f(true,pred) for true,pred in zip(gt,out)])/len(gt)
  

# Frequent
freq_gt = [l.replace('_eos', '').replace('_go', '').strip().split() for l in open(args.data_path + 'test_freq.tgt').readlines()]
freq_out = [l.strip().split() for l in open(args.save_path + 'freq_out.tgt').readlines()]
print("F-1 score:", eval(freq_gt, freq_out, f1))
print("Distinct Unigrams:", eval(freq_gt, freq_out, unigram_div))
print("Distinct Bigrams:", eval(freq_gt, freq_out, bigram_div))

# Rare
rare_gt = [l.replace('_eos', '').replace('_go', '').strip().split() for l in open(args.data_path + 'test_rare.tgt').readlines()]
rare_out = [l.strip().split() for l in open(args.save_path + 'rare_out.tgt').readlines()]
print("F-1 score:", eval(rare_gt, rare_out, f1))
print("Distinct Unigrams:", eval(rare_gt, rare_out, unigram_div))
print("Distinct Bigrams:", eval(rare_gt, rare_out, bigram_div))
