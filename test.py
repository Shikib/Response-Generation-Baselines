import argparse
import json
import math
import random
import model
import sys

from collections import Counter
from tqdm import tqdm

def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Topical-Chat Training Script')

parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 42)')

parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=64, metavar='N')
parser.add_argument('--use_attn', type=str2bool, const=True, nargs='?', default=False)

parser.add_argument('--emb_size', type=int, default=300)
parser.add_argument('--hid_size', type=int, default=300)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--l2_norm', type=float, default=0.00001)
parser.add_argument('--clip', type=float, default=5.0, help='clip the gradient by norm')

parser.add_argument('--seq2seq', type=str2bool, const=True, nargs='?', default=False)
parser.add_argument('--transformer', type=str2bool, const=True, nargs='?', default=False)

parser.add_argument('--use_knowledge', type=str2bool, const=True, nargs='?', default=False)

parser.add_argument('--data_path', type=str, default='processed_output/')
parser.add_argument('--data_size', type=float, default=-1.0)
parser.add_argument('--save_path', type=str, default='save/')

args = parser.parse_args()

assert args.seq2seq or args.transformer, "Must turn on one training flag"

if not args.data_path.endswith('/'):
  args.data_path = args.data_path + '/'

if not args.save_path.endswith('/'):
  args.save_path = args.save_path + '/'

def load_data(split):
  src = [l.strip() for l in open(args.data_path + split + '.src').readlines()]
  tgt = [l.strip() for l in open(args.data_path + split + '.tgt').readlines()]
  fct = [l.strip() for l in open(args.data_path + split + '.fct').readlines()]
  return list(zip(src,tgt,fct))

# Load data
train = load_data('train')
valid_freq = load_data('valid_freq')
valid_rare = load_data('valid_rare')
test_freq = load_data('test_freq')
test_rare = load_data('test_rare')
print("Number of training instances:", len(train))
print("Number of validation (freq) instances:", len(valid_freq))
print("Number of validation (rare) instances:", len(valid_rare))
print("Number of testing (freq) instances:", len(test_freq))
print("Number of testing (rare) instances:", len(test_rare))

# Build vocabulary
i2w = [w.strip() for w in open(args.save_path + 'vocab.txt').readlines()]
w2i = {w:i for i,w in enumerate(i2w)}

# Create models
assert args.num_layers == 1, "num_layers > 1 not supported yet"

if args.seq2seq:
  encoder = model.Encoder(vocab_size=len(i2w), 
                          emb_size=args.emb_size, 
                          hid_size=args.hid_size,
                          num_layers=args.num_layers)
  
  decoder = model.Decoder(emb_size=args.emb_size,
                          hid_size=args.hid_size,
                          vocab_size=len(i2w),
                          num_layers=args.num_layers,
                          use_attn=args.use_attn)
  
  model = model.Seq2Seq(encoder=encoder,
                        decoder=decoder,
                        i2w=i2w,
                        use_knowledge=args.use_knowledge,
                        args=args,
                        test=True).cuda()
elif args.transformer:
  model = model.Transformer(i2w=i2w, use_knowledge=args.use_knowledge, args=args, test=True).cuda()


# TEST EVALUATION
best_epoch = args.epoch
model.load("{0}/model_{1}.bin".format(args.save_path, best_epoch))
model.transformer.eval()

# Iterate over batches
num_batches = math.ceil(len(valid_freq)/args.batch_size)
cum_loss = 0
cum_words = 0
predicted_sentences = []
indices = list(range(len(valid_freq)))
for batch in tqdm(range(num_batches)):
  # Prepare batch
  batch_indices = indices[batch*args.batch_size:(batch+1)*args.batch_size]
  batch_rows = [valid_freq[i] for i in batch_indices]

  # Encode batch. If facts are being used, they'll be prepended to the input
  input_seq, input_lens, target_seq, target_lens = model.prep_batch(batch_rows)

  # Decode batch
  predicted_sentences += model.decode(input_seq, input_lens)


# Save predictions
open("{0}/valid_freq_out.tgt".format(args.save_path), "w+").writelines([l+"\n" for l in predicted_sentences])
