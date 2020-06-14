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

parser.add_argument('--num_epochs', type=int, default=20)
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
print("Number of training instances:", len(train))

# Build vocabulary
all_words = [w for row in train for sent in row for w in sent.split()]
counter = Counter(all_words)
i2w = ['_unk', '_pad'] + [w for w in counter if counter[w] > 10]

# Save vocabulary to file
open(args.save_path + 'vocab.txt', 'w+').writelines([w+'\n' for w in i2w])

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
                        args=args).cuda()
elif args.transformer:
  model = model.Transformer(i2w=i2w, use_knowledge=args.use_knowledge, args=args).cuda()
else:
  raise Exception("Must be one of transformer or seq2seq")

step = 0
constant =  2.0 * (args.hid_size ** -0.5)
warmup = 16000
for epoch in range(args.num_epochs):
  indices = list(range(len(train)))
  random.shuffle(indices)

  num_batches = math.ceil(len(train)/args.batch_size)
  cum_loss = 0
  for batch in tqdm(range(num_batches)):
    step += 1
    # Prepare batch
    batch_indices = indices[batch*args.batch_size:(batch+1)*args.batch_size]
    batch_rows = [train[i] for i in batch_indices]

    # Encode batch. If facts are being used, they'll be prepended to the input
    input_seq, input_lens, target_seq, target_lens = model.prep_batch(batch_rows)


    # Set learning rate
    lr = constant * min(1.0, step/warmup)
    lr *= max(step, warmup) ** -0.5

    # Set learning rate in model
    for p in model.optim.param_groups:
      p['lr'] = lr

    # Train batch
    cum_loss += model.train(input_seq, input_lens, target_seq, target_lens)

    # Log batch 
    if batch > 0 and batch % 50 == 0:
      print("Epoch {0}/{1} Batch {2}/{3} Avg Loss {4:.2f} LR {5:.4f}".format(epoch+1, args.num_epochs, batch, num_batches, cum_loss/50, lr))
      cum_loss = 0

  model.save("{0}/model_{1}.bin".format(args.save_path, epoch))
