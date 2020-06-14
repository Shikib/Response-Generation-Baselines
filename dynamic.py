import tornado.ioloop
import tornado.web
import argparse
import json
import math
import random
import model
import sys

from collections import Counter
from tqdm import tqdm

import spacy
import string
from sklearn.feature_extraction.text import TfidfVectorizer

def clean(s):
  return ''.join([c for c in s.lower() if c not in string.punctuation])

def build_tfidf():
  vectorizer = TfidfVectorizer()
  corpus = []
  for fn in ["src", "tgt", "fct"]:
    corpus += [e.strip() for e in open("processed_output/train.{0}".format(fn)).readlines()]

  corpus = [clean(e) for e in corpus]
  vectorizer.fit(corpus)
  return vectorizer

nlp = spacy.load('en')
def tokenize(data):
  new_data = []
  print("Tokenizing")
  docs = nlp.tokenizer.pipe([' '.join(s.lower().split()) for s in data])
  for doc in tqdm(docs):
    # Tokenize with spacy
    tokenized = ' '.join([e.text for e in doc])

    # Fix mis-tokenized tags
    tokenized = tokenized.replace('_ eos', '_eos').replace('_ go', '_go').replace('_ nofact', '_nofact')

    new_data.append(tokenized)

  return new_data

vectorizer = build_tfidf()
potential_facts = [e.strip() for e in open("processed_output/train.fct").readlines()]
potential_facts = [e for e in potential_facts if len(e.split()) < 20]
f_vec = vectorizer.transform([clean(e) for e in potential_facts])
def best_fact(message):
  r_vec = vectorizer.transform([clean(message)])
  sim = r_vec.dot(f_vec.transpose()).todense()

  return potential_facts[sim.argmax()]

def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Topical-Chat Interactive Script')

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

# Load model
model.load("{0}/model_{1}.bin".format(args.save_path, args.num_epochs-1))
model.transformer.eval()

def reply(history):
  clean_history = [clean(str(m).strip()) for m in tokenize(history)]
  src = " ".join([e + " _eos" for e in clean_history])
  tgt = ""
  fct = best_fact(clean_history[-1])

  print(src)
  print(fct)

  input_seq, input_lens, target_seq, target_lens = model.prep_batch([(src,tgt,fct)])
  output = model.decode(input_seq, input_lens)
  return output[0]


histories = {

}

class MainHandler(tornado.web.RequestHandler):
  def set_default_headers(self):
    self.set_header("Access-Control-Allow-Origin", "*")
    self.set_header("Access-Control-Allow-Headers", "Content-Type, Access-Control-Allow-Headers, Authorization, X-Requested-With")
    self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')

  def options(self):
    pass

  def post(self):
    body = json.loads(self.request.body.decode())
    if body["text"] == "SSTTAARRTT":
      histories[body["userID"]] = []
    else:
      histories[body["userID"]].append(body["text"])
      response = reply(histories[body["userID"]])
      histories[body["userID"]].append(response)
      self.write(json.dumps({"body": json.dumps({"utterance": response})}))

def make_app():
  return tornado.web.Application([
    (r"/", MainHandler),
  ])

if __name__ == "__main__":
  app = make_app()
  app.listen(8889)
  tornado.ioloop.IOLoop.current().start()


