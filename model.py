import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim

import transformer

class Encoder(nn.Module):
  def __init__(self, vocab_size, emb_size, hid_size, num_layers):
    super(Encoder, self).__init__() 
    self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=3)
    self.encoder = nn.LSTM(emb_size, hid_size)

  def forward(self, seqs, lens):
    # Embed
    emb_seqs = self.embedding(seqs)

    # Sort by length
    sort_idx = sorted(range(len(lens)), key=lambda i: -lens[i])
    emb_seqs = emb_seqs[:,sort_idx]
    lens = [lens[i] for i in sort_idx]

    # Pack sequence
    packed = torch.nn.utils.rnn.pack_padded_sequence(emb_seqs, lens)

    # Forward pass through LSTM
    outputs, hidden = self.encoder(packed)

    # Unpack outputs
    outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
    
    # Unsort
    unsort_idx = sorted(range(len(lens)), key=lambda i: sort_idx[i])
    outputs = outputs[:,unsort_idx]
    hidden = (hidden[0][:,unsort_idx], hidden[1][:,unsort_idx])

    return outputs, hidden

class Decoder(nn.Module):
  def __init__(self, emb_size, hid_size, vocab_size, num_layers, use_attn=True):
    super(Decoder, self).__init__()
    self.embedding = nn.Embedding(vocab_size, emb_size)
    self.out = nn.Linear(hid_size, vocab_size)
    self.use_attn = use_attn
    self.hid_size = hid_size
    if use_attn:
      self.decoder = nn.LSTM(emb_size+hid_size, hid_size)
      self.W_a = nn.Linear(hid_size * 2, hid_size)
      self.v = nn.Linear(hid_size, 1)
    else:
      self.decoder = nn.LSTM(emb_size, hid_size)

  def forward(self, hidden, last_word, encoder_outputs, ret_out=False, ret_logits=False, ret_attn=False):
    if not self.use_attn:
      embedded = self.embedding(last_word)
      output, hidden = self.decoder(embedded, hidden)
      if not ret_out:
        return F.log_softmax(self.out(output), dim=2), hidden
      else:
        return F.log_softmax(self.out(output), dim=2), hidden, output
    else:
      embedded = self.embedding(last_word)

      # Attn
      h = hidden[0].repeat(encoder_outputs.size(0), 1, 1)
      attn_energy = F.tanh(self.W_a(torch.cat((h, encoder_outputs), dim=2)))
      attn_logits = self.v(attn_energy).squeeze(-1) - 1e5 * (encoder_outputs.sum(dim=2) == 0).float()
      attn_weights = F.softmax(attn_logits, dim=0).permute(1,0).unsqueeze(1)
      context_vec = attn_weights.bmm(encoder_outputs.permute(1,0,2)).permute(1,0,2)

      # Concat with embeddings
      rnn_input = torch.cat((context_vec, embedded), dim=2)

      # Forward
      output, hidden = self.decoder(rnn_input, hidden)
      if ret_attn:
        if not ret_out:
          if ret_logits:
            return self.out(output), hidden, attn_weights
          else:
            return F.log_softmax(self.out(output), dim=2), hidden, attn_weights
        else:
          return F.log_softmax(self.out(output), dim=2), hidden, output, attn_weights
      else:
        if not ret_out:
          if ret_logits:
            return self.out(output), hidden
          else:
            return F.log_softmax(self.out(output), dim=2), hidden
        else:
          return F.log_softmax(self.out(output), dim=2), hidden, output

class Seq2Seq(nn.Module):
  def __init__(self, encoder, decoder, i2w, use_knowledge, args, test=False):
    super(Seq2Seq, self).__init__()

    self.args = args
    self.use_knowledge = use_knowledge

    # Model
    self.encoder = encoder
    self.decoder = decoder

    # Vocab
    self.i2w = i2w
    self.w2i = {w:i for i,w in enumerate(i2w)}

    # Training
    if test:
      self.criterion = nn.NLLLoss(ignore_index=self.w2i['_pad'], reduction='sum')
    else:
      self.criterion = nn.NLLLoss(ignore_index=self.w2i['_pad'])
    self.optim = optim.Adam(lr=args.lr, params=self.parameters(), weight_decay=args.l2_norm)

  def prep_batch(self, rows, wh=128, wk=64):
    def _pad(arr, pad):
      # Given an array of integer arrays, pad all arrays to the same length
      lengths = [len(e) for e in arr]
      max_len = max(lengths)
      return [e+[pad]*(max_len-len(e)) for e in arr], lengths

    # Split all rows
    rows = [[e.split() for e in row] for row in rows]

    # Form inputs
    if self.use_knowledge:
      inputs = [row[0][-wh:] + row[2][:wk] + ['_eos'] for row in rows]
    else:
      inputs = [row[0][-wh:] for row in rows]

    # Input seq 
    inputs = [[self.w2i.get(w, self.w2i['_unk']) for w in inp] for inp in inputs]
    input_seq, input_lens = _pad(inputs, self.w2i['_pad'])
    input_seq = torch.cuda.LongTensor(input_seq).t()

    # Target seq
    targets = [[self.w2i.get(w, self.w2i['_unk']) for w in row[1]] for row in rows]
    target_seq, target_lens = _pad(targets, pad=self.w2i['_pad'])
    target_seq = torch.cuda.LongTensor(target_seq).t()

    return input_seq, input_lens, target_seq, target_lens

  def forward(self, input_seq, input_lens, target_seq, target_lens):
    # Encoder
    encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)

    # Decoder
    decoder_hidden = encoder_hidden
    probas = torch.zeros(target_seq.size(0), target_seq.size(1), len(self.w2i)).cuda()
    last_word = target_seq[0].unsqueeze(0)
    for t in range(1,target_seq.size(0)):
      # Pass through decoder
      decoder_output, decoder_hidden = self.decoder(decoder_hidden, last_word, encoder_outputs)

      # Save output
      probas[t] = decoder_output

      # Set new last word
      last_word = target_seq[t].unsqueeze(0)

    return probas

  def train(self, input_seq, input_lens, target_seq, target_lens):
    self.optim.zero_grad()

    # Forward
    proba = self.forward(input_seq, input_lens, target_seq, target_lens)

    # Loss
    loss = self.criterion(proba.view(-1, proba.size(-1)), target_seq.flatten())

    # Backwards
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip)
    self.optim.step()

    return loss.item()

  def eval_ppl(self, input_seq, input_lens, target_seq, target_lens):
    # Forward
    proba = self.forward(input_seq, input_lens, target_seq, target_lens)

    # Loss
    loss = self.criterion(proba.view(-1, proba.size(-1)), target_seq.flatten())

    return loss.item()

  def decode(self, input_seq, input_lens, top_p=0, max_len=100, p_copy=0):
    batch_size = input_seq.size(1)
    predictions = torch.zeros((batch_size, max_len))

    with torch.no_grad():
      # Encoder
      encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)

      # Decoder
      decoder_hidden = encoder_hidden
      last_word = torch.cuda.LongTensor([[self.w2i['_go'] for _ in range(batch_size)]])

      # Input one-hot
      input_oh = torch.eye(len(self.w2i))[input_seq].cuda()
      for t in range(max_len):
        # Pass through decoder
        decoder_output, decoder_hidden, attn = self.decoder(decoder_hidden, last_word, encoder_outputs, ret_logits=top_p>0, ret_attn=True)
        copy_prob = attn.bmm(input_oh.permute(1, 0, 2)).permute(1, 0, 2)

        # Get top candidates
        if top_p == 0:
          topv, topi = (torch.exp(decoder_output) + p_copy*copy_prob).data.topk(1)
        else:
          probs = F.softmax(decoder_output, dim=-1) + p_copy*copy_prob
          s_probs, s_inds = torch.sort(probs, descending=True)
          cum_probs = torch.cumsum(s_probs, dim=-1)

          # Remove all outside the nucleus
          sinds_to_remove = cum_probs > top_p

          # HuggingFace implementation did this to ensure first one is kept
          sinds_to_remove[:,:,1:] = sinds_to_remove[:,:,:-1].clone()
          sinds_to_remove[:,:,0] = 0

          for b in range(s_inds.size(1)):
            # Remove
            inds_to_remove = s_inds[:,b][sinds_to_remove[:,b]]

            # Set to be filtered in original
            probs[0,b,inds_to_remove] = 0

          # Sample
          topi = torch.multinomial((probs).squeeze(0), 1)

        topi = topi.view(-1)
        predictions[:, t] = topi

        # Set new last word
        last_word = topi.detach().view(1, -1)

    predicted_sentences = []
    for sentence in predictions:
      sent = []
      for ind in sentence:
        word = self.i2w[ind.long().item()]
        if word == '_eos':
          break
        sent.append(word)
      predicted_sentences.append(' '.join(sent))

    return predicted_sentences
          
  def save(self, name):
    torch.save(self, name)

  def load(self, name):
    self.load_state_dict(torch.load(name).state_dict())

class Transformer(nn.Module):
  def __init__(self, i2w, use_knowledge, args, test=False):
    super(Transformer, self).__init__()

    self.args = args
    self.use_knowledge = use_knowledge

    # Vocab
    self.i2w = i2w
    self.w2i = {w:i for i,w in enumerate(i2w)}

    self.transformer = transformer.Transformer(len(i2w), len(i2w), src_pad_idx=self.w2i['_pad'], trg_pad_idx=self.w2i['_pad'])

    # Training
    if test:
      self.criterion = nn.CrossEntropyLoss(ignore_index=self.w2i['_pad'], reduction='sum')
    else:
      self.criterion = nn.CrossEntropyLoss(ignore_index=self.w2i['_pad'])
    self.optim = optim.Adam(lr=args.lr, params=self.parameters(), betas=(0.9, 0.997), eps=1e-09)

  def prep_batch(self, rows, wh=64, wk=64):
    def _pad(arr, pad):
      # Given an array of integer arrays, pad all arrays to the same length
      lengths = [len(e) for e in arr]
      max_len = max(lengths)
      return [e+[pad]*(max_len-len(e)) for e in arr], lengths

    def _sample(arr):
      arr = arr[1:-1]
      #rate = random.random()
      rate = 0.5
      new = random.sample(range(len(arr)), int(rate * min(1000, len(arr))))
      new = [arr[i] for i in sorted(new)]
      # TODO: add shuffle
      #random.shuffle(new)
      return new

    # Split all rows
    rows = [[e.split() for e in row] for row in rows]

    # Form inputs
    if self.use_knowledge:
      inputs = [row[0][-wh:] + row[2][:wk] + ['_eos'] + ['_eos'] for row in rows]
    else:
      inputs = [row[0][-wh:] for row in rows]

    # Input seq 
    inputs = [[self.w2i.get(w, self.w2i['_unk']) for w in inp] for inp in inputs]
    input_seq, input_lens = _pad(inputs, self.w2i['_pad'])
    input_seq = torch.cuda.LongTensor(input_seq)

    # Target seq
    targets = [[self.w2i.get(w, self.w2i['_unk']) for w in row[1]] for row in rows]
    target_seq, target_lens = _pad(targets, pad=self.w2i['_pad'])
    target_seq = torch.cuda.LongTensor(target_seq)

    return input_seq, input_lens, target_seq, target_lens

  def forward(self, input_seq, input_lens, target_seq, target_lens):
    return self.transformer(input_seq, target_seq)

  def train(self, input_seq, input_lens, target_seq, target_lens):
    self.optim.zero_grad()

    # Forward
    proba = self.forward(input_seq, input_lens, target_seq, target_lens)

    # Loss
    loss = self.criterion(proba.view(-1, proba.size(-1)), target_seq.flatten())

    # Backwards
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip)
    self.optim.step()

    return loss.item()

  def eval_ppl(self, input_seq, input_lens, target_seq, target_lens):
    # Forward
    proba = self.forward(input_seq, input_lens, target_seq, target_lens)

    # Loss
    loss = self.criterion(proba.view(-1, proba.size(-1)), target_seq.flatten())

    return loss.item()

  def decode(self, input_seq, input_lens, top_p=0, max_len=100):
    batch_size = input_seq.size(0)
    predictions = [['_go'] for _ in range(batch_size)]
    eos_seen = [False for _ in range(batch_size)]

    def _pad(arr, pad):
      # Given an array of integer arrays, pad all arrays to the same length
      lengths = [len(e) for e in arr]
      max_len = max(lengths)
      return [e+[pad]*(max_len-len(e)) for e in arr], lengths

    with torch.no_grad():
      enc_output = self.transformer.enc(input_seq)

      for t in range(max_len):
        # Create the targets so far
        targets = [[self.w2i.get(w, self.w2i['_unk']) for w in row + ['_pad']] for row in predictions]
        target_seq, target_lens = _pad(targets, pad=self.w2i['_pad'])
        target_seq = torch.cuda.LongTensor(target_seq)

        # Pass through transformer
        proba = F.softmax(self.transformer(input_seq, target_seq, enc_output=enc_output), dim=-1)[:,-1]

        # Get top candidates
        if top_p == 0:
          topv, topi = proba.topk(1)
        else:
          s_probs, s_inds = torch.sort(proba, descending=True)
          cum_probs = torch.cumsum(s_probs, dim=-1)

          # Remove all outside the nucleus
          sinds_to_remove = cum_probs > top_p

          # HuggingFace implementation did this to ensure first one is kept
          sinds_to_remove[:,1:] = sinds_to_remove[:,:-1].clone()
          sinds_to_remove[:,0] = 0

          for b in range(s_inds.size(0)):
            # Remove
            inds_to_remove = s_inds[b][sinds_to_remove[b]]

            # Set to be filtered in original
            proba[b,inds_to_remove] = 0

          # Sample
          topi = torch.multinomial(proba.squeeze(0), 1)

        topi = topi.view(-1)
        words = [self.i2w[e.item()] for e in topi]
        for i in range(len(predictions)):
          predictions[i].append(words[i])
          if words[i] == '_eos':
            eos_seen[i] = True

        if all(eos_seen):
          break

    predicted_sentences = []
    for sentence in predictions:
      predicted_sentences.append(' '.join(sentence[1:-1 if '_eos' not in sentence else sentence.index('_eos')]))

    return predicted_sentences
          
  def save(self, name):
    torch.save(self, name)

  def load(self, name):
    self.load_state_dict(torch.load(name).state_dict())

