# Process the dataset and save in a friendlier format for the 
# sequence generation task: (context, response, best fact)

import json
import spacy
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

def clean(s):
  return ''.join([c for c in s.lower() if c not in string.punctuation])

def build_tfidf():
  vectorizer = TfidfVectorizer()
  corpus = []
  for split in ['train', 'valid_freq', 'valid_rare', 'test_freq', 'test_rare']:
    conversations = json.load(open("conversations/{0}.json".format(split)))
    reading_sets = json.load(open("reading_sets/post-build/{0}.json".format(split)))
    print("Building TFIDF for split", split)
    for dkey in tqdm(conversations.keys()):
      # Add history
      corpus.append(' '.join([turn["message"] for turn in conversations[dkey]["content"]]))

      # Add facts
      for source in reading_sets[dkey]["agent_1"].values():
        corpus += source["fun_facts"]

      for source in reading_sets[dkey]["agent_2"].values():
        corpus += source["fun_facts"]

      for article_paragraph in ["AS1", "AS2", "AS3", "AS4"]:
        if article_paragraph in reading_sets[dkey]["article"]:
          corpus.append(reading_sets[dkey]["article"][article_paragraph])

  # Fit TF-IDF
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

def process(split):
  conversations = json.load(open("conversations/{0}.json".format(split)))
  reading_sets = json.load(open("reading_sets/post-build/{0}.json".format(split)))

  # Iterate over the dialogs
  contexts = []
  responses = []
  facts = []
  print("Processing", split)
  for dkey in tqdm(conversations.keys()):
    history = ""
    for turn in conversations[dkey]["content"]:
      if len(history) > 0:
        contexts.append(history)
        responses.append("_go " + turn["message"] + " _eos")

        # Find the right fact
        potential_facts = []
        for fact_source in turn["knowledge_source"]:
          if fact_source.startswith("F"):
            potential_facts += reading_sets[dkey][turn["agent"]][fact_source]["fun_facts"]
          elif fact_source.startswith("A"):
            potential_facts.append(reading_sets[dkey]["article"][fact_source])

        if len(potential_facts) == 0:
          facts.append("_nofact")
        else:
          # TF-IDF
          r_vec = vectorizer.transform([clean(turn["message"])])
          f_vec = vectorizer.transform([clean(e) for e in potential_facts])
          sim = r_vec.dot(f_vec.transpose()).todense()

          # Add best fact
          facts.append(potential_facts[sim.argmax()])

    
      # Update history
      history += " " + turn["message"] + " _eos"
      

  contexts = tokenize(contexts)
  responses = tokenize(responses)
  facts = tokenize(facts)
  open("processed_output2/{0}.src".format(split), "w+").writelines([e.strip()+"\n" for e,f in zip(contexts, facts)])
  open("processed_output2/{0}.fct".format(split), "w+").writelines([f.strip()+"\n" for e,f in zip(contexts, facts)])
  open("processed_output2/{0}.tgt".format(split), "w+").writelines([e.strip()+"\n" for e in responses])

vectorizer = build_tfidf()
for split in ['train', 'valid_freq', 'valid_rare', 'test_freq', 'test_rare']:
  process(split)
