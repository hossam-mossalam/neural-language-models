import os
import urllib.request
import zipfile
from collections import Counter, defaultdict

import numpy as np
import tensorflow as tf
import wget

url = 'http://mattmahoney.net/dc/'
filename = 'text8.zip'

def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words."""
  with zipfile.ZipFile(filename) as f:
    data = f.read(f.namelist()[0]).decode('utf8').split()
  return data

def download_text8(url, filename, _dir):
  """ download text8 dataset """
  URL = url + filename
  original_dir = os.getcwd()
  if not os.path.exists(_dir):
    os.makedirs(_dir)
  if not os.path.exists(_dir + '/' + filename):
    os.chdir(_dir)
    filename = wget.download(URL)
    os.chdir(original_dir)

def build_dataset(data, vocab_sz):
  """Process raw inputs into a dataset."""
  counts = [['UNK', -1]]
  counts.extend(Counter(data).most_common(vocab_sz - 1))
  word2id = dict()
  id2word = dict()
  word2id['UNK'] = 0
  id2word[0] = 'UNK'
  current_id = 1
  for word, _ in counts[1:]:
    word2id[word] = current_id
    id2word[current_id] = word
    current_id += 1

  modified_data = []
  count_unk = 0
  for w in data:
    w_id = word2id.get(w, 0)
    if w_id == 0:
      count_unk += 1
    modified_data.append(w_id)
  counts[0][1] = count_unk
  return np.array(modified_data), counts, word2id, id2word

def build_char_dataset(data):
  """Process raw input into a character based dataset."""
  data = ' '.join(data)
  chars = list(set(data))
  print('Dataset has %d characters, and %d unique characters' \
        % (len(data), len(chars)))
  char2id = {}
  id2char = {}
  for i, c in enumerate(chars):
    char2id[c] = i
    id2char[i] = c
  modified_data = [char2id[c] for c in data]
  return np.array(modified_data), char2id, id2char, len(chars)

def generate_neural_ngram_data(data, window_sz = 2):
  data_ngram = []
  labels = []
  for i in range(window_sz, len(data)):
    center_id = data[i]
    ngram = data[i - window_sz:i]
    labels.append(center_id)
    data_ngram.append(ngram)

  return np.array(data_ngram), np.array(labels)

def generate_sequences(data, bs, seq_len):
  data_sz = len(data)
  batch_len = data_sz // bs
  data_batches = np.zeros((bs, batch_len), dtype=np.int32)
  for i in range(bs):
    data_batches[i] = data[batch_len * i: batch_len * (i + 1)]

  epoch_sequences = batch_len // seq_len

  if epoch_sequences == 0:
    raise ValueError('Epoch sequences are 0')

  print('Data consists of %d sequences.' % (epoch_sequences))

  for i in range(epoch_sequences):
    X = data_batches[:, seq_len * i: seq_len * (i + 1)]
    Y = data_batches[:, seq_len * i + 1: seq_len * (i + 1) + 1]
    yield X, Y

def get_minibatch(data, labels, bs):
  sz = len(labels)
  indices = np.random.choice(sz, bs)
  return data[indices], labels[indices]
