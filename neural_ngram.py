import argparse
import math
from collections import namedtuple

import numpy as np
import tensorflow as tf

from utils import *
from rnn_lm import evaluate_nearest_embedding

class NeuralNgram():

  def __init__(self, vocab_sz=50000, embedding_sz=128, window_sz=2, h_sz=100,
               lr=5e-4, num_sampled=6, reg=0):
    self.vocab_sz = vocab_sz
    self.embedding_sz = embedding_sz
    self.window_sz = window_sz
    self.h_sz = h_sz
    self.lr = lr
    self.num_sampled = num_sampled
    self.reg = reg
    self._create_graph()

  def _create_placeholders(self):
    with tf.variable_scope('placeholders'):
      self._inputs = tf.placeholder(tf.int32, shape=[None, self.window_sz],
                                    name = 'X')
      self.outputs = tf.placeholder(tf.int64, shape=[None, 1],
                                    name='y')
      self.similarity_word_ids = tf.placeholder(tf.int32, shape=[None,],
                                                name='similarity_word_ids')

  def _create_embeddings(self):
    with tf.name_scope('embeddings'):
      self.input_embeddings = tf.Variable(
          tf.random_uniform([self.vocab_sz, self.embedding_sz], -1.0, 1.0))

  def _create_hidden_weights(self):
    with tf.name_scope('hidden_weights'):
      self.W_h =  tf.Variable(
          tf.truncated_normal(
              [self.window_sz * self.embedding_sz, self.h_sz],
              stddev=1.0 )/math.sqrt(self.h_sz))
      self.b_h = tf.Variable(tf.zeros([self.h_sz]))

  def _create_output_weights(self):
    with tf.name_scope('output_weights'):
      self.W_o =  tf.Variable(
          tf.truncated_normal(
              [self.h_sz, self.vocab_sz],
              stddev=1.0 / math.sqrt(self.embedding_sz)))
      self.b_o = tf.Variable(tf.zeros([self.vocab_sz]))

  def _create_loss(self):
    with tf.name_scope('loss'):
      embed = tf.nn.embedding_lookup(self.input_embeddings, self._inputs)
      embed = tf.reshape(embed, [-1, self.window_sz * self.embedding_sz])
      h = tf.nn.relu(tf.matmul(embed, self.W_h) + self.b_h)
      self.loss = tf.reduce_mean(
                        tf.nn.nce_loss(
                            weights=tf.transpose(self.W_o),
                            biases=self.b_o,
                            labels=self.outputs,
                            inputs=h,
                            num_sampled=self.num_sampled,
                            num_classes=self.vocab_sz),
                        name = 'loss')

  def _create_regularization(self):
    with tf.name_scope('regularized-loss'):
      self.loss = self.loss \
                + 0.5 * self.reg * tf.reduce_sum(self.input_embeddings ** 2) \
                + 0.5 * self.reg * tf.reduce_sum(self.W_h ** 2) \
                + 0.5 * self.reg * tf.reduce_sum(self.W_o ** 2)

  def _create_optimizer(self):
    with tf.name_scope('optimizer'):
      self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

  def _embedding_similarity(self):
    with tf.name_scope('evaluating_similarity'):
      # Compute the cosine similarity between minibatch examples and all embeddings.
      norm = tf.sqrt(tf.reduce_sum(tf.square(self.input_embeddings), 1, keepdims=True))
      normalized_embeddings = self.input_embeddings / norm
      normalized_embed = tf.nn.embedding_lookup(normalized_embeddings,
                                                self.similarity_word_ids)
      self.similarity = tf.matmul(
          normalized_embed, normalized_embeddings, transpose_b=True)

  def _create_summaries(self):
    with tf.name_scope("summaries"):
      tf.summary.scalar("loss", self.loss)

      self.summary_op = tf.summary.merge_all()

  def _create_graph(self):
    self._create_placeholders()
    self._create_embeddings()
    self._create_hidden_weights()
    self._create_output_weights()
    self._create_loss()
    self._create_regularization()
    self._create_optimizer()
    self._embedding_similarity()
    self._create_summaries()

def train(model, data, labels, runs = 100001, bs = 128):

  similarity_eval_ids = np.random.choice(100, 16, replace=False)

  config = tf.ConfigProto()

  with tf.Session() as sess:
    with tf.device('/cpu:0'):
      print('starting:')
      sess.run(tf.global_variables_initializer())

      average_loss = 0

      for r in range(runs):
        _inputs, outputs = get_minibatch(data, labels, bs)

        outputs = outputs.reshape((-1, 1))
        _, current_loss = sess.run([model.train_op, model.loss],
                                    feed_dict={model._inputs: _inputs,
                                               model.outputs: outputs})
        average_loss += current_loss

        if r % 100 == 0:
          if not r == 0:
            average_loss /= 100
          print('@ run: ', r, ' average loss: ', average_loss)
          average_loss = 0
          [sim] = sess.run([model.similarity],
                    feed_dict={model.similarity_word_ids: similarity_eval_ids})

          evaluate_nearest_embedding(sim, similarity_eval_ids, id2word)
          print('')

if __name__ == '__main__':

  Config = namedtuple('Config',
                      ['bs', 'model', 'lr', 'embedding_sz', 'vocab_sz',
                        'window_sz', 'num_sampled', 'runs'])

  parser = argparse.ArgumentParser()
  parser.add_argument('--bs', type = int, default = 128)
  parser.add_argument('--model', default = 'cbow')
  parser.add_argument('--lr', type = float, default = 5e-2)
  parser.add_argument('--embedding-sz', type = int, default = 10)
  parser.add_argument('--vocab-sz', type = int, default = 50000)
  parser.add_argument('--window-sz', type = int, default = 2)
  parser.add_argument('--num-sampled', type = int, default = 6)
  parser.add_argument('--runs', type = int, default = 100001)

  args = parser.parse_args()

  config = Config(bs = args.bs, model = args.model,
                  lr = args.lr, embedding_sz = args.embedding_sz,
                  vocab_sz = args.vocab_sz, window_sz = args.window_sz,
                  num_sampled = args.num_sampled, runs = args.runs)

  model = NeuralNgram(config.vocab_sz, config.embedding_sz, config.window_sz,
                 100, config.lr, config.num_sampled, reg=0)

  _dir = './data/'
  download_text8(url, filename, _dir)
  data = read_data(_dir + filename)
  data, counts, word2id, id2word = build_dataset(data, config.vocab_sz)

  data, labels = generate_neural_ngram_data(data, window_sz=config.window_sz)

  train(model, data, labels, config.runs, config.bs)
