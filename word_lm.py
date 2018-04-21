import math
from abc import ABCMeta, abstractmethod

import numpy as np
import tensorflow as tf

from rnn_lm import *
from utils import *

class WordRNNLM(RNNLM):

  def __init__(self, bs=128, vocab_sz=50000, embedding_sz=128, cell_sz=128,
               seq_len=256, lr=5e-4, reg=0, num_sampled=20,
               initializer=tf.contrib.layers.xavier_initializer(),
               optimizer=tf.train.AdamOptimizer):

    super(WordRNNLM, self).__init__(bs, vocab_sz, embedding_sz, cell_sz,
                                    seq_len, lr, reg, initializer, optimizer)
    self.num_sampled = num_sampled
    self._create_graph()

  def _create_rnncell(self):
    with tf.name_scope('rnn_cell'):
      self.cell = tf.nn.rnn_cell.LSTMCell(self.cell_sz)

  def _create_loss(self):
    with tf.name_scope('embed'):
      embed = tf.nn.embedding_lookup(self.input_embeddings, self.inputs)

    with tf.name_scope('cell_output'):
      (cell_output, self.next_state) = tf.nn.dynamic_rnn(
                                            self.cell, embed,
                                            initial_state=self.cell_state)

    cell_output = tf.reshape(cell_output, [-1, self.cell_sz])
    target_output = tf.reshape(self.outputs, [-1, 1])

    with tf.name_scope('loss'):
      self.loss = tf.reduce_mean(
                        tf.nn.nce_loss(
                            weights=tf.transpose(self.W),
                            biases=self.b,
                            labels=target_output,
                            inputs=cell_output,
                            num_sampled=self.num_sampled,
                            num_classes=self.vocab_sz),
                        name = 'loss')

  def _create_regularization(self):
    pass

  def _create_model_summaries(self):
    pass


def train(model, data, word2id, id2word, runs=10, bs=32, vocab_sz=50000):

  similarity_eval_ids = np.random.choice(100, 16, replace=False)
  sample_seq_len = 32

  print('training')
  config = tf.ConfigProto()

  with tf.Session() as sess:
    with tf.device('/cpu:0'):
      print('starting:')
      sess.run(tf.global_variables_initializer())

      average_loss = 0

      for r in range(runs):

        average_loss = 0

        state_c = np.zeros(model.cell_state_c.shape)
        state_h = np.zeros(model.cell_state_h.shape)

        for i, (X, Y) in enumerate(generate_sequences(data, bs, 32)):

          _, current_loss, state_c, state_h = sess.run(
                                      [model.train_op, model.loss,
                                       model.next_state.c, model.next_state.h],
                                      feed_dict={model.inputs: X,
                                                 model.outputs: Y,
                                                 model.cell_state_c: state_c,
                                                 model.cell_state_h: state_h})

          average_loss += current_loss

          if i % 1000 == 0:
            if not i == 0:
              average_loss /= 1000
            print('@ run: ', r, ' batch: ', i, ' average loss: ', average_loss)
            average_loss = 0
            print('Sample Sequence:')
            sample(model, sess, sample_seq_len, word2id, id2word, vocab_sz)
            print('Similarity: ')
            [sim] = sess.run([model.similarity],
                          feed_dict={
                              model.similarity_word_ids: similarity_eval_ids})

            evaluate_nearest_embedding(sim, similarity_eval_ids, id2word)
            print('')


if __name__ == '__main__':
  vocab_sz = 50000
  bs = 64

  _dir = './data/'
  download_text8(url, filename, _dir)
  data = read_data(_dir + filename)
  data, counts, word2id, id2word = build_dataset(data, vocab_sz)

  model = WordRNNLM(bs=bs, vocab_sz=vocab_sz, embedding_sz=128, cell_sz=128,
                    seq_len=32, lr=5e-4, num_sampled=6, reg=0)

  train(model, data, word2id, id2word, runs=100, bs=bs, vocab_sz=vocab_sz)
