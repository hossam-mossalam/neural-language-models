import math
from abc import ABCMeta, abstractmethod

import numpy as np
import tensorflow as tf

from rnn_lm import *
from utils import *

class CharRNNLM(RNNLM):

  def __init__(self, bs=32, vocab_sz=26, embedding_sz=32, cell_sz=128,
               seq_len=128, lr=5e-4, reg=0,
               initializer=tf.contrib.layers.xavier_initializer(),
               optimizer=tf.train.AdamOptimizer):

    super(CharRNNLM, self).__init__(bs, vocab_sz, embedding_sz, cell_sz,
                                    seq_len, lr, reg, initializer, optimizer)
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
    target_output = tf.reshape(self.outputs, [-1])

    logits = tf.matmul(cell_output, self.W) + self.b

    with tf.name_scope('loss'):
      self.loss = tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=target_output,
                            logits=logits),
                        name = 'loss')

  def _create_regularization(self):
    pass

  def _create_model_summaries(self):
    pass


def train(model, data, char2id, id2char, runs=10, bs=32, vocab_sz=26,
          seq_len=128):

  similarity_eval_ids = np.random.choice(vocab_sz, 16, replace=False)

  print('training')
  config = tf.ConfigProto()

  with tf.Session() as sess:
    with tf.device('/cpu:0'):
      print('starting:')
      sess.run(tf.global_variables_initializer())

      average_loss = 0

      for r in range(runs):

        state_c = np.zeros(model.cell_state_c.shape)
        state_h = np.zeros(model.cell_state_h.shape)

        for i, (X, Y) in enumerate(generate_sequences(data, bs, seq_len)):

          _, current_loss, state_c, state_h = sess.run(
                                      [model.train_op, model.loss,
                                       model.next_state.c, model.next_state.h],
                                      feed_dict={model.inputs: X,
                                                 model.outputs: Y,
                                                 model.cell_state_c: state_c,
                                                 model.cell_state_h: state_h})

          average_loss += current_loss

          if i % 100 == 0:
            if not i == 0:
              average_loss /= 1000
            print('@ run: ', r, ' batch: ', i, ' average loss: ', average_loss)
            average_loss = 0
            if i % 1000 == 0:
              print('Sample Sequence:')
              sample(model, sess, seq_len, char2id, id2char,
                     vocab_sz, delimiter='')


if __name__ == '__main__':
  _dir = './data/'
  download_text8(url, filename, _dir)
  data = read_data(_dir + filename)
  data, char2id, id2char, vocab_sz = build_char_dataset(data)
  seq_len = 128
  bs = 32

  model = CharRNNLM(bs, vocab_sz, embedding_sz = 32,
               cell_sz = 128, seq_len = seq_len,
               lr = 5e-4, reg = 0)

  print('created model')
  train(model, data, char2id, id2char, vocab_sz=vocab_sz, seq_len=seq_len)
