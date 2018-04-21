import math
from abc import ABCMeta, abstractmethod

import numpy as np
import tensorflow as tf

from utils import *

def sample(model, sess, seq_len, vocab2id, id2vocab, vocab_sz, delimiter=' ',
           start_element_id=None):
  if not start_element_id:
    current_id = np.random.choice(int(vocab_sz * 0.1))
  else:
    current_id = start_element_id
  current_element = id2vocab[current_id]
  s = [current_element]

  current_state_c = np.zeros(model.sample_state_c.shape)
  current_state_h = np.zeros(model.sample_state_h.shape)

  for _ in range(seq_len):
    current_logits, current_state_c, current_state_h = \
            sess.run([model.sample_logits, model.next_sample_state.c,
                      model.next_sample_state.h],
                      feed_dict={model.sample_input: [[current_id]],
                                 model.sample_state_c: current_state_c,
                                 model.sample_state_h: current_state_h})
        # feed_dict={model.sample_input: np.array([[current_id]]),

    current_logits = current_logits[0]
    current_probs = np.exp(current_logits) / np.sum(np.exp(current_logits))
    current_id = np.random.choice(vocab_sz, p=current_probs)
    current_element = id2vocab[current_id]
    s += [current_element]
  print(delimiter.join(s))
  print()

def evaluate_nearest_embedding(sim, validation_ids, id2word):
  for i, valid_id in enumerate(validation_ids):
    valid_word = id2word[valid_id]
    top_k = 10  # number of nearest neighbors
    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
    log_str = 'Nearest to %s:' % valid_word
    for k in range(top_k):
      close_word = id2word[nearest[k]]
      log_str = '%s %s,' % (log_str, close_word)
    print(log_str)

class RNNLM(metaclass=ABCMeta):

  def __init__(self, bs, vocab_sz, embedding_sz, cell_sz, seq_len, lr, reg,
               initializer, optimizer):
    self.bs = bs
    self.vocab_sz = vocab_sz
    self.embedding_sz = embedding_sz
    self.cell_sz = cell_sz
    self.seq_len = seq_len
    self.lr = lr
    self.reg = reg
    self.initializer = initializer
    self.optimizer = optimizer

  def _create_placeholders(self):
    with tf.variable_scope('placeholders'):
      self.inputs  = tf.placeholder(tf.int32, shape=[self.bs, self.seq_len],
                                    name='X')
      self.outputs = tf.placeholder(tf.int32, shape=[self.bs, self.seq_len],
                                    name='y')


  def _create_global_counter(self):
    with tf.name_scope('global_step'):
      self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False,
                                     name='global_step')

  def _create_embeddings(self):
    with tf.name_scope('embeddings'):
      self.input_embeddings = tf.Variable(
          tf.random_uniform([self.vocab_sz, self.embedding_sz], -1.0, 1.0),
          name='embeddings')

  @abstractmethod
  def _create_rnncell(self):
    pass

  def _create_output_weights(self):
    with tf.variable_scope('output_layer'):
      self.W = tf.get_variable('W',
                  [self.cell_sz, self.vocab_sz],
                  initializer=self.initializer)

      self.b = tf.get_variable('b', [self.vocab_sz],
                  initializer=tf.zeros_initializer)

  @abstractmethod
  def _create_loss(self):
    pass

  @abstractmethod
  def _create_regularization(self):
    pass

  def _create_optimizer(self):
    with tf.name_scope('optimizer'):
      self.train_op = self.optimizer(self.lr).minimize(
                                                self.loss,
                                                global_step=self.global_step)

  def _create_summaries(self):
    with tf.name_scope("summaries"):
      tf.summary.scalar("loss", self.loss)

  @abstractmethod
  def _create_model_summaries(self):
    pass

  def _merge_summaries(self):
    with tf.name_scope('merging_summaries'):
      self.summary_op = tf.summary.merge_all()

  def _embedding_similarity(self):
    with tf.name_scope('similarity_placeholder'):
      self.similarity_word_ids = tf.placeholder(tf.int32, shape=[None,],
                                                name='word_ids')

    # Compute cosine similarity between minibatch examples and all embeddings.
    with tf.name_scope('evaluating_similarity'):
      norm = tf.sqrt(tf.reduce_sum(tf.square(self.input_embeddings),
                                   axis=1, keepdims=True))
      normalized_embeddings = self.input_embeddings / norm
      normalized_embed = tf.nn.embedding_lookup(normalized_embeddings,
                                                self.similarity_word_ids)
      self.similarity = tf.matmul(
          normalized_embed, normalized_embeddings, transpose_b=True)

  def _sample(self):
    with tf.name_scope('sample_placeholder'):
      self.sample_input = tf.placeholder(tf.int32, shape=[None,1],
                                                name = 'sample_input')

    with tf.name_scope('sample_embed'):
      embed = tf.nn.embedding_lookup(self.input_embeddings, self.sample_input)

    with tf.name_scope('sample_output'):
      (cell_output, self.next_sample_state) = tf.nn.dynamic_rnn(
                                              self.cell, embed,
                                              initial_state=self.sample_state)

    cell_output = tf.reshape(cell_output, [-1, self.cell_sz])

    self.sample_logits = tf.matmul(cell_output, self.W) + self.b



  def _create_cell_state(self):
    state_sz = self.cell.state_size
    with tf.name_scope('cell_state'):
      self.cell_state_c = tf.placeholder(tf.float32, [self.bs, state_sz.c])
      self.cell_state_h = tf.placeholder(tf.float32, [self.bs, state_sz.h])
    self.cell_state = tf.nn.rnn_cell.LSTMStateTuple(self.cell_state_c,
                                                    self.cell_state_h)

  def _create_sample_state(self):
    state_sz = self.cell.state_size
    with tf.name_scope('cell_state'):
      self.sample_state_c = tf.placeholder(tf.float32, [1, state_sz.c])
      self.sample_state_h = tf.placeholder(tf.float32, [1, state_sz.h])
    self.sample_state = tf.nn.rnn_cell.LSTMStateTuple(self.sample_state_c,
                                                      self.sample_state_h)

  def _create_graph(self):
    self._create_placeholders()
    self._create_global_counter()
    self._create_embeddings()
    self._create_rnncell()
    self._create_cell_state()
    self._create_sample_state()
    self._create_output_weights()
    self._create_loss()
    self._create_regularization()
    self._create_optimizer()
    self._create_summaries()
    self._create_model_summaries()
    self._merge_summaries()
    self._sample()
    self._embedding_similarity()



