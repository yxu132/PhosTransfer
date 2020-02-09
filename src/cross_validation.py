# Copyright 2016 Monash University
#
# Author: Ying Xu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy
import tensorflow as tf

class CrossValidation(object):

  def __init__(self, features, labels, extra_negatives=None, extra_labels=None):
    self._features = features
    self._labels = labels
    self._extra_negative_features = extra_negatives
    self._extra_negative_labels = extra_labels
    self._fold = 0
    self._numFold = 10

  def count(self):
      return len(self._features), len(self._labels)

  def nextFold(self):
    train_feature = self._features[0:self._fold]+self._features[self._fold+1:]
    train_label = self._labels[0:self._fold]+self._labels[self._fold+1:]
    test_features = self._features[self._fold]
    test_labels = self._labels[self._fold]

    train_features = train_feature[0]
    for x in train_feature[1:]:
      train_features = numpy.concatenate((train_features, x), axis=0)

    train_labels = train_label[0]
    for x in train_label[1:]:
      train_labels = numpy.concatenate((train_labels, x), axis=0)

    self._fold += 1
    return DataSet(train_features, train_labels,
                   extra_negatives=self._extra_negative_features, extra_labels=self._extra_negative_labels),\
           DataSet(test_features, test_labels, is_test=True)

  def get_i(self, i):
    return DataSet(self._features[i], self._labels[i], is_test=True)

  def all(self):
    train_feature = self._features
    train_label = self._labels
    train_features = train_feature[0]
    for x in train_feature[1:]:
      train_features = numpy.concatenate((train_features, x), axis=0)
    train_labels = train_label[0]
    for x in train_label[1:]:
      train_labels = numpy.concatenate((train_labels, x), axis=0)
    return DataSet(train_features, train_labels, is_test=True)


class DataSet(object):

  def __init__(self, features, labels, extra_negatives=None, extra_labels=None, dtype=tf.float32, is_test=False):
    """Construct a DataSet.

    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = tf.as_dtype(dtype).base_dtype
    if dtype not in (tf.uint8, tf.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    assert features.shape[0] == labels.shape[0], ('features.shape: %s labels.shape: %s' % (features.shape,labels.shape))
    self._num_examples = features.shape[0]
    # features = features.reshape(features.shape[0], features.shape[1] * features.shape[2] * features.shape[3])
    if dtype == tf.float32:
      # Convert from [0, 255] -> [0.0, 1.0].
      features = features.astype(numpy.float32)
      features = numpy.multiply(features, 1.0 / 255.0)
    self._features = features
    self._labels = labels
    self._extra_negative_features = extra_negatives
    self._extra_negative_labels = extra_labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

    # randomise at initialisation
    if not is_test:
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._features = self._features[perm]
      self._labels = self._labels[perm]

  @property
  def count(self):
      return len(self._features), len(self._labels)

  @property
  def features(self):
    return self._features

  @property
  def labels(self):
    return self._labels

  @property
  def disos(self):
    return self._disos

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed


  def next_batch(self, batch_size):
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      if self._extra_negative_features != None:
          extra_index = self._epochs_completed % len(self._extra_negative_features)
          extra_negatives = self._extra_negative_features[extra_index]
          extra_negatives = extra_negatives.astype(numpy.float32)
          extra_negatives = numpy.multiply(extra_negatives, 1.0 / 255.0)
          extra_labels = self._extra_negative_labels[extra_index]

          positive_features, positive_labels = [], []
          for ind, feature in enumerate(self._features):
              if self._labels[ind][1] == 1:
                  positive_features.append(feature)
                  positive_labels.append(self._labels[ind])
          features = numpy.concatenate([numpy.array(positive_features), numpy.array(extra_negatives)], axis=0)
          labels = numpy.concatenate([numpy.array(positive_labels), numpy.array(extra_labels)], axis=0)
          self._features = features
          self._labels = labels

      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(len(self._features))
      numpy.random.shuffle(perm)
      self._features = self._features[perm]
      self._labels = self._labels[perm]

      self._num_examples = len(self._features)
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples

    end = self._index_in_epoch
    return self._features[start:end], self._labels[start:end], None

  def all(self):
    return self._features[:], self._labels[:], None

