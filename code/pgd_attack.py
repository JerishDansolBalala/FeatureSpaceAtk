"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import cifar10_input


def l2_norm(x):
  l2norm = np.sqrt(np.sum(np.multiply(x, x), axis=(1,2,3)))
  return l2norm

class LinfPGDAttack:
  def __init__(self, loss_func, x_input, y_input, epsilon, num_steps, step_size, random_start):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
    self.loss_func = loss_func
    self.epsilon = epsilon
    self.num_steps = num_steps
    self.step_size = step_size
    self.rand = random_start

    #self.ray.init(num_cpus=2)

    loss = loss_func
    self.x_input = x_input
    self.y_input = y_input

    self.grad = tf.gradients(loss, self.x_input)[0]
    grad_l2_norm = tf.sqrt(tf.reduce_sum(tf.square(self.grad), axis=[1,2,3],keep_dims=True))
    self.grad_l2 = self.grad / (grad_l2_norm + 1e-5) 

  def perturb(self, x_nat, y, sess):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
    if self.rand:
      x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
      x = np.clip(x, 0, 255)  # ensure valid pixel range
    else:
      x = np.copy(x_nat)

    for i in range(self.num_steps):
      grad = sess.run(self.grad, feed_dict={self.x_input: x,
                                            self.y_input: y})
      grad = np.nan_to_num(grad)
      if i%10 ==0:
        loss = sess.run(self.loss_func, feed_dict={self.x_input: x,
                                            self.y_input: y})
        print(loss)

      x = np.add(x, self.step_size * np.sign(grad), out=x, casting='unsafe')

      x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
      x = np.clip(x, 0, 255)  # ensure valid pixel range

    return x

  def perturb_l2(self, x_nat, y, sess):
    #0.2*255
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
    xplus_lower = 0.0 - x_nat
    xplus_upper = 255.0 - x_nat
    bs = x_nat.shape[0]
    if self.rand:
      x_plus = np.random.uniform(low=-0.001, high=0.001, size=x_nat.shape)
      x_plus = np.clip(x_plus, xplus_lower, xplus_upper)  # ensure valid pixel range
    else:
      x_plus = np.zeros(x_nat.shape)
    #print(x_plus)
    x = x_nat + x_plus
    for i in range(self.num_steps):
      grad = sess.run(self.grad_l2, feed_dict={self.x_input: x,
                                            self.y_input: y})
      #grad = np.nan_to_num(grad)
      if i%10 ==0:
        loss = sess.run(self.loss_func, feed_dict={self.x_input: x,
                                            self.y_input: y})
        print(loss)
      x_plus = np.add(x_plus, self.step_size* self.epsilon * grad)
      x_plus = np.clip(x_plus, xplus_lower, xplus_upper)

      l2 = l2_norm(x_plus)
      #print(l2[0],end="\t")
      for j in range(bs):
        if l2[j] > self.epsilon:
          x_plus[j, :, :, :] = x_plus[j, :, :, :] / l2[j] * self.epsilon
      x = x_nat + x_plus

    return x


class LinfPGDAttack_org:
  def __init__(self, model, epsilon, num_steps, step_size, random_start, loss_func):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
    self.model = model
    self.epsilon = epsilon
    self.num_steps = num_steps
    self.step_size = step_size
    self.rand = random_start

    #self.ray.init(num_cpus=2)

    if loss_func == 'xent':
      loss = model.xent
    elif loss_func == 'cw':
      label_mask = tf.one_hot(model.y_input,
                              10,
                              on_value=1.0,
                              off_value=0.0,
                              dtype=tf.float32)
      correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
      wrong_logit = tf.reduce_max((1-label_mask) * model.pre_softmax - 1e4*label_mask, axis=1)
      loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
    else:
      print('Unknown loss function. Defaulting to cross-entropy')
      loss = model.xent

    self.grad = tf.gradients(loss, model.x_input)[0]



  def perturb(self, x_nat, y, sess):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
    if self.rand:
      x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
      x = np.clip(x, 0, 255) # ensure valid pixel range
    else:
      x = np.copy(x_nat)

    for i in range(self.num_steps):
      grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
                                            self.model.y_input: y})

      """grad_1,grad_2=np.split(grad,2)
      x_1,x_2=np.split(x,2)
      x=self.ray.get([f.remote(x_1, grad_1, self.step_size)],
                   [f.remote(x_2, grad_2, self.step_size)])
      x=np.concatenate(x)"""
      x = np.add(x, self.step_size * np.sign(grad), out=x, casting='unsafe')

      x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
      x = np.clip(x, 0, 255) # ensure valid pixel range


    return x


if __name__ == '__main__':
  import json
  import sys
  import math


  from model import Model

  with open('config.json') as config_file:
    config = json.load(config_file)

  model_file = tf.train.latest_checkpoint(config['model_dir'])
  if model_file is None:
    print('No model found')
    sys.exit()

  model = Model(mode='eval')
  attack = LinfPGDAttack(model,
                         config['epsilon'],
                         config['num_steps'],
                         config['step_size'],
                         config['random_start'],
                         config['loss_func'])
  saver = tf.train.Saver()

  data_path = config['data_path']
  cifar = cifar10_input.CIFAR10Data(data_path)

  with tf.Session() as sess:
    # Restore the checkpoint
    saver.restore(sess, model_file)

    # Iterate over the samples batch-by-batch
    num_eval_examples = config['num_eval_examples']
    eval_batch_size = config['eval_batch_size']
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

    x_adv = [] # adv accumulator

    print('Iterating over {} batches'.format(num_batches))

    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)
      print('batch size: {}'.format(bend - bstart))

      x_batch = cifar.eval_data.xs[bstart:bend, :]
      y_batch = cifar.eval_data.ys[bstart:bend]

      x_batch_adv = attack.perturb(x_batch, y_batch, sess)

      x_adv.append(x_batch_adv)

    print('Storing examples')
    path = config['store_adv_path']
    x_adv = np.concatenate(x_adv, axis=0)
    np.save(path, x_adv)
    print('Examples stored in {}'.format(path))
