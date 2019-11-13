"""
Infinite evaluation loop going through the checkpoints in the model directory
as they appear and evaluating them. Accuracy and average loss are printed and
added as tensorboard summaries.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import math
import os
import sys
import time

import tensorflow as tf

import cifar10_input
from cifar10_class import Model
from pgd_attack import LinfPGDAttack

# Global constants
config_path = "config.json"

with open(config_path) as config_file:
    config = json.load(config_file)
num_eval_examples = config['num_eval_examples']
eval_batch_size = config['eval_batch_size']
eval_on_cpu = config['eval_on_cpu']
data_path = config['data_path']
opt_method = config["method"]

# Set upd the data, hyperparameters, and the model
cifar = cifar10_input.CIFAR10Data(data_path)

if eval_on_cpu:
  with tf.device("/cpu:0"):
    model = Model(mode='eval', data=cifar.train_images)
    attack = LinfPGDAttack(model,
                           config['epsilon'],
                           config['num_steps'],
                           config['step_size'],
                           config['random_start'],
                           config['loss_func'])
else:
  model = Model(mode='eval', data=cifar.train_images)
  model._build_model_easy()
  attack = LinfPGDAttack(model,
                         config['epsilon'],
                         config['num_steps'],
                         config['step_size'],
                         config['random_start'],
                         config['loss_func'])

global_step = tf.contrib.framework.get_or_create_global_step()


model_dir="./"
ckpt_file = "pretrained.ckpt"
# Setting up the Tensorboard and checkpoint outputs
if not os.path.exists(model_dir):
  os.makedirs(model_dir)
eval_dir = os.path.join(model_dir, 'eval')
if not os.path.exists(eval_dir):
  os.makedirs(eval_dir)

last_checkpoint_filename = ''
already_seen_state = False

var_list = tf.get_collection(
    tf.GraphKeys.GLOBAL_VARIABLES, scope="model")

saver = tf.train.Saver(var_list)
summary_writer = tf.summary.FileWriter(eval_dir)

# A function for evaluating a single checkpoint
def evaluate_checkpoint(filename):
  tf_config = tf.ConfigProto()
  #tf_config.gpu_options.per_process_gpu_memory_fraction=0.5
  tf_config.gpu_options.allow_growth = True
  with tf.Session(config=tf_config) as sess:
    # Restore the checkpoint
    sess.run(global_step.initializer)
    saver.restore(sess, filename)

    # Iterate over the samples batch-by-batch
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
    total_xent_nat = 0.
    total_xent_adv = 0.
    total_corr_nat = 0
    total_corr_adv = 0

    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)

      x_batch = cifar.eval_data.xs[bstart:bend, :]
      y_batch = cifar.eval_data.ys[bstart:bend]

      dict_nat = {model.x_input: x_batch,
                  model.y_input: y_batch}

      x_batch_adv = attack.perturb(x_batch, y_batch, sess)

      dict_adv = {model.x_input: x_batch_adv,
                  model.y_input: y_batch}

      cur_corr_nat, cur_xent_nat = sess.run(
                                      [model.num_correct,model.xent],
                                      feed_dict = dict_nat)
      cur_corr_adv, cur_xent_adv = sess.run(
                                      [model.num_correct,model.xent],
                                      feed_dict = dict_adv)

      print(eval_batch_size)
      print("Correctly classified natural examples: {}".format(cur_corr_nat))
      print("Correctly classified adversarial examples: {}".format(cur_corr_adv))
      total_xent_nat += cur_xent_nat
      total_xent_adv += cur_xent_adv
      total_corr_nat += cur_corr_nat
      total_corr_adv += cur_corr_adv

    avg_xent_nat = total_xent_nat / num_eval_examples
    avg_xent_adv = total_xent_adv / num_eval_examples
    acc_nat = total_corr_nat / num_eval_examples
    acc_adv = total_corr_adv / num_eval_examples

    summary = tf.Summary(value=[
          tf.Summary.Value(tag='xent adv eval', simple_value= avg_xent_adv),
          tf.Summary.Value(tag='xent adv', simple_value= avg_xent_adv),
          tf.Summary.Value(tag='xent nat', simple_value= avg_xent_nat),
          tf.Summary.Value(tag='accuracy adv eval', simple_value= acc_adv),
          tf.Summary.Value(tag='accuracy adv', simple_value= acc_adv),
          tf.Summary.Value(tag='accuracy nat', simple_value= acc_nat)])
    summary_writer.add_summary(summary, global_step.eval(sess))

    print('natural: {:.2f}%'.format(100 * acc_nat))
    print('adversarial: {:.2f}%'.format(100 * acc_adv))
    print('avg nat loss: {:.4f}'.format(avg_xent_nat))
    print('avg adv loss: {:.4f}'.format(avg_xent_adv))


# Infinite eval loop
print('\nCheckpoint {}, evaluating ...   ({})'.format(ckpt_file,
                                                      datetime.now()))


evaluate_checkpoint(ckpt_file)

