"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import shutil
from timeit import default_timer as timer

import tensorflow as tf
import numpy as np

from cifar10_class import Model
import cifar10_input
from pgd_attack import LinfPGDAttack_org


with open('config.json') as config_file:
    config = json.load(config_file)

# seeding randomness
tf.set_random_seed(config['tf_random_seed'])
np.random.seed(config['np_random_seed'])

# Setting up training parameters
max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
step_size_schedule = config['step_size_schedule']
weight_decay = config['weight_decay']
data_path = config['data_path']
momentum = config['momentum']
batch_size = config['training_batch_size']

# Setting up the data and the model
raw_cifar = cifar10_input.CIFAR10Data(data_path)
global_step = tf.contrib.framework.get_or_create_global_step()
model = Model(mode='train',data=raw_cifar.train_images)
model._build_model_easy()

# Setting up the optimizer
boundaries = [int(sss[0]) for sss in step_size_schedule]
boundaries = boundaries[1:]
values = [sss[1] for sss in step_size_schedule]
learning_rate = tf.train.piecewise_constant(
    tf.cast(global_step, tf.int32),
    boundaries,
    values)
total_loss = model.mean_xent + weight_decay * model.weight_decay_loss
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
  train_step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(
    total_loss,
    global_step=global_step)

# Set up adversary
attack = LinfPGDAttack_org(model,
                       config['epsilon'],
                       config['num_steps'],
                       config['step_size'],
                       config['random_start'],
                       config['loss_func'])

# Setting up the Tensorboard and checkpoint outputs

adv_training=True
if adv_training == True:
  model_dir = "./"
  model_name = "hardened.ckpt"
else:
  model_dir = "./"
  model_name = "pretrained.ckpt"

if not os.path.exists(model_dir):
  os.makedirs(model_dir)
summary_path = os.path.join(model_dir, "summary")
if not os.path.exists(summary_path):
  os.makedirs(summary_path)

# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs

var_list = tf.get_collection(
    tf.GraphKeys.GLOBAL_VARIABLES, scope="model")
assert (len(var_list)>10)
saver = tf.train.Saver(var_list,max_to_keep=1)


# keep the configuration file with the model for reproducibility
#shutil.copy('config.json', model_dir)

tf_config = tf.ConfigProto()
#tf_config.gpu_options.per_process_gpu_memory_fraction=0.5
tf_config.gpu_options.allow_growth = True



with tf.Session(config=tf_config) as sess:

  # initialize data augmentation
  cifar = cifar10_input.AugmentedCIFAR10Data(raw_cifar, sess, model, batch_size)

  # Initialize the summary writer, global variables, and our time counter.
  summary_writer = tf.summary.FileWriter(summary_path, sess.graph)
  sess.run(tf.global_variables_initializer())
  training_time = 0.0

  # Main training loop
  for ii in range(max_num_training_steps):
    x_batch, y_batch = cifar.train_data.get_next_batch()

    # Compute Adversarial Perturbations

    if adv_training:
      start = timer()
      x_batch_adv = attack.perturb(x_batch, y_batch, sess)
      end = timer()
      training_time += end - start

      adv_dict = {model.x_input: x_batch_adv,
                  model.y_input: y_batch}

    nat_dict = {model.x_input: x_batch,
                model.y_input: y_batch}
      # Output to stdout
    if ii % num_output_steps == 0:
      nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
      print('Step {}:    ({})'.format(ii, datetime.now()))
      print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
      if adv_training:
        adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
        print('    training adv accuracy {:.4}%'.format(adv_acc * 100))
      if ii != 0:
        print('    {} examples per second'.format(
            num_output_steps * batch_size / training_time))
        training_time = 0.0

    # Write a checkpoint
    if ii % num_checkpoint_steps == 0:
      saver.save(sess,
                 os.path.join(model_dir, model_name))
                 #global_step=global_step)

    # Actual training step
    start = timer()
    if adv_training:
      sess.run(train_step, feed_dict=adv_dict)
    else:
      sess.run(train_step, feed_dict=nat_dict)
    end = timer()
    training_time += end - start
