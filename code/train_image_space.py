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
import settings

task_name = "attack"
data_set = "cifar10"  # "imagenet"
model_name = "cifar10_adv"
decoder_name = "cifar10_balance"

exec(open('base.py').read())

base_dir_model = os.path.join(base_dir_model, "imagegen")

with open('config.json') as config_file:
    config = json.load(config_file)

# seeding randomness
tf.set_random_seed(config['tf_random_seed'])
np.random.seed(config['np_random_seed'])

# Setting up training parameters
max_num_training_steps = 30000# config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
# [[0, 0.1], [500, 0.01], [1000, 0.001]]#config['step_size_schedule']
step_size_schedule = [[0, 0.1], [10000, 0.01], [20000, 0.001]]
weight_decay = config['weight_decay']
data_path = config['data_path']
momentum = config['momentum']
batch_size = config['training_batch_size']

# Setting up the data and the model
raw_cifar = cifar10_input.CIFAR10Data(data_path)
global_step = tf.contrib.framework.get_or_create_global_step()
model = Model(mode='train',data=raw_cifar.train_images)
model._build_model_easy()

\

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



# Setting up the Tensorboard and checkpoint outputs


if not os.path.exists(base_dir_model):
  os.makedirs(base_dir_model)


# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs

var_list = tf.get_collection(
    tf.GraphKeys.GLOBAL_VARIABLES, scope="model")
assert (len(var_list)>10)



# keep the configuration file with the model for reproducibility
#shutil.copy('config.json', model_dir)

tf_config = tf.ConfigProto()
#tf_config.gpu_options.per_process_gpu_memory_fraction=0.5
tf_config.gpu_options.allow_growth = True


def merge_dict(dict_tot, dict1):
    for k, v in dict1.items():
        if k in dict_tot:
            dict_tot[k] = np.concatenate([dict_tot[k], dict1[k]])
        else:
            dict_tot[k] = dict1[k]
    return dict_tot

def get_np_dict(prefix):
    np_dict = {}
    for i in range(1, 100):
        np_file_path = os.path.join(
            base_dir_model, "%s_saved_samples%d.npy" % (prefix, i))
        if os.path.exists(np_file_path):
            _np_dict = np.load(np_file_path).item()
            merge_dict(np_dict, _np_dict)
    return np_dict

def iterator(x,y,batch_size=64, shuffle=False):
    x=np.copy(x)
    y=np.copy(y)
    assert x.shape[0]==y.shape[0]
    sp = x.shape[0]
    batch = sp // batch_size
    while True:
        if shuffle:
            perm = np.random.permutation(sp)
            x = x[perm,:,:,:]
            y = y[perm,]
        for i in range(batch):
            
            yield x[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size]
    

train_dict = get_np_dict("train")
eval_dict = get_np_dict("eval")
"""x = train_dict["pgd_adv"]
y = train_dict["y"]
gen = iterator(x, y, shuffle=False)
for i in range(3):
    x_batch, y_batch = next(gen)
    print(i, y_batch)
exit()"""
with tf.Session(config=tf_config) as sess:
    # ["pgd", "selfaug"]:  # , "polygon"]:
    for aname in ["pgd", "polygon", "selfaug"]:
        x = train_dict[aname + "_adv"]
        y = train_dict["y"]
        x_1 = train_dict["x"]
        saver = tf.train.Saver(var_list, max_to_keep=1)
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, settings.config["hardened_model"])
        training_time = 0.0
        gen = iterator(x,y,shuffle=True)
        gen1 = iterator(x_1,y,shuffle=True)
        # Main training loop
        for ii in range(max_num_training_steps):
            
            # Compute Adversarial Perturbations
            x_batch,y_batch = next(gen)
            #x_batch1,y_batch1 = next(gen1)
            #print(x_batch.shape,y_batch.shape)
            #x_batch=np.concatenate([x_batch,x_batch1])
            #y_batch=np.concatenate([y_batch,y_batch1])
            nat_dict = {model.x_input: x_batch,
                        model.y_input: y_batch}
            
            nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
            #if nat_acc > 0.9:
            #    break
            # Output to stdout
            if ii % 100== 0:                
                print('Step {}:    ({})'.format(ii, datetime.now()))
                print('    training nat accuracy {:.4}%'.format(nat_acc * 100))

            # Write a checkpoint
            if ii % num_checkpoint_steps == 0:
                saver.save(sess,
                        os.path.join(base_dir_model, aname + "_pertrained.ckpt"))
                            #global_step=global_step)

            # Actual training step
            start = timer()
            sess.run(train_step, feed_dict=nat_dict)
            end = timer()
            training_time += end - start

        saver.save(sess,
                    os.path.join(base_dir_model, aname + "_pertrained.ckpt"))
