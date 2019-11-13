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
model = Model(mode='eval',data=raw_cifar.train_images)
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
saver = tf.train.Saver(var_list,max_to_keep=1)


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


rst = {}
num_output_steps=500
with tf.Session(config=tf_config) as sess:
    for aname in ["pgd", "selfaug", "polygon"]:
        saver.restore(sess,
                    os.path.join(base_dir_model, aname + "_pertrained.ckpt"))
        rst[aname]={} 
        for bname in ["pgd","selfaug","polygon","nat"]:
            if bname =="nat":
                x = eval_dict["x"]
            else:
                x = eval_dict[bname + "_adv"]
            
            y = eval_dict["y"]

            training_time = 0.0
            gen = iterator(x,y,shuffle=False)
            # Main training loop
            batches = 100
            tot_acc = 0
            for ii in range(batches):
                
                # Compute Adversarial Perturbations
                x_batch,y_batch = next(gen)
                #print(x_batch.shape,y_batch.shape)
                nat_dict = {model.x_input: x_batch,
                            model.y_input: y_batch}
                # Output to stdout
                if ii % 30 == 0:
                    nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
                    print('Step {}:    ({})'.format(ii, datetime.now()))
                    print('    training nat accuracy {:.4}%'.format(nat_acc * 100))

                tot_acc += nat_acc
            tot_acc /= batches
            rst[aname][bname] = {tot_acc}
            print(aname,bname,tot_acc )

    print("pgd", "selfaug", "polygon")
    for aname in ["pgd", "selfaug", "polygon"]:
        print("hardened by %s:"%aname)
        for bname in ["pgd", "selfaug", "polygon", "nat"]:
            print(rst[aname][bname])
