# Train the Style Transfer Net

from __future__ import print_function

import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import tensorflow as tf
import os
import settings
import deepknn_detect
from pgd_attack import LinfPGDAttack

task_name = "deepknn"
data_set = "cifar10"  # "imagenet"
model_name = "cifar10_adv"
decoder_name = "cifar10"

"""
data_set = "cifar10"  # "imagenet"
model_name = "cifar10_adv"
decoder_name = "cifar10"
"""
exec(open('base.py').read())

from style_transfer_net import StyleTransferNet, StyleTransferNet_adv
from utils import get_train_images
from cifar10_class import Model
import cifar10_input
from PIL import Image
from adaptive_instance_norm import normalize


STYLE_LAYERS = settings.config["STYLE_LAYERS"]

# (height, width, color_channels)
TRAINING_IMAGE_SHAPE = settings.config["IMAGE_SHAPE"]

EPOCHS = 4
EPSILON = 1e-5
BATCH_SIZE = settings.config["BATCH_SIZE"]
if data_set=="cifar10":
    LEARNING_RATE = 1e-2
    LR_DECAY_RATE = 1e-4 #5e-5
    DECAY_STEPS = 1.0
    adv_weight = 5000
    ITER=500
    CLIP_NORM_VALUE = 10.0
else:
    if model_name == "imagenet_shallowest":
        LEARNING_RATE = 5e-3
    else:
        LEARNING_RATE = 1e-2
    LR_DECAY_RATE = 1e-3 # 5e-5
    DECAY_STEPS = 1.0
    adv_weight = 128 
    ITER=500
    CLIP_NORM_VALUE = 10.0

style_weight = 1

if data_set == "cifar10":
    raw_cifar = cifar10_input.CIFAR10Data("cifar10_data")

    def get_data(sess):
        x_batch, y_batch = raw_cifar.eval_data.get_next_batch(
            batch_size=BATCH_SIZE, multiple_passes=True)
        return x_batch, y_batch
elif data_set == "imagenet":
    inet = imagenet(BATCH_SIZE, "val")

    def get_data(sess):
        x_batch, y_batch = inet.get_next_batch(sess)

        return x_batch, y_batch



def save_rgb_img( img, path):
    img = img.astype(np.uint8)
    #img=np.reshape(img,[28,28])
    Image.fromarray(img, mode='RGB').save(path)


def get_scope_var(scope_name):
    var_list = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)
    assert (len(var_list) >= 1)
    return var_list

encoder_path=ENCODER_WEIGHTS_PATH
#model_save_path= "./transform.ckpt"
debug=True
logging_period=100
if debug:
    from datetime import datetime
    start_time = datetime.now()

def grad_attack():
    sess.run(stn.init_style, feed_dict=fdict)
    sess.run(global_step.initializer)
    rst_img, rst_loss, rst_acc,rst_mean,rst_sigma = sess.run(
        [adv_img, content_loss_y, adv_acc, stn.meanS, stn.sigmaS],  feed_dict=fdict)
    
    for i in range(ITER):
        #_,  acc, aloss, closs, closs1, sigma, mean, sigmaS, meanS = sess.run(
        #    [train_op,  adv_acc, adv_loss, content_loss_y, content_loss, stn.sigmaC, stn.meanC, stn.sigmaS, stn.meanS], feed_dict=fdict)
        _ = sess.run([train_op],  feed_dict=fdict)
        sess.run(stn.style_bound, feed_dict = fdict)
        _adv_img, acc, aloss, closs, _mean, _sigma = sess.run(
            [adv_img, adv_acc, adv_loss, content_loss_y, stn.meanS, stn.sigmaS],  feed_dict=fdict)
        for j in range(BATCH_SIZE):
            if acc[j]<rst_acc[j] or (acc[j]==rst_acc[j] and closs[j]<rst_loss[j]):
                rst_img[j]=_adv_img[j]
                rst_acc[j] = acc[j]
                rst_loss[j] = closs[j]
                rst_mean[j] = _mean[j]
                rst_sigma[j] = _sigma[j]

        if i%50==0 :
            acc=np.mean(acc)
            
            print(i,acc,"advl",aloss,"contentl",closs)
        #if np.sum(acc) == 0 and np.all(np.less_equal(closs,2*128)):
            #break
            #if i==1:
            #    exit()
    #sess.run(stn.asgn, feed_dict={stn.meanS_ph: rst_mean, stn.sigmaS_ph: rst_sigma})
    return rst_img


def rand_attack():
    for i in range(10):
        sess.run(stn.init_style, feed_dict=fdict)
        sess.run(global_step.initializer)
        for j in range(10):
            _,  acc, aloss, closs = sess.run(
                [train_op,  adv_acc, adv_loss, content_loss], feed_dict=fdict)
            save_rgb_img(save_out, path=full_path)
            sess.run(stn.style_bound, feed_dict = fdict)
            print(i,acc,"advl",aloss,"contentl",closs)
        if acc < 0.05 and closs < 2000:
            break


# get the traing image shape
HEIGHT, WIDTH, CHANNELS = TRAINING_IMAGE_SHAPE
INPUT_SHAPE = (None, HEIGHT, WIDTH, CHANNELS)

def gradient(opt, vars, loss ):
    gradients, variables = zip(*opt.compute_gradients(loss,vars))
    g_split = [tf.unstack(g, BATCH_SIZE, axis=0) for g in gradients]
    g1_list=[]
    g2_list=[]
    DIM = settings.config["DECODER_DIM"][-1]
    limit = 10/np.sqrt(DIM)    
    for g1,g2 in zip(g_split[0],g_split[1]):
        #(g1, g2), _ = tf.clip_by_global_norm([g1, g2], CLIP_NORM_VALUE)
        g1 = tf.clip_by_value(g1,-1/np.sqrt(limit),1/np.sqrt(limit))
        g2 = tf.clip_by_value(g2,-1/np.sqrt(limit),1/np.sqrt(limit))
        g1_list.append(g1)
        g2_list.append(g2)
    gradients = [tf.stack(g1_list, axis=0), tf.stack(g2_list, axis=0)]
    #gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    opt = opt.apply_gradients(zip(gradients, variables), global_step=global_step)
    return opt


def gradient1(opt, vars, loss):
    gradients, variables = zip(*opt.compute_gradients(loss, vars))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    opt = opt.apply_gradients(
        zip(gradients, variables), global_step=global_step)
    return opt

# create the graph
tf_config = tf.ConfigProto()
#tf_config.gpu_options.per_process_gpu_memory_fraction=0.5
tf_config.gpu_options.allow_growth = True
with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:

    content = tf.placeholder(tf.float32, shape=INPUT_SHAPE, name='content')
    #style = tf.placeholder(tf.float32, shape=INPUT_SHAPE, name='style')
    label = tf.placeholder(tf.int64, shape =None, name="label")
    #style   = tf.placeholder(tf.float32, shape=INPUT_SHAPE, name='style')



    # create the style transfer net
    stn = StyleTransferNet_adv(encoder_path)

    # pass content and style to the stn, getting the generated_img
    generated_img , generated_img_adv = stn.transform(content)
    adv_img=generated_img_adv
    img = generated_img

    stn_vars = get_scope_var("transform")
    # get the target feature maps which is the output of AdaIN
    target_features = stn.target_features

    # pass the generated_img to the encoder, and use the output compute loss
    generated_img_adv = tf.reverse(
        generated_img_adv, axis=[-1])  # switch RGB to BGR
    adv_img_bgr = generated_img_adv
    generated_img_adv = stn.encoder.preprocess(
        generated_img_adv)  # preprocess image
    enc_gen_adv, enc_gen_layers_adv = stn.encoder.encode(generated_img_adv)
    
    generated_img = tf.reverse(
        generated_img, axis=[-1])  # switch RGB to BGR
    img_bgr = generated_img
    generated_img = stn.encoder.preprocess(
        generated_img)  # preprocess image
    enc_gen, enc_gen_layers = stn.encoder.encode(generated_img)

    deepknn = deepknn_detect.dknn_detection((raw_cifar.train_images, raw_cifar.train_labels, raw_cifar.eval_images, raw_cifar.eval_labels),
                                            sess, adv_img, content, label)
    deepknn.build_model()

    if data_set == "cifar10":
        logits, adv_loss = deepknn.get_logits(adv_img,label)
        logits_normal, adv_loss_normal = deepknn.get_logits(content, label)
        adv_acc = tf.cast(tf.equal(tf.argmax(logits,-1),label),tf.float32)
        acc = tf.cast(tf.equal(tf.argmax(logits_normal, -1), label), tf.float32)
        adv_loss = - adv_loss
        adv_loss_normal = -adv_loss_normal
    # compute the content loss
    bar=3000/64/128
    content_loss_y = tf.reduce_sum(
        tf.reduce_mean(tf.square(enc_gen_adv - target_features), axis=[1, 2]),axis=-1)
    content_loss = tf.reduce_sum(
        tf.reduce_mean(tf.square(enc_gen_adv - target_features), axis=[1, 2]))
    #
    #content_loss += tf.reduce_sum(tf.reduce_mean(
    #    tf.square(enc_gen - stn.norm_features), axis=[1, 2]))

    # compute the style loss
    
    style_layer_loss = []

    # compute the total loss
    # adv_loss * adv_weight
    loss = content_loss + tf.reduce_sum(adv_loss * BATCH_SIZE * adv_weight)# style_weight * style_loss

    l2_embed = normalize(enc_gen)[0] - normalize(stn.norm_features)[0]
    l2_embed = tf.reduce_mean(
        tf.sqrt(tf.reduce_sum((l2_embed * l2_embed), axis=[1, 2, 3])))


    loss=loss
    decoder_vars = get_scope_var("decoder")
    # Training step
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.inverse_time_decay(LEARNING_RATE, global_step, DECAY_STEPS, LR_DECAY_RATE)
    #tf.train.AdamOptimizer(learning_rate).minimize(  # MomentumOptimizer(learning_rate, momentum=0.9) tf.train.GradientDescentOptimizer(learning_rate)
    #train_op = tf.train.AdamOptimizer(learning_rate).minimize(
    #    loss, var_list=stn_vars, global_step=global_step)

    ##gradient clipping
    train_op = gradient(tf.train.AdamOptimizer(learning_rate, beta1= 0.5),vars=stn_vars, loss=loss)

    #train_op = tf.train.AdamOptimizer(learning_rate).minimize(
    #    loss, var_list=stn_vars, global_step=global_step)  
    pgd_attack = LinfPGDAttack(- adv_loss_normal, content, label,
        epsilon=8.0, num_steps=50, step_size=2.0, random_start=True)
    atk_func = pgd_attack.perturb
    sess.run(tf.global_variables_initializer())
    deepknn.pretrain()

    # saver
    saver = tf.train.Saver(decoder_vars, max_to_keep=1)
    saver.restore(sess,Decoder_Model)
    ###### Start Training ######
    step = 0

    if debug:
        elapsed_time = datetime.now() - start_time
        start_time = datetime.now()
        print('\nElapsed time for preprocessing before actually train the model: %s' % elapsed_time)
        print('Now begin to train the model...\n')



    uid = 0

    total_batch=100

    tot_m_adv=0
    tot_m_benign=0
    tot_acc_adv =0
    tot_acc_normal=0
    tot_m_pgd=0
    for batch in range(1,total_batch+1):

        x_batch, y_batch = deepknn.test_data[batch* \
                                             64:(batch+1)*64], deepknn.y_test[batch*64:(batch+1)*64]
        y_batch_label = np.argmax(y_batch,-1)
        fdict = {content: x_batch, label: y_batch_label}
        perturbed_x = atk_func(x_batch, y_batch_label, sess)
        grad_attack()

        gan_out,_acc_normal,_acc_adv = sess.run([adv_img,acc,adv_acc], feed_dict=fdict)
        tot_acc_adv += np.mean(_acc_adv.astype(np.float32)) 
        tot_acc_normal += np.mean(_acc_normal.astype(np.float32))
        tot_m_adv += deepknn.test_acc(gan_out,y_batch)    
        tot_m_benign += deepknn.test_acc(x_batch,y_batch)    
        tot_m_pgd += deepknn.test_acc(perturbed_x, y_batch)

        print(batch, tot_m_adv, tot_m_benign, tot_m_pgd, tot_acc_adv, tot_acc_normal)
    print("==="*10)
    tot_m_adv/=total_batch
    tot_m_benign/=total_batch
    tot_m_pgd /= total_batch
    tot_acc_adv/=total_batch
    tot_acc_normal /= total_batch
    print("final", tot_m_adv, tot_m_benign, tot_m_pgd, tot_acc_adv, tot_acc_normal)


    if debug:
        elapsed_time = datetime.now() - start_time
        print('Done training! Elapsed time: %s' % elapsed_time)
        #print('Model is saved to: %s' % model_save_path)

