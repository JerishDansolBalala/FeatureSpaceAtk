# Train the Style Transfer Net
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os

import settings
data_set = "imagenet"#cifar10"  # "imagenet"
settings.init_settings("imagenet_shallow")

from adaptive_instance_norm import normalize
from PIL import Image
import cifar10_input
from cifar10_class import Model
from utils import get_train_images
from style_transfer_net import StyleTransferNet, StyleTransferNet_adv
from imagenetmod.interface import build_imagenet_model, imagenet, restore_parameter

STYLE_LAYERS = settings.config["STYLE_LAYERS"]

# (height, width, color_channels)
TRAINING_IMAGE_SHAPE = settings.config["IMAGE_SHAPE"]

EPOCHS = 4
EPSILON = 1e-5
BATCH_SIZE = settings.config["BATCH_SIZE"]
if data_set == "cifar10":
    LEARNING_RATE = 1e-1
    LR_DECAY_RATE = 1e-4  # 5e-5
    DECAY_STEPS = 1.0
    adv_weight = 5000
    ITER = 4000
else:
    LEARNING_RATE = 1e-3
    LR_DECAY_RATE = 1e-5  # 5e-5
    DECAY_STEPS = 1.0
    adv_weight = 12#128
    ITER = 1000
style_weight = 1

if data_set == "cifar10":
    raw_cifar = cifar10_input.CIFAR10Data("cifar10_data")

    def get_data(sess):
        x_batch, y_batch = raw_cifar.train_data.get_next_batch(
            batch_size=BATCH_SIZE, multiple_passes=True)
        return x_batch, y_batch
elif data_set == "imagenet":
    inet = imagenet(BATCH_SIZE, "train")

    def get_data(sess):
        x_batch, y_batch = inet.get_next_batch(sess)

        return x_batch, y_batch

ENCODER_WEIGHTS_PATH = 'vgg19_normalised.npz'
if data_set == "cifar10":
    # 'transform2.ckpt-147000'#./trans_pretrained/transform.ckpt'
    Decoder_Model = "./trans_pretrained/cifar10transform1.ckpt-574000"
elif data_set == "imagenet":
    # "./trans_pretrained/imagenetshallowtransform1.ckpt-104000"
    Decoder_Model = "./imagenetshallowtransform1.ckpt.mode2"


def save_rgb_img(img, path):
    img = img.astype(np.uint8)
    #img=np.reshape(img,[28,28])
    Image.fromarray(img, mode='RGB').save(path)


def get_scope_var(scope_name):
    var_list = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)
    assert (len(var_list) >= 1)
    return var_list


encoder_path = ENCODER_WEIGHTS_PATH
#model_save_path= "./transform.ckpt"
debug = True
logging_period = 100
if debug:
    from datetime import datetime
    start_time = datetime.now()


def internal_attack_exp(sess, grad, store, x_batch, y_batch, sigma_nat, mean_nat, sigma_sym, mean_sym, p):
    step_divide = 100
    iters = 1000

    log_sigma_nat = np.log(sigma_nat + 1e-7)

    sigma_lower_bound = log_sigma_nat - np.log(p)
    sigma_upper_bound = log_sigma_nat + np.log(p)
    sigma_step_size = np.log(p) / step_divide

    sign_mean_nat = np.sign(mean_nat)
    abs_mean_nat = np.abs(mean_nat)
    log_abs_mean_nat = np.log(abs_mean_nat + 1e-7)

    mean_lower_bound = log_abs_mean_nat - np.log(p)
    mean_upper_bound = log_abs_mean_nat + np.log(p)
    mean_step_size = np.log(p) / step_divide

    log_sigma = np.copy(log_sigma_nat)
    log_mean = np.copy(log_abs_mean_nat)
    for i in range(iters):
        _grad_sigma, _grad_mean = sess.run(grad, feed_dict={sigma_sym: np.exp(log_sigma),
                                                            mean_sym: np.exp(log_mean) * sign_mean_nat,
                                                            label: y_batch})
        log_sigma = log_sigma - np.sign(_grad_sigma) * sigma_step_size
        log_mean = log_mean - np.sign(_grad_mean) * mean_step_size
        log_sigma = np.clip(log_sigma, sigma_lower_bound, sigma_upper_bound)
        log_mean = np.clip(log_mean, mean_lower_bound, mean_upper_bound)
    return np.exp(log_mean) * sign_mean_nat, np.exp(log_sigma)


def internal_attack1(sess, grad, store, x_batch, y_batch, sigma_nat, mean_nat, sigma_sym, mean_sym, p):
    step_divide = 16
    iters = 1000

    sigma_nat_range = sigma_nat * (p-1)

    sigma_lower_bound = sigma_nat / p
    sigma_upper_bound = sigma_nat * p
    sigma_nat_range = (sigma_upper_bound - sigma_lower_bound)
    sigma_step_size = sigma_nat_range / step_divide

    abs_mean_nat = np.abs(mean_nat)

    mean_lower_bound = abs_mean_nat / p
    mean_upper_bound = abs_mean_nat * p
    mean_sign = np.sign(mean_nat)
    abs_mean_nat_range = (mean_upper_bound - mean_lower_bound)
    mean_step_size = abs_mean_nat_range / step_divide

    sigma = np.copy(sigma_nat)
    mean = np.copy(abs_mean_nat)
    for i in range(iters):
        (_grad_sigma, _grad_mean), _content_loss, _adv_acc = sess.run([grad, content_loss, adv_acc], feed_dict={sigma_sym: sigma,
                                                                                                                mean_sym: mean * mean_sign,
                                                                                                                label: y_batch})
        sigma = sigma - np.sign(_grad_sigma) * sigma_step_size
        mean = mean - np.sign(_grad_mean) * mean_step_size
        sigma = np.clip(sigma, sigma_lower_bound, sigma_upper_bound)
        mean = np.clip(mean, mean_lower_bound, mean_upper_bound)
        if (i % 10 == 0):
            print("content loss", _content_loss, " acc", _adv_acc)
    return mean_sign*mean, sigma

def internal_attack(sess, grad, store, x_batch, y_batch, sigma_nat, mean_nat, sigma_sym, mean_sym, p):
    step_divide = 4
    iters = 1000

    sigma_nat_range = sigma_nat * (p-1) 

    sigma_lower_bound = sigma_nat - sigma_nat_range
    sigma_upper_bound = sigma_nat + sigma_nat_range
    sigma_step_size = sigma_nat_range / step_divide


    abs_mean_nat_range = np.abs(mean_nat) * (p-1)

    mean_lower_bound = mean_nat - abs_mean_nat_range
    mean_upper_bound = mean_nat + abs_mean_nat_range
    mean_step_size = abs_mean_nat_range / step_divide

    sigma = np.copy(sigma_nat)
    mean = np.copy(mean_nat)
    for i in range(iters):
        (_grad_sigma, _grad_mean), _content_loss, _adv_acc = sess.run([grad, content_loss, adv_acc], feed_dict={sigma_sym: sigma,
                                                            mean_sym: mean,
                                                            label: y_batch})
        sigma = sigma - np.sign(_grad_sigma) * sigma_step_size
        mean = mean - np.sign(_grad_mean) * mean_step_size
        sigma = np.clip(sigma, sigma_lower_bound, sigma_upper_bound)
        mean = np.clip(mean, mean_lower_bound, mean_upper_bound)
        if (i%10==0):
            print("content loss", _content_loss," acc",_adv_acc)
    return mean, sigma


# get the traing image shape
HEIGHT, WIDTH, CHANNELS = TRAINING_IMAGE_SHAPE
INPUT_SHAPE = (None, HEIGHT, WIDTH, CHANNELS)
if data_set == "cifar10":
    INTERNAL_SHAPE = (BATCH_SIZE, 16, 16, 128)
    FEATURE_SPACE = (None, 1, 1, 128)
else:
    INTERNAL_SHAPE = (BATCH_SIZE, 56, 56, 256)#(BATCH_SIZE, 28, 28, 512)
    FEATURE_SPACE = (BATCH_SIZE, 1, 1, 256)
#
# create the graph
tf_config = tf.ConfigProto()
#tf_config.gpu_options.per_process_gpu_memory_fraction=0.5
tf_config.gpu_options.allow_growth = True
with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:

    content = tf.placeholder(tf.float32, shape=INPUT_SHAPE, name='content')
    style = tf.placeholder(tf.float32, shape=INPUT_SHAPE, name='style')
    label = tf.placeholder(tf.int64, shape=None, name="label")

    #style   = tf.placeholder(tf.float32, shape=INPUT_SHAPE, name='style')

    stored_internal = tf.Variable(tf.zeros(INTERNAL_SHAPE), dtype=tf.float32)
    internal_sigma = tf.placeholder(tf.float32, shape=FEATURE_SPACE)
    internal_mean = tf.placeholder(tf.float32, shape=FEATURE_SPACE)

    # create the style transfer net
    stn = StyleTransferNet_adv(encoder_path)

    # pass content and style to the stn, getting the generated_img
    generated_img, generated_img_adv = stn.transform_from_internal(content, stored_internal,
                                                                   internal_sigma, internal_mean)
    adv_img = generated_img_adv
    img = generated_img

    stn_vars = []  # get_scope_var("transform")
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

    # compute the content loss
    content_loss = tf.reduce_sum(tf.reduce_mean(
        tf.square(enc_gen_adv - target_features), axis=[1, 2]))
    #content_loss += tf.reduce_sum(tf.reduce_mean(
    #    tf.square(enc_gen - stn.norm_features), axis=[1, 2]))
    if data_set == "cifar10":
        classifier = Model("eval", raw_cifar.train_images)
        classifier._build_model(adv_img, label, reuse=False)
        adv_loss = - classifier.target_loss
        adv_acc = classifier.accuracy
        classifier._build_model(img, label, reuse=True)
        normal_loss = - classifier.relaxed_y_xent
        norm_acc = classifier.accuracy
    elif data_set == "imagenet":
        classifier = build_imagenet_model(adv_img_bgr, label)
        adv_loss = - classifier.target_loss
        adv_acc = classifier.accuracy
        classifier = build_imagenet_model(img_bgr, label, reuse=True)
        normal_loss = - classifier.xent
        norm_acc = classifier.accuracy
    # compute the style loss

    style_layer_loss = []

    # compute the total loss
    # adv_loss * adv_weight
    loss = content_loss + adv_loss * adv_weight  # style_weight * style_loss
    adv_grad = tf.gradients(loss, [internal_sigma, internal_mean])

    loss = loss
    if data_set=="cifar10":
        classifier_vars = get_scope_var("model")
    decoder_vars = get_scope_var("decoder")
    # Training step
    global_step = tf.Variable(0, trainable=False)
    # tf.train.inverse_time_decay(LEARNING_RATE, global_step, DECAY_STEPS, LR_DECAY_RATE)
    learning_rate = 1e-4
    #train_op = tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(
    #    loss, var_list=stn_vars, global_step=global_step)  # stn_vars+

    sess.run(tf.global_variables_initializer())
    if data_set == "cifar10":
        classifier_saver = tf.train.Saver(classifier_vars, max_to_keep=1)
        classifier_saver.restore(sess, settings.config["hardened_model"])
    elif data_set == "imagenet":
        restore_parameter(sess)

    # saver
    saver = tf.train.Saver(decoder_vars, max_to_keep=1)
    saver.restore(sess, Decoder_Model)
    ###### Start Training ######
    step = 0

    if debug:
        elapsed_time = datetime.now() - start_time
        start_time = datetime.now()
        print('\nElapsed time for preprocessing before actually train the model: %s' % elapsed_time)
        print('Now begin to train the model...\n')

    """for batch in range(2):
        # retrive a batch of content and style images
        #content_batch_path = content_imgs_path[batch*BATCH_SIZE:(batch*BATCH_SIZE + BATCH_SIZE)]
        #style_batch_path   = style_imgs_path[batch*BATCH_SIZE:(batch*BATCH_SIZE + BATCH_SIZE)]

        #content_batch = get_train_images(content_batch_path, crop_height=HEIGHT, crop_width=WIDTH)
        #style_batch   = get_train_images(style_batch_path,   crop_height=HEIGHT, crop_width=WIDTH)

        # run the training step
        
        x_batch, y_batch = raw_cifar.train_data.get_next_batch(batch_size=64, multiple_passes=True)

        fdict = {content: x_batch, label: y_batch}
        sess.run(stn.init_style,feed_dict=fdict)
        for i in range(10000):
            sess.run([train_op, stn.style_bound], feed_dict=fdict)
            #sess.run(stn.style_bound, feed_dict = fdict)

        step += 1

        for i in range(60):
            gan_out = sess.run(adv_img, feed_dict=fdict)
            save_out = np.concatenate(
                (gan_out[i], x_batch[i], np.abs(gan_out[i]-x_batch[i])))
            full_path = os.path.join("advimg", "%d"%step,  "%d.jpg"% i) 
            os.makedirs(os.path.join("advimg", "%d" % step),exist_ok=True)
            save_out = np.reshape(save_out, newshape=[32*3, 32, 3])
            save_rgb_img(save_out, path=full_path)"""

    for batch in range(2):
        # retrive a batch of content and style images
        #content_batch_path = content_imgs_path[batch*BATCH_SIZE:(batch*BATCH_SIZE + BATCH_SIZE)]
        #style_batch_path   = style_imgs_path[batch*BATCH_SIZE:(batch*BATCH_SIZE + BATCH_SIZE)]

        #content_batch = get_train_images(content_batch_path, crop_height=HEIGHT, crop_width=WIDTH)
        #style_batch   = get_train_images(style_batch_path,   crop_height=HEIGHT, crop_width=WIDTH)

        # run the training step

        x_batch, y_batch = get_data(sess)

        fdict = {content: x_batch, label: y_batch}
        _, mean, sigma = sess.run(
            [stn.store_normalize, stn.meanC, stn.sigmaC], feed_dict=fdict)

        adv_mean, adv_sigma = internal_attack(sess, adv_grad, stored_internal,
                                              x_batch, y_batch, mean_nat=mean, sigma_nat=sigma, sigma_sym=internal_sigma, mean_sym=internal_mean, p=1.5)

        fdict_adv = { content: x_batch,
                     internal_mean: adv_mean,
                     internal_sigma: adv_sigma,
                     label: y_batch}
        fdict_normal = {content: x_batch,
                        internal_mean: mean,
                        internal_sigma: sigma,
                        label: y_batch}
        step += 1

        for i in range(8):
            gan_out = sess.run(adv_img, feed_dict=fdict_adv)
            save_out = np.concatenate(
                (gan_out[i], x_batch[i], np.abs(gan_out[i]-x_batch[i])))
            sz = TRAINING_IMAGE_SHAPE[1]
            full_path = os.path.join("advimg", "%d" % step,  "%d.jpg" % i)
            os.makedirs(os.path.join("advimg", "%d" % step), exist_ok=True)
            save_out = np.reshape(save_out, newshape=[sz*3, sz, 3])
            save_rgb_img(save_out, path=full_path)

        if batch % 1 == 0:

            elapsed_time = datetime.now() - start_time
            _adv_img, _content_loss, _adv_acc, _adv_loss, _loss, _internal_diff = sess.run([adv_img, content_loss, adv_acc, adv_loss, loss , stn.loss_l1],
                                                                            feed_dict=fdict_adv)
            _normal_loss, _normal_acc = sess.run([normal_loss, norm_acc],
                                                 feed_dict=fdict)
            l2_loss = (_adv_img - x_batch) / 255
            l2_loss = np.sum(l2_loss*l2_loss)/64
            li_loss = np.mean(
                np.amax(np.abs(_adv_img - x_batch) / 255, axis=-1))
            l1_loss = np.mean(
                np.sum(np.abs(_adv_img - x_batch) / 255, axis=-1))
            print("l2_loss", l2_loss, "li_loss", li_loss, "l1_loss", l1_loss)
            print('step: %d,  total loss: %.3f,  elapsed time: %s' %
                  (step, _loss, elapsed_time))
            print('content loss: %.3f' % (_content_loss))
            print('adv loss  : %.3f,  weighted adv loss: %.3f , adv acc %.3f, internal diff %.3f' %
                  (_adv_loss, adv_weight * _adv_loss, _adv_acc, _internal_diff))
            print('normal loss : %.3f normal acc: %.3f\n' %
                  (_normal_loss, _normal_acc))

    ###### Done Training & Save the model ######
    #saver.save(sess, model_save_path)

    if debug:
        elapsed_time = datetime.now() - start_time
        print('Done training! Elapsed time: %s' % elapsed_time)
