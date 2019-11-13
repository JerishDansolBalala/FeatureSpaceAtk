# Train the Style Transfer Net

from __future__ import print_function

import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import tensorflow as tf
import os
import settings
"""
data_set = "imagenet"  # "imagenet"
model_name = "imagenet_denoise"
decoder_name = "imagenet_shallow"
"""
task_name = "attack"
data_set = "cifar10"  # "imagenet"
model_name = "cifar10_trades"
decoder_name = "cifar10_balance"

exec(open('base.py').read())

from style_transfer_net import StyleTransferNet, StyleTransferNet_adv
from utils import get_train_images
from cifar10_class import Model
from trade_interface import build_model as build_model_trades
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
    LEARNING_RATE = 1e-3
    LR_DECAY_RATE = 1e-4 #5e-5
    DECAY_STEPS = 1.0
    adv_weight = 256
    ITER=2000
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
        [adv_img, l2_embed, adv_acc_y, stn.meanS, stn.sigmaS],  feed_dict=fdict)
    
    for i in range(ITER):
        #_,  acc, aloss, closs, closs1, sigma, mean, sigmaS, meanS = sess.run(
        #    [train_op,  adv_acc, adv_loss, content_loss_y, content_loss, stn.sigmaC, stn.meanC, stn.sigmaS, stn.meanS], feed_dict=fdict)
        _ = sess.run([train_op],  feed_dict=fdict)
        sess.run(stn.style_bound, feed_dict = fdict)
        _adv_img, acc, aloss, closs, _mean, _sigma = sess.run( [adv_img, adv_acc_y, adv_loss, l2_embed, stn.meanS, stn.sigmaS],  feed_dict = fdict)
        for j in range(BATCH_SIZE):
            if acc[j]<rst_acc[j] or (acc[j]==rst_acc[j] and closs[j]<rst_loss[j]):
                rst_img[j]=_adv_img[j]
                rst_acc[j] = acc[j]
                rst_loss[j] = closs[j]
                rst_mean[j] = _mean[j]
                rst_sigma[j] = _sigma[j]

        if i%50==0 :
            """for j in range(BATCH_SIZE):
                gan_out = sess.run(adv_img, feed_dict=fdict)
                save_out = np.concatenate(
                    (gan_out[j], x_batch[j], np.abs(gan_out[j]-x_batch[j])))
                sz = TRAINING_IMAGE_SHAPE[1]
                full_path = os.path.join("temp", "%d" % i,  "%d.jpg" % j)
                os.makedirs(os.path.join("temp", "%d" % i), exist_ok=True)
                save_out = np.reshape(save_out, newshape=[sz*3, sz, 3])
                save_rgb_img(save_out, path=full_path)"""
            #print("sigma", sigma[0])
            #print("mean", mean[0])
            #print("sigma", sigma)
            #print("mean", mean)
            #print("sigmaS", sigma)
            #print("meanS", mean)
            acc=np.mean(acc)
            closs=np.mean(closs)
            print(i,acc,"advl",aloss,"contentl",closs, np.mean(rst_loss))
        #if np.sum(acc) == 0 and np.all(np.less_equal(closs,2*128)):
            #break
            #if i==1:
            #    exit()
    sess.run(stn.asgn, feed_dict={stn.meanS_ph: rst_mean, stn.sigmaS_ph: rst_sigma})
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
    style = tf.placeholder(tf.float32, shape=INPUT_SHAPE, name='style')
    label = tf.placeholder(tf.int64, shape =None, name="label")
    #style   = tf.placeholder(tf.float32, shape=INPUT_SHAPE, name='style')



    # create the style transfer net
    stn = StyleTransferNet_adv(encoder_path)

    # pass content and style to the stn, getting the generated_img
    generated_img , generated_img_adv = stn.transform(content,p=1.5)
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

    if data_set == "cifar10":
        if model_name in ["cifar10_nat","cifar10_adv"]:
            classifier = Model("eval", raw_cifar.train_images)
            classifier._build_model(adv_img, label, reuse=False, conf=0.1)
            adv_loss = - classifier.target_loss
            adv_acc = classifier.accuracy
            adv_acc_y = tf.cast(classifier.correct_prediction, tf.float32)
            classifier._build_model(content, label, reuse=True, conf=0.1)
            normal_loss = - classifier.target_loss
            norm_acc = classifier.accuracy
            acc_y = tf.cast(classifier.correct_prediction, tf.float32)
            classifier._build_model(img, label, reuse=True)
            decode_acc_y = tf.cast(classifier.correct_prediction, tf.float32)
        elif model_name in ["cifar10_trades"]:
            classifier = build_model_trades(adv_img, label, conf=0.1)
            #classifier._build_model(adv_img, label, reuse=False, conf=0.1)
            adv_loss = - classifier.target_loss
            adv_acc = classifier.accuracy
            adv_acc_y = tf.cast(classifier.correct_prediction, tf.float32)
            #classifier._build_model(content, label, reuse=True, conf=0.1)
            classifier = build_model_trades(content, label, conf=0.1)
            normal_loss = - classifier.target_loss
            norm_acc = classifier.accuracy
            acc_y = tf.cast(classifier.correct_prediction, tf.float32)
            #classifier._build_model(img, label, reuse=True)
            classifier = build_model_trades(img, label, conf=0.1)
            decode_acc_y = tf.cast(classifier.correct_prediction, tf.float32)
        else:
            assert False

    elif data_set == "imagenet":
        classifier = build_imagenet_model(adv_img_bgr, label, conf=1)
        adv_loss = - classifier.target_loss5
        adv_acc = classifier.accuracy
        adv_acc_y = classifier.acc_y
        adv_acc_y_5 = classifier.acc_y_5
        content_bgr = tf.reverse(
            content, axis=[-1])  # switch RGB to BGR
        classifier = build_imagenet_model(content_bgr, label, reuse=True)
        normal_loss = - classifier.target_loss5
        norm_acc = classifier.accuracy
        acc_y = classifier.acc_y
        acc_y_5 = classifier.acc_y_5
        classifier = build_imagenet_model(img_bgr, label, reuse=True)
        decode_acc_y = classifier.acc_y
        decode_acc_y_5 = classifier.acc_y_5

    l2_embed_d = normalize(enc_gen_adv)[0] - normalize(stn.norm_features)[0]
    l2_embed = tf.sqrt(tf.reduce_sum((l2_embed_d * l2_embed_d), axis=[1, 2, 3]))

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
    #content_loss
    loss = content_loss * 50 + tf.reduce_sum(adv_loss *
                                                   BATCH_SIZE * adv_weight)  # style_weight * style_loss

    loss=loss
    if data_set == "cifar10":
        if model_name in ["cifar10_nat","cifar10_adv"]:
            classifier_vars = get_scope_var("model")
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

    sess.run(tf.global_variables_initializer())
    if data_set == "cifar10":
        if model_name=="cifar10_adv":
            classifier_saver = tf.train.Saver(classifier_vars, max_to_keep=1)
            classifier_saver.restore(sess, settings.config["hardened_model"])
        elif model_name=="cifar10_nat":
            classifier_saver = tf.train.Saver(classifier_vars, max_to_keep=1)
            classifier_saver.restore(sess, settings.config["pretrained_model"])
        elif model_name=="cifar10_trades":
            pass

    elif data_set == "imagenet":
        restore_parameter(sess)
    

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

    x_batch, y_batch = get_data(sess)
    fdict = {content: x_batch, label: y_batch}

    uid = 0

    report_batch = 10
    for batch in range(1,150+1):

        if batch % report_batch == 1:
            np_adv_image = []
            np_benign_image = []
            np_content_loss = []
            np_acc_attack = []
            np_acc = []
            np_decode_acc = []
            np_label = []
        # run the training step
        
        x_batch, y_batch = get_data(sess)
        fdict = {content: x_batch, label: y_batch}
        grad_attack()

        step += 1

        for i in range(BATCH_SIZE):
            gan_out = sess.run(adv_img, feed_dict=fdict)
            save_out = np.concatenate(
                (gan_out[i], x_batch[i], np.abs(gan_out[i]-x_batch[i])))
            sz = TRAINING_IMAGE_SHAPE[1]
            full_path = os.path.join(
                base_dir_model, "%d" % step,  "%d.jpg" % i)
            os.makedirs(os.path.join(base_dir_model, "%d" %
                                     step), exist_ok=True)
            save_out = np.reshape(save_out, newshape=[sz*3, sz, 3])
            save_rgb_img(save_out, path=full_path)

        if batch % 1 == 0:

            elapsed_time = datetime.now() - start_time
            _content_loss, _adv_acc, _adv_loss, _loss,   \
                = sess.run([ content_loss, adv_acc, adv_loss, loss,], feed_dict=fdict)
            _adv_img, _loss_y, _adv_acc_y,  _acc_y, _decode_acc_y,  = sess.run([
                adv_img, content_loss_y, adv_acc_y,  acc_y,  decode_acc_y], feed_dict=fdict)
            #_normal_loss, _normal_acc = sess.run([normal_loss, norm_acc], feed_dict=fdict)
            np_adv_image.append(_adv_img)
            np_benign_image.append(x_batch)
            np_content_loss.append(_loss_y)
            np_acc_attack.append(_adv_acc_y)
            np_acc .append(_acc_y)
            np_label.append(y_batch)
            np_decode_acc.append(_decode_acc_y)

            _adv_loss = np.sum(_adv_loss)
            #_normal_loss = np.sum(_normal_loss)
            l2_loss = (_adv_img - x_batch) /255
            l2_loss = np.sum(l2_loss*l2_loss)/8
            li_loss = np.mean( np.amax(np.abs(_adv_img - x_batch) / 255, axis=-1))
            l1_loss = np.mean(np.sum(np.abs(_adv_img - x_batch) / 255, axis=-1))
            #print(_normal_acc)
            print("l2_loss", l2_loss, "li_loss", li_loss, "l1_loss", l1_loss)
            print('step: %d,  total loss: %.3f,  elapsed time: %s' % (step, _loss, elapsed_time))
            print('content loss: %.3f' % (_content_loss))
            print('adv loss  : %.3f,  weighted adv loss: %.3f , adv acc %.3f' %
                  (_adv_loss, adv_weight * _adv_loss, _adv_acc))
            print("_acc_y", _acc_y)
            print("_adv_acc_y", _adv_acc_y)
            #print('normal loss : %.3f normal acc: %.3f\n' %
            #      (_normal_loss, _normal_acc))

        if batch % report_batch == 0:
            np_adv_image_arr = np.concatenate(np_adv_image)
            np_benign_image_arr = np.concatenate(np_benign_image)
            np_content_loss_arr = np.concatenate(np_content_loss)
            np_acc_attack_arr = np.concatenate(np_acc_attack)
            np_acc_arr = np.concatenate(np_acc)
            np_decode_acc_arr = np.concatenate(np_decode_acc)
            np_label_arr = np.concatenate(np_label)

            saved_dict = {"adv_image": np_adv_image_arr, 
                        "benign_image": np_benign_image_arr,
                        "content_loss": np_content_loss_arr,
                        "acc_attack": np_acc_attack_arr,
                        "acc": np_acc_arr,
                        "decode_acc": np_decode_acc_arr,
                          "label": np_label_arr}

            np.save(os.path.join(base_dir_model, "saved_samples%d.npy" %
                                 (batch//report_batch)), saved_dict)

    ###### Done Training & Save the model ######
    #saver.save(sess, model_save_path)

    if debug:
        elapsed_time = datetime.now() - start_time
        print('Done training! Elapsed time: %s' % elapsed_time)
        #print('Model is saved to: %s' % model_save_path)

