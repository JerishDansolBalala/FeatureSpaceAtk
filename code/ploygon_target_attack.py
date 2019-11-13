# Train the Style Transfer Net

from __future__ import print_function

import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import tensorflow as tf
import os
import settings
data_set = "imagenet"  # "imagenet"
model_name = "imagenet_denoise"
decoder_name = "imagenet"
task_name = "target_polygon_attack"

exec(open("base.py").read())

from style_transfer_net import StyleTransferNet, StyleTransferNet_adv
from utils import get_train_images
from cifar10_class import Model
import cifar10_input
from PIL import Image
from adaptive_instance_norm import normalize


STYLE_LAYERS = settings.config["STYLE_LAYERS"]

# (height, width, color_channels)
TRAINING_IMAGE_SHAPE = settings.config["IMAGE_SHAPE"]


class datapair():
    def __init__(self, class_num, batch_size, stack_num=10):
        self.class_num = class_num
        self.batch_size = batch_size
        self.bucket = [[] for _ in range(self.class_num)]
        self.bucket_size = [0 for _ in range(self.class_num)]
        self.tot_pair = 0
        self.index = 0
        self.stack_num = stack_num
        self.loaded = 0

    def add_data(self, x, y):

        if self.bucket_size[y] < self.stack_num:
            self.bucket[y].append(x)
            self.bucket_size[y] += 1
            if self.bucket_size[y] == self.stack_num:
                self.loaded += 1
                self.bucket[y] = np.stack(self.bucket[y])

    def feed_pair(self, x_batch, y_batch):
        for i in range(self.batch_size):
            self.add_data(x_batch[i], y_batch[i])
        if self.loaded == self.class_num:
            return False
        else:
            return True

EPOCHS = 4
EPSILON = 1e-5
BATCH_SIZE = settings.config["BATCH_SIZE"]
if data_set=="cifar10":
    LEARNING_RATE = 1e-1
    LR_DECAY_RATE = 1e-4 #5e-5
    DECAY_STEPS = 1.0
    adv_weight = 5000
    ITER=4000
else:
    if model_name == "imagenet_shallowest":
        LEARNING_RATE = 5e-3
    else:
        LEARNING_RATE = 1e-1
    LR_DECAY_RATE = 1e-3 # 5e-5
    DECAY_STEPS = 1.0
    adv_weight = 128 
    ITER=1000
    CLIP_NORM_VALUE = 10.0
    INCLUDE_SELF = False

style_weight = 1

if data_set == "cifar10":
    raw_cifar = cifar10_input.CIFAR10Data("cifar10_data")

    def get_data(sess):
        x_batch, y_batch = raw_cifar.train_data.get_next_batch(
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
    #sess.run(stn.init_style, feed_dict=fdict)
    sess.run(global_step.initializer)
    rst_img, rst_loss, rst_acc,rst_coef = sess.run(
        [adv_img, content_loss_y, adv_acc_y_5, stn.coef],  feed_dict=fdict)
    
    for i in range(ITER):
        #_,  acc, aloss, closs, closs1, sigma, mean, sigmaS, meanS = sess.run(
        #    [train_op,  adv_acc, adv_loss, content_loss_y, content_loss, stn.sigmaC, stn.meanC, stn.sigmaS, stn.meanS], feed_dict=fdict)
        _ = sess.run([train_op],  feed_dict=fdict)
        sess.run(stn.regulate)
        _succ_attack,_succ_attack_rate,_adv_img, acc, aloss, closs, _coef= sess.run( [succ_attack,succ_attack_rate, adv_img, adv_acc_y_5, adv_loss, content_loss_y, stn.coef],  feed_dict = fdict)
        for j in range(BATCH_SIZE):
            if _succ_attack[j]>rst_acc[j] or (acc[j]==rst_acc[j] and closs[j]<rst_loss[j]):
                rst_img[j]=_adv_img[j]
                rst_acc[j] = _succ_attack[j]
                rst_loss[j] = closs[j]
                rst_coef[j] = _coef[j]

        if i%1==0 :
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
            print(i, acc,
                  "advl", aloss, "contentl", closs, "succ_atk", _succ_attack_rate)
        if np.sum(acc) == 0 and np.all(np.less_equal(closs,2*128)):
            break
            #if i==1:
            #    exit()
    sess.run(stn.coef_asgn, feed_dict={stn.coef_ph: rst_coef})
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

if not INCLUDE_SELF:
    settings.config["INTERPOLATE_NUM"] -= 1

if data_set == "cifar10":
    INTERNAL_SHAPE = (BATCH_SIZE, 16, 16, 128)
    FEATURE_SPACE = (None, 1, 1, 128)
else:
    INTERPOLATE_NUM = settings.config["INTERPOLATE_NUM"]
    INTERNAL_SHAPE = (BATCH_SIZE, 56, 56, 256)  # (BATCH_SIZE, 28, 28, 512)
    FEATURE_SPACE = (BATCH_SIZE, 1, 1, 256)
    INTERPOLATE_SHAPE = (BATCH_SIZE, INTERPOLATE_NUM, 1, 1, 256)


def gradient(opt, vars, loss):
    gradients, variables = zip(*opt.compute_gradients(loss, vars))
    g_split = tf.unstack(gradients[0], BATCH_SIZE, axis=0)
    print(g_split)
    g1_list = []
    DIM = settings.config["DECODER_DIM"][-1]
    limit = 10/np.sqrt(DIM)
    for g1 in g_split:
        g1 =  tf.clip_by_value(g1,-1/np.sqrt(limit),1/np.sqrt(limit))
        g1_list.append(g1)
    gradients = [tf.stack(g1_list, axis=0)]
    #gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    opt = opt.apply_gradients(
        zip(gradients, variables), global_step=global_step)
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
    generated_img, generated_img_adv = stn.transform_from_internal_poly(
        content)
    internal_sigma = stn.internal_sigma
    internal_mean = stn.internal_mean
    img = generated_img
    adv_img = generated_img_adv

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

    def create_random_target(label):
        label_offset = tf.random_uniform(
            tf.shape(label), minval=1, maxval=1000, dtype=tf.int64)
        return tf.floormod(label + label_offset, tf.constant(1000, tf.int64))

    if data_set == "cifar10":
        classifier = Model("eval", raw_cifar.train_images)
        classifier._build_model(adv_img, label, reuse=False)
        adv_loss = - classifier.relaxed_y_xent
        adv_acc = classifier.accuracy
        classifier._build_model(img, label, reuse=True)
        normal_loss = - classifier.relaxed_y_xent
        norm_acc = classifier.accuracy
    elif data_set == "imagenet":
        classifier = build_imagenet_model(adv_img_bgr, label, conf=1)
        target_label = tf.Variable(
            initial_value=[0 for _ in range(BATCH_SIZE)], dtype=tf.int64)

        label_asgn = tf.assign(
            target_label, create_random_target(target_label))
        target_loss = -tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=classifier.logits, labels=target_label)
        #target_loss = - tf.nn.relu(tf.reduce_sum((tf.one_hot(classifier.label, depth=1000) -
        #               tf.one_hot(target_label, depth=1000))*classifier.logits, axis=-1)+1)
            
        succ_attack = tf.cast(
            tf.equal(target_label, classifier.label), tf.float32)
        succ_attack_rate = tf.reduce_mean(succ_attack)

        adv_loss = - target_loss # classifier.target_loss5
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

    loss=loss
    if data_set == "cifar10":
        classifier_vars = get_scope_var("model")
    decoder_vars = get_scope_var("decoder")
    # Training step
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.inverse_time_decay(LEARNING_RATE, global_step, DECAY_STEPS, LR_DECAY_RATE)
    #tf.train.AdamOptimizer(learning_rate).minimize(  # MomentumOptimizer(learning_rate, momentum=0.9) tf.train.GradientDescentOptimizer(learning_rate)
    #train_op = tf.train.AdamOptimizer(learning_rate).minimize(
    #    loss, var_list=stn_vars, global_step=global_step)

    ##gradient clipping
    train_op = gradient(tf.train.GradientDescentOptimizer(
        learning_rate), vars=stn_vars, loss=loss)

    #train_op = tf.train.AdamOptimizer(learning_rate).minimize(
    #    loss, var_list=stn_vars, global_step=global_step)  

    sess.run(tf.global_variables_initializer())
    if data_set == "cifar10":
        classifier_saver = tf.train.Saver(classifier_vars, max_to_keep=1)
        classifier_saver.restore(sess, settings.config["hardened_model"])
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

    mean_file = os.path.join(task_dir,"polygon_mean.npy" )
    sigma_file = os.path.join(task_dir,"polygon_sigma.npy" )
    if os.path.exists(mean_file) and os.path.exists(sigma_file):
        _mean_all = np.load(mean_file)
        _sigma_all = np.load(sigma_file)
    else:
        ## Populate polygon point
        dp = datapair(1000, batch_size=8, stack_num=INTERPOLATE_NUM-1)
        f = True
        while f:
            x_batch, y_batch = get_data(sess)
            f = dp.feed_pair(x_batch, y_batch)
        print("data loaded")
        polygon_arr = np.concatenate(dp.bucket)
        len_arr = polygon_arr.shape[0]
        _mean = []
        _sigma = []
        for i in range((len_arr - 1) // BATCH_SIZE + 1):
            _meanC, _sigmaC = sess.run([stn.meanC, stn.sigmaC], feed_dict={
                content: polygon_arr[i*BATCH_SIZE:(i+1)*BATCH_SIZE]})
            _mean.append(_meanC)
            _sigma.append(_sigmaC)
            print(i)
        _mean_all = np.concatenate(_mean, axis=0)
        _sigma_all = np.concatenate(_sigma, axis=0)
        np.save(mean_file, _mean_all)
        np.save(sigma_file, _sigma_all)

    def popoulate_data(_meanC, _sigmaC, y_batch, include_self=True):

        res_mean = []
        res_sigma = []

        if include_self:
            real_num = INTERPOLATE_NUM - 1
            for i in range(BATCH_SIZE):
                y = y_batch[i]
                meanCi = _meanC[i: i+1]
                meanC_pop = _mean_all[y*real_num:(y+1)*real_num]
                res_mean.append(np.concatenate([meanCi, meanC_pop]))
                sigmaCi = _sigmaC[i: i+1]
                sigmaC_pop = _sigma_all[y*real_num:(y+1)*real_num]
                res_sigma.append(np.concatenate([sigmaCi, sigmaC_pop]))
        else:
            real_num = INTERPOLATE_NUM
            for i in range(BATCH_SIZE):
                y = y_batch[i]
                assert y<1000
                meanC_pop = _mean_all[y*real_num:(y+1)*real_num]
                res_mean.append(meanC_pop)
                sigmaCi = _sigmaC[i: i+1]
                sigmaC_pop = _sigma_all[y*real_num:(y+1)*real_num]
                res_sigma.append(sigmaC_pop)
        return np.stack(res_mean), np.stack(res_sigma)


    uid = 0

    report_batch = 50
    for batch in range(1,1000+1):

        if batch % report_batch == 1:
            np_adv_image = []
            np_benign_image = []
            np_content_loss = []
            np_acc_attack = []
            np_acc_attack_5 = []
            np_acc = []
            np_acc_5 = []
            np_decode_acc = []
            np_decode_acc_5 = []
            np_acc_5 = []
            np_label = []
            np_succ_attack = []
            np_target_label = []
        # run the training step

        x_batch, y_batch = get_data(sess)
        fdict = {content: x_batch, label: y_batch}
        _meanC, _sigmaC = sess.run([stn.meanC, stn.sigmaC], feed_dict={
            content: x_batch})

        sess.run(label_asgn)
        attack_label = sess.run(target_label)
        _meanC, _sigmaC = popoulate_data(
            _meanC, _sigmaC, attack_label, include_self=INCLUDE_SELF)

        fdict = {internal_mean: _meanC, internal_sigma: _sigmaC,
                 label: y_batch, content: x_batch}
        grad_attack()
        step += 1

        for i in range(BATCH_SIZE):
            gan_out = sess.run(adv_img, feed_dict=fdict)
            save_out = np.concatenate(
                (gan_out[i], x_batch[i], np.abs(gan_out[i]-x_batch[i])))
            sz = TRAINING_IMAGE_SHAPE[1]
            full_path = os.path.join(
                task_dir, "%d" % step,  "%d.jpg" % i)
            os.makedirs(os.path.join(task_dir, "%d" %
                                     step), exist_ok=True)
            save_out = np.reshape(save_out, newshape=[sz*3, sz, 3])
            save_rgb_img(save_out, path=full_path)

        if batch % 1 == 0:

            elapsed_time = datetime.now() - start_time
            _content_loss, _adv_acc, _adv_loss, _loss,   \
                = sess.run([ content_loss, adv_acc, adv_loss, loss,], feed_dict=fdict)
            _adv_img, _loss_y, _adv_acc_y, _adv_acc_y_5, _acc_y, _acc_y_5, _decode_acc_y, _decode_acc_y_5, _succ_attack,_target_label = sess.run([
                adv_img, content_loss_y, adv_acc_y, adv_acc_y_5, acc_y, acc_y_5, decode_acc_y, decode_acc_y_5, succ_attack, target_label], feed_dict=fdict)
            #_normal_loss, _normal_acc = sess.run([normal_loss, norm_acc], feed_dict=fdict)
            np_adv_image.append(_adv_img)
            np_benign_image.append(x_batch)
            np_content_loss.append(_loss_y)
            np_acc_attack.append(_adv_acc_y)
            np_acc_attack_5 .append(_adv_acc_y_5)
            np_acc_5 .append(_acc_y_5)
            np_acc .append(_acc_y)
            np_label.append(y_batch)
            np_decode_acc.append(_decode_acc_y)
            np_decode_acc_5.append(_decode_acc_y_5)
            np_succ_attack .append(_succ_attack)
            np_target_label.append(_target_label)

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
            print("_acc_y_5", _acc_y_5)
            print("_adv_acc_y_5", _adv_acc_y_5)
            #print('normal loss : %.3f normal acc: %.3f\n' %
            #      (_normal_loss, _normal_acc))

        if batch % report_batch == 0:
            np_adv_image_arr = np.concatenate(np_adv_image)
            np_benign_image_arr = np.concatenate(np_benign_image)
            np_content_loss_arr = np.concatenate(np_content_loss)
            np_acc_attack_arr = np.concatenate(np_acc_attack)
            np_acc_attack_5_arr = np.concatenate(np_acc_attack_5)
            np_acc_arr = np.concatenate(np_acc)
            np_acc_5_arr = np.concatenate(np_acc_5)
            np_decode_acc_arr = np.concatenate(np_decode_acc)
            np_decode_acc_5_arr = np.concatenate(np_decode_acc_5)
            np_label_arr = np.concatenate(np_label)
            np_succ_attack = np.concatenate(np_succ_attack)
            np_target_label = np.concatenate(np_target_label)

            saved_dict = {"adv_image": np_adv_image_arr, 
                        "benign_image": np_benign_image_arr,
                        "content_loss": np_content_loss_arr,
                        "acc_attack": np_acc_attack_arr,
                        "acc_attack_5": np_acc_attack_5_arr,
                        "acc": np_acc_arr,
                        "acc_5": np_acc_5_arr,   
                        "decode_acc": np_decode_acc_arr,
                        "decode_acc_5": np_decode_acc_5_arr,
                          "label": np_label_arr,
                          "succ_attack": np_succ_attack,
                          "target_label":np_target_label}

            np.save(os.path.join(task_dir, "saved_samples%d.npy" %
                                 (batch//report_batch)), saved_dict)

    ###### Done Training & Save the model ######
    #saver.save(sess, model_save_path)

    if debug:
        elapsed_time = datetime.now() - start_time
        print('Done training! Elapsed time: %s' % elapsed_time)
        #print('Model is saved to: %s' % model_save_path)

