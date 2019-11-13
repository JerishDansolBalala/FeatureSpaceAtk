# Train the Style Transfer Net

from __future__ import print_function

import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import tensorflow as tf
import os
import settings

data_set = "cifar10"#"cifar10"#"imagenet"
decoder_name = "cifar10"
model_name = "cifar10_nat"
"""
data_set = "imagenet"  # "cifar10"#"imagenet"
decoder_name = "imagenet_shallow"
model_name = "imagenet_normal" 
"""

task_name = "noise_detect"

exec(open('base.py').read())

import gaussiansmooth.interfacegaussian as gaussdetect
from imagenetmod.interface import  imagenet

if data_set == "imagenet":
    shrink_class = 1000



from style_transfer_net import StyleTransferNet, StyleTransferNet_adv
from utils import get_train_images
from cifar10_class import Model
import cifar10_input
from PIL import Image
from adaptive_instance_norm import normalize
from pgd_attack import LinfPGDAttack
from nn_robust_attacks.l2_attack import CarliniL2


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
        if y >= self.class_num:
            return
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

STYLE_LAYERS = settings.config["STYLE_LAYERS"]

# (height, width, color_channels)
TRAINING_IMAGE_SHAPE = settings.config["IMAGE_SHAPE"]

EPOCHS = 4
EPSILON = 1e-5
BATCH_SIZE = settings.config["BATCH_SIZE"]
if data_set=="cifar10":
    LEARNING_RATE = 1e-1
    LR_DECAY_RATE = 1e-4 #5e-5
    DECAY_STEPS = 1.0
    adv_weight = 5000
    ITER=100
else:
    if decoder_name == "imagenet_shallowest":
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

ENCODER_WEIGHTS_PATH = 'vgg19_normalised.npz'
if data_set == "cifar10":
    Decoder_Model = "./trans_pretrained/cifar10transform1.ckpt-574000"#'transform2.ckpt-147000'#./trans_pretrained/transform.ckpt'
elif data_set == "imagenet":
    if decoder_name == "imagenet_shallowest":
        Decoder_Model = "./imagenetshallowesttransform1.ckpt.mode2"
    elif decoder_name == "imagenet_shallow":
        Decoder_Model = "./imagenetshallowtransform1.ckpt.mode2"#"./trans_pretrained/imagenetshallowtransform1.ckpt-104000"
    elif decoder_name == "imagenet":
        Decoder_Model = "./imagenettransform1.ckpt.mode2"

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
    rst_img, rst_loss, rst_acc, rst_mean, rst_sigma = sess.run(
        [adv_img, content_loss_y, adv_acc_y, stn.meanS, stn.sigmaS],  feed_dict=fdict)

    for i in range(ITER):
        #_,  acc, aloss, closs, closs1, sigma, mean, sigmaS, meanS = sess.run(
        #    [train_op,  adv_acc, adv_loss, content_loss_y, content_loss, stn.sigmaC, stn.meanC, stn.sigmaS, stn.meanS], feed_dict=fdict)
        _ = sess.run([train_op],  feed_dict=fdict)
        sess.run(stn.style_bound, feed_dict=fdict)
        _adv_img, acc, aloss, closs, _mean, _sigma = sess.run(
            [adv_img, adv_acc_y, adv_loss, content_loss_y, stn.meanS, stn.sigmaS],  feed_dict=fdict)
        for j in range(BATCH_SIZE):
            if acc[j] < rst_acc[j] or (acc[j] == rst_acc[j] and closs[j] < rst_loss[j]):
                rst_img[j] = _adv_img[j]
                rst_acc[j] = acc[j]
                rst_loss[j] = closs[j]
                rst_mean[j] = _mean[j]
                rst_sigma[j] = _sigma[j]

        if i % 50 == 0:
            acc = np.mean(acc)
            print(i, acc, "advl", aloss, "contentl", closs)
        if np.sum(acc) == 0 and np.all(np.less_equal(closs, 2*128)):
            break
    sess.run(stn.asgn, feed_dict={
             stn.meanS_ph: rst_mean, stn.sigmaS_ph: rst_sigma})
    return rst_img

def grad_attack1():
    sess.run(stn.init_style, feed_dict=fdict)
    sess.run(global_step.initializer)
    for i in range(ITER):
        #_,  acc, aloss, closs, closs1, sigma, mean, sigmaS, meanS = sess.run(
        #    [train_op,  adv_acc, adv_loss, content_loss_y, content_loss, stn.sigmaC, stn.meanC, stn.sigmaS, stn.meanS], feed_dict=fdict)
        _, acc, aloss, closs, = sess.run([train_op,  adv_acc, adv_loss, content_loss_y],  feed_dict=fdict)
        sess.run(stn.style_bound, feed_dict = fdict)
        if i%100==0 :
            #print("sigma", sigma[0])
            #print("mean", mean[0])
            #print("sigma", sigma)
            #print("mean", mean)
            #print("sigmaS", sigma)
            #print("meanS", mean)
            print(i,acc,"advl",aloss,"contentl",closs)
        #if acc == 0 and np.all(np.less_equal(closs,2*128)):
            #break
            #if i==1:
            #    exit()


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

    if data_set == "cifar10":
        classifier = Model("eval", raw_cifar.train_images)
        classifier._build_model(adv_img, label, reuse=False, conf=0.1)
        adv_loss = - classifier.target_loss
        adv_acc = classifier.accuracy
        adv_acc_y = tf.cast(classifier.correct_prediction, tf.float32)
        classifier._build_model(content, label, reuse=True)
        normal_loss = - classifier.target_loss
        norm_acc = classifier.accuracy
        logits = classifier.pre_softmax
        pgd_attack = LinfPGDAttack(classifier.xent, content, label, epsilon=0.25 *
                                   255, num_steps=200, step_size=0.05*255, random_start=True)
        CarliniL2.pgd_attack()
    elif data_set == "imagenet":
 
        classifier = build_imagenet_model(
            adv_img_bgr, label, conf=0.1, shrink_class=shrink_class)
        adv_loss = - classifier.target_loss
        adv_acc = classifier.accuracy
        adv_acc_y = classifier.acc_y
        adv_acc_y_5 = classifier.acc_y_5
        #logits = classifier.logits

        
        content_bgr = tf.reverse(
            content, axis=[-1])  # switch RGB to BGR
        classifier = build_imagenet_model(
            content_bgr, label, reuse=True, shrink_class=shrink_class)
        normal_loss = - classifier.target_loss
        norm_acc = classifier.accuracy
        acc_y = classifier.acc_y
        acc_y_5 = classifier.acc_y_5
        logits = classifier.logits

        pgd_attack=LinfPGDAttack(classifier.xent,content,label, epsilon= 2.0, num_steps=1000, step_size=2.0/5,random_start=True )
        """
        classifier = build_imagenet_model(img_bgr, label, reuse=True)
        decode_acc_y = classifier.acc_y
        decode_acc_y_5 = classifier.acc_y_5
        """



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
    train_op = gradient(tf.train.AdamOptimizer(learning_rate, beta1= 0.5),vars=stn_vars, loss=loss)

    #train_op = tf.train.AdamOptimizer(learning_rate).minimize(
    #    loss, var_list=stn_vars, global_step=global_step)  

    sess.run(tf.global_variables_initializer())
    if data_set == "cifar10":
        if model_name == "cifar10_adv":
            classifier_saver = tf.train.Saver(classifier_vars, max_to_keep=1)
            classifier_saver.restore(sess, settings.config["hardened_model"])
        elif model_name == "cifar10_nat":
            classifier_saver = tf.train.Saver(classifier_vars, max_to_keep=1)
            classifier_saver.restore(
                sess, settings.config["pretrained_model"])

    elif data_set == "imagenet":
        restore_parameter(sess)


    # saver
    saver = tf.train.Saver(decoder_vars, max_to_keep=1)
    saver.restore(sess,Decoder_Model)
    if data_set=="imagenet":
        image_per_class=10
        
        dp = datapair(shrink_class, batch_size=8, stack_num=image_per_class)
        f = True
        while f:
            x_batch, y_batch = get_data(sess)
            f = dp.feed_pair(x_batch, y_batch)
        print("data loaded")
        x_train = np.concatenate(dp.bucket)
        y_train = np.asarray(
            [i//image_per_class for i in range(image_per_class*shrink_class)], dtype=np.int64)
        gaussdetect.build_detect(sess, batch_size=BATCH_SIZE, base_dir=model_name +
                                "temp", input=content, logits=logits, x_train=x_train, y_train=y_train, dataset=data_set)
    elif data_set=="cifar10":
        batch= 20000// BATCH_SIZE
        x_train=[]
        y_train=[]
        for _ in range(batch):
            _x_batch, _y_batch = get_data(sess)
            x_train.append(_x_batch)
            y_train.append(_y_batch)

        print("data loaded")
        x_train = np.concatenate(x_train)
        y_train = np.concatenate(y_train)
        gaussdetect.build_detect(sess, batch_size=BATCH_SIZE, base_dir=model_name +
                                 "temp", input=content, logits=logits, x_train=x_train, y_train=y_train, dataset=data_set)


    ###### Start Training ######
    step = 0

    if debug:
        elapsed_time = datetime.now() - start_time
        start_time = datetime.now()
        print('\nElapsed time for preprocessing before actually train the model: %s' % elapsed_time)
        print('Now begin to train the model...\n')

    """
    np_dict=np.load(os.path.join("advimg", model_name, "saved_samples.npy")).item()

    np_adv_image_arr = np_dict["adv_image"]
    np_benign_image_arr = np_dict["benign_image"]
    np_content_loss_arr = np_dict["content_loss"]
    np_acc_attack_arr = np_dict["acc_attack"]
    np_acc_attack_5_arr = np_dict["acc_attack_5"]
    np_acc_arr = np_dict["acc"]
    np_acc_5_arr = np_dict["acc_5"]
    np_decode_acc_arr = np_dict["decode_acc"]
    np_decode_acc_5_arr = np_dict["decode_acc_5"]
    np_label_arr = np_dict["label"]    
    """
    if data_set=="imagenet":
        inet = imagenet(BATCH_SIZE, "val")

        def get_data(sess):
            x_batch, y_batch = inet.get_next_batch(sess)
            return x_batch, y_batch
    else:
        raw_cifar = cifar10_input.CIFAR10Data("cifar10_data")
        def get_data(sess):
            x_batch, y_batch = raw_cifar.eval_data.get_next_batch(
                batch_size=BATCH_SIZE, multiple_passes=True)
            return x_batch, y_batch

    """for i in range(1000):
        x_batch,y_batch = get_data(sess)
        for j in y_batch:
            print(j)
            if j>=0 and j<=999:
                pass
            else:
                print(j)
                print("error")
                assert False
    """

    report_batch = 5
    for i in range(11,126):

        if i% report_batch ==1:
            np_adv_image = []
            np_benign_image = []
            np_label =[]
            np_pgd_image = []
            np_pred_normal = []
            np_detection_normal = []
            np_pred_adv = []
            np_detection_adv =[]
            np_pred_pgd = []
            np_detection_pgd =[]            

        x_train_val, y_train_val = get_data(sess)

        x_train_perturbed = pgd_attack.perturb_l2(x_train_val, y_train_val,sess)

        fdict = {content: x_train_perturbed, label: y_train_val}
        _acc = sess.run(norm_acc, feed_dict=fdict)

        fdict = {content: x_train_val, label: y_train_val}

        print("result pgd:")
        _, p_set_pgd, p_det_pgd = gaussdetect.detect(x_train_perturbed, y_train_val, batch_size=BATCH_SIZE)    



        np_pred_pgd.append(p_set_pgd)
        np_detection_pgd.append(p_det_pgd)

        if i % report_batch == 0:

            np_pred_pgd= np.concatenate(np_pred_pgd)
            np_detection_pgd = np.concatenate(np_detection_pgd)

            saved_dict = {
                        "pred_pgd": np_pred_pgd,
                          "detection_pgd": np_detection_pgd,
                        "label": np_label}

            np.save(os.path.join(task_dir, "l2_saved_samples%d.npy" %
                                 (i//report_batch)), saved_dict)
            

    ###### Done Training & Save the model ######
    #saver.save(sess, model_save_path)

    if debug:
        elapsed_time = datetime.now() - start_time
        print('Done training! Elapsed time: %s' % elapsed_time)
        #print('Model is saved to: %s' % model_save_path)

