# Train the Style Transfer Net

from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import logging

import settings
data_set = "imagenet"  # "imagenet"
model_name = "imagenet_denoise"
decoder_name = "imagenet_shallow"
task_name = "attack"

exec(open('base.py').read())

#data_set="cifar10"
#settings.init_settings("cifar10")
logger=settings.logger

from imagenetmod.interface import build_imagenet_model, imagenet, restore_parameter
from style_transfer_net import StyleTransferNet
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
LEARNING_RATE = 1e-4
LR_DECAY_RATE = 2e-5
# 2e-5  30000 -> half
DECAY_STEPS = 1.0
adv_weight=500
style_weight=settings.config["style_weight"]

class datapair():
    def __init__(self,class_num, batch_size):
        self.class_num=class_num
        self.batch_size=batch_size
        self.bucket=[ [] for _ in range(self.class_num)]
        self.bucket_size= [0 for _ in range(self.class_num)]
        self.tot_pair=0
        self.index=0

    def add_data(self,x,y):
        self.bucket_size[y]+=1
        
        if self.bucket_size[y] % 2==0:
            self.tot_pair+=1
        self.bucket[y].append(x)

    def feed_pair(self,x_batch,y_batch):
        for i in range(self.batch_size):
            self.add_data(x_batch[i],y_batch[i])
    
    def get_pair(self):
        if self.tot_pair<self.batch_size:
            return None
        else:
            x1=[]
            y1=[]
            x2=[]
            y2=[]
            left=self.batch_size
            i = self.index  # ensure random start of each class
            for _ in range(self.class_num):
                if left==0:
                    break
                sz = self.bucket_size[i]
                if sz>=2:
                    pairs=min(left,sz//2)
                else:
                    i = (i+1) % self.class_num
                    continue
                x1.extend(self.bucket[i][:pairs])
                x2.extend(self.bucket[i][pairs:2*pairs])
                y1.extend([i]*pairs)
                y2.extend([i]*pairs)
                self.bucket[i] = self.bucket[i][2*pairs:]
                self.bucket_size[i]-=2*pairs
                left-=pairs
                i= (i+1)%self.class_num
                #print(i)
            self.index = i
            self.tot_pair-=self.batch_size
            x1=np.stack(x1)
            x2=np.stack(x2)
            y1=np.stack(y1)
            y2=np.stack(y2)
        return x1,y1,x2,y2


model_save_path = settings.config["model_save_path"]

if data_set == "cifar10":
    raw_cifar = cifar10_input.CIFAR10Data("cifar10_data")
    mode = 2
    if mode == 1:
        def get_data(sess):
            x_batch, y_batch = raw_cifar.train_data.get_next_batch(
                    batch_size=BATCH_SIZE, multiple_passes=True)
            x_batch_style, y_batch_style = raw_cifar.train_data.get_next_batch(
                batch_size=BATCH_SIZE, multiple_passes=True)
            return x_batch, y_batch, x_batch_style, y_batch_style
    else:
        dp = datapair(10, BATCH_SIZE)
        def get_data(sess):
            res = dp.get_pair()
            while res is None:
                x_batch, y_batch = raw_cifar.train_data.get_next_batch(
                    batch_size=BATCH_SIZE, multiple_passes=True)
                dp.feed_pair(x_batch, y_batch)
                res = dp.get_pair()
            return res

elif data_set == "imagenet":
    inet = imagenet(BATCH_SIZE, dataset="train")
    mode = settings.config["data_mode"]
    model_save_path+=".mode%d"%mode
    if mode == 1:
        def get_data(sess):
            x_batch, y_batch = inet.get_next_batch(sess)
            x_batch_style, y_batch_style = inet.get_next_batch(sess)
            return x_batch, y_batch, x_batch_style, y_batch_style
    else:
        dp = datapair(1000,BATCH_SIZE)
        def get_data(sess):
            res = dp.get_pair()
            while res is None:
                x_batch, y_batch = inet.get_next_batch(sess)
                dp.feed_pair(x_batch,y_batch)
                res = dp.get_pair()
            return res

get_data(None)

ENCODER_WEIGHTS_PATH = 'vgg19_normalised.npz'


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
debug=True
logging_period=100
if debug:
    from datetime import datetime
    start_time = datetime.now()



# get the traing image shape
HEIGHT, WIDTH, CHANNELS = TRAINING_IMAGE_SHAPE
INPUT_SHAPE = [None, HEIGHT, WIDTH, CHANNELS]
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
    stn = StyleTransferNet(encoder_path)

    # pass content and style to the stn, getting the generated_img
    generated_img , generated_img_adv = stn.transform(content, style)
    adv_img=generated_img_adv
    img = generated_img

    print(adv_img.shape.as_list())
    stn_vars = []#get_scope_var("transform")
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

    l2_embed = normalize(enc_gen)[0] - normalize(stn.norm_features)[0]
    l2_embed = tf.reduce_mean(tf.sqrt(tf.reduce_sum((l2_embed * l2_embed),axis=[1,2,3])))

    # compute the content loss
    content_loss = tf.reduce_sum(tf.reduce_mean(
        tf.square(enc_gen_adv - target_features), axis=[1, 2])) 
    #content_loss += tf.reduce_sum(tf.reduce_mean(
    #    tf.square(enc_gen - stn.norm_features), axis=[1, 2]))

    # Build the classifier
    if data_set=="cifar10":
        classifier = Model("eval", raw_cifar.train_images)
        classifier._build_model(adv_img, label, reuse=False)
        adv_loss = - classifier.relaxed_y_xent
        adv_acc = classifier.accuracy
        classifier._build_model(img, label, reuse=True)
        normal_loss = - classifier.relaxed_y_xent
        norm_acc = classifier.accuracy
    elif data_set=="imagenet":
        classifier = build_imagenet_model(adv_img_bgr, label)
        adv_loss = - classifier.xent
        adv_acc = classifier.accuracy
        classifier = build_imagenet_model(img_bgr, label, reuse=True)
        normal_loss = - classifier.xent
        norm_acc = classifier.accuracy
    # compute the style loss
    
    style_layer_loss = []
    for layer in STYLE_LAYERS:
        enc_style_feat = stn.encoded_style_layers[layer]
        enc_gen_feat = enc_gen_layers_adv[layer]  # enc_gen_layers[layer]

        meanS, varS = tf.nn.moments(enc_style_feat, [1, 2])
        meanG, varG = tf.nn.moments(enc_gen_feat,   [1, 2])

        sigmaS = tf.sqrt(varS + EPSILON)
        sigmaG = tf.sqrt(varG + EPSILON)

        l2_mean  = tf.reduce_sum(tf.square(meanG - meanS))
        l2_sigma = tf.reduce_sum(tf.square(sigmaG - sigmaS))

        style_layer_loss.append(l2_mean + l2_sigma)

    style_loss = tf.reduce_sum(style_layer_loss)

    
    # compute the total loss
    loss = content_loss +  style_weight * style_loss #adv_loss * adv_weight
    if data_set == "cifar10":
        classifier_vars = get_scope_var("model")
    decoder_vars = get_scope_var("decoder")
    # Training step
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.inverse_time_decay(LEARNING_RATE, global_step, DECAY_STEPS, LR_DECAY_RATE)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(
        loss, var_list=stn_vars+decoder_vars, global_step=global_step)  # stn_vars+

    sess.run(tf.global_variables_initializer())
    if data_set=="cifar10":
        classifier_saver = tf.train.Saver(classifier_vars, max_to_keep=1)
        classifier_saver.restore(sess, settings.config["hardened_model"])
    elif data_set=="imagenet":
        restore_parameter(sess)

    # saver
    saver = tf.train.Saver(stn_vars+decoder_vars, max_to_keep=1)
    saver.restore(sess, model_save_path)
    #saver.restore(sess, "imagenetshallowtransform1.ckpt-80000")
    ###### Start Training ######
    step = 0

    if debug:
        elapsed_time = datetime.now() - start_time
        start_time = datetime.now()
        print('\nElapsed time for preprocessing before actually train the model: %s' % elapsed_time)
        print('Now begin to train the model...\n')

    def save_img_gen(lst , prefix = ""):
        
        def rs(x):
            nonlocal i,cnt
            cnt+=1
            full_path = os.path.join(
                "temp", "%s_%d_%d.jpg" % (prefix,i,cnt))
            print(x.shape)
            x= np.reshape(x, newshape=[sz, sz, 3])
            save_rgb_img(x, path=full_path)

        os.makedirs(os.path.join("temp"), exist_ok=True)
        for i in range(BATCH_SIZE):
            cnt = 0
            sz = TRAINING_IMAGE_SHAPE[1]

            for j in range(len(lst)):
                rs(lst[j][i])


    for batch in range(10):
        # retrive a batch of content and style images
        #content_batch_path = content_imgs_path[batch*BATCH_SIZE:(batch*BATCH_SIZE + BATCH_SIZE)]
        #style_batch_path   = style_imgs_path[batch*BATCH_SIZE:(batch*BATCH_SIZE + BATCH_SIZE)]

        #content_batch = get_train_images(content_batch_path, crop_height=HEIGHT, crop_width=WIDTH)
        #style_batch   = get_train_images(style_batch_path,   crop_height=HEIGHT, crop_width=WIDTH)

        # run the training step
        x_batch, y_batch, x_batch_style, y_batch_style = get_data(sess)
        fdict = {content: x_batch, label: y_batch, style: x_batch_style}

        _img = sess.run(adv_img,feed_dict=fdict)
        save_img_gen([x_batch,x_batch_style,_img],"stylemotivation")





    if debug:
        elapsed_time = datetime.now() - start_time
        print('Done training! Elapsed time: %s' % elapsed_time)
        print('Model is saved to: %s' % model_save_path)

