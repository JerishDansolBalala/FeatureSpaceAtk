# Train the Style Transfer Net

from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import settings
data_set = "imagenet"#"imagenet"
data_set_name = "imagenet_shallowest"
settings.init_settings(data_set_name, suffix="_attack")
logger=settings.logger

from imagenetmod.interface import build_imagenet_model, imagenet, restore_parameter
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
    LEARNING_RATE = 1e-1
    LR_DECAY_RATE = 1e-4 #5e-5
    DECAY_STEPS = 1.0
    adv_weight = 5000
    ITER=4000
else:
    #LEARNING_RATE = 5e-2
    #LR_DECAY_RATE = 1e-5  # 5e-5
    DECAY_STEPS = 1.0
    adv_weight = 128 
    ITER=100

style_weight = 1

if data_set == "cifar10":
    raw_cifar = cifar10_input.CIFAR10Data("cifar10_data")

    def get_data(sess):
        x_batch, y_batch = raw_cifar.train_data.get_next_batch(
            batch_size=BATCH_SIZE, multiple_passes=True)
        return x_batch, y_batch
elif data_set == "imagenet":
    inet = imagenet(BATCH_SIZE, dataset="train")

    def get_data(sess):
        x_batch, y_batch = inet.get_next_batch(sess)

        return x_batch, y_batch

ENCODER_WEIGHTS_PATH = 'vgg19_normalised.npz'
if data_set == "cifar10":
    Decoder_Model = "./trans_pretrained/cifar10transform1.ckpt-574000"#'transform2.ckpt-147000'#./trans_pretrained/transform.ckpt'
elif data_set == "imagenet":
    if data_set_name == "imagenet_shallow":
        Decoder_Model = "./imagenetshallowtransform1.ckpt.mode2"
    elif data_set_name == "imagenet_shallowest":
        Decoder_Model = "./imagenetshallowesttransform1.ckpt.mode2"

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
    for i in range(ITER):
        _,  acc, aloss, closs = sess.run(
            [train_op,  adv_acc, adv_loss, content_loss], feed_dict=fdict)
        sess.run(stn.style_bound, feed_dict = fdict)
        if i%10==0 :
            print(i,acc,"advl",aloss,"contentl",closs)
            #if acc < 0.05 and closs < 2000:
            #    break

def rand_attack():
    for i in range(10):
        sess.run(stn.init_style, feed_dict=fdict)
        sess.run(global_step.initializer)
        for j in range(10):
            _,  acc, aloss, closs = sess.run(
                [train_op,  adv_acc, adv_loss, content_loss], feed_dict=fdict)
            sess.run(stn.style_bound, feed_dict = fdict)
        print(i,acc,"advl",aloss,"contentl",closs)
        if acc < 0.05 and closs < 2000:
            break


# get the traing image shape
HEIGHT, WIDTH, CHANNELS = TRAINING_IMAGE_SHAPE
INPUT_SHAPE = (None, HEIGHT, WIDTH, CHANNELS)

# create the graph
tf_config = tf.ConfigProto()
#tf_config.gpu_options.per_process_gpu_memory_fraction=0.5
tf_config.gpu_options.allow_growth = True
with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:

    content = tf.placeholder(tf.float32, shape=INPUT_SHAPE, name='content')
    style = tf.placeholder(tf.float32, shape=INPUT_SHAPE, name='style')
    label = tf.placeholder(tf.int64, shape =None, name="label")
    #style   = tf.placeholder(tf.float32, shape=INPUT_SHAPE, name='style')
    loss_choice = tf.placeholder(tf.int64, shape = None)
    LEARNING_RATE = tf.placeholder(tf.float32, shape = None)
    LR_DECAY_RATE = tf.placeholder(tf.float32, shape = None)

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
        classifier._build_model(adv_img, label, reuse=False)
        adv_loss = - classifier.relaxed_y_xent
        adv_acc = classifier.accuracy
        classifier._build_model(img, label, reuse=True)
        normal_loss = - classifier.relaxed_y_xent
        norm_acc = classifier.accuracy
    elif data_set == "imagenet":
        classifier = build_imagenet_model(adv_img_bgr, label)
        losses = tf.stack(
            [classifier.rev_xent, classifier.poss_loss, classifier.xent, classifier.xent_filter])
        adv_loss = - tf.reduce_sum(losses[loss_choice])
        adv_acc = classifier.accuracy
        acc_y = classifier.acc_y
        classifier = build_imagenet_model(img_bgr, label, reuse=True)
        losses = tf.stack(
            [classifier.rev_xent, classifier.poss_loss, classifier.xent, classifier.xent_filter])
        normal_loss = - tf.reduce_sum(losses[loss_choice])
        norm_acc = classifier.accuracy

    # compute the content loss
    bar=3000/64/128
    #content_loss = tf.reduce_sum(tf.nn.relu( 
    #    tf.reduce_mean(tf.square(enc_gen_adv - target_features), axis=[1, 2]) - bar)
    #    )
    content_loss_y = tf.reduce_sum(
        tf.reduce_mean(tf.square(enc_gen_adv - target_features), axis=[1, 2]), axis=1)
    content_loss = tf.reduce_sum(
        tf.reduce_mean(tf.square(enc_gen_adv - target_features), axis=[1, 2]))
    #content_loss += tf.reduce_sum(tf.reduce_mean(
    #    tf.square(enc_gen - stn.norm_features), axis=[1, 2]))

    # compute the style loss
    
    style_layer_loss = []

    # compute the total loss
    # adv_loss * adv_weight
    loss = content_loss + adv_loss  * adv_weight# style_weight * style_loss

    loss=loss
    if data_set == "cifar10":
        classifier_vars = get_scope_var("model")
    decoder_vars = get_scope_var("decoder")
    # Training step
    global_step = tf.Variable(0, trainable=False)
    
    learning_rate = tf.train.inverse_time_decay(LEARNING_RATE, global_step, DECAY_STEPS, LR_DECAY_RATE)
    #tf.train.AdamOptimizer(learning_rate).minimize(  # 
    train_ops = {}
    train_ops["momentum"] = tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(
        loss, var_list = stn_vars, global_step = global_step)
    train_ops ["adam"] = tf.train.AdamOptimizer(learning_rate).minimize(
        loss, var_list=stn_vars, global_step=global_step)
    train_ops ["sgd"] = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss, var_list=stn_vars, global_step=global_step)

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



    saved_fdict = []
    for i in range(10):
        x_batch, y_batch = get_data(sess)
        fdict = {content: x_batch, label: y_batch}
        saved_fdict.append(fdict)
    
    content_loss_thresh = 500
    logger.info("content thresh: %.2f adv weight: %.2f" % (content_loss_thresh, adv_weight) )
    img_id=0
    for opt_name in ["momentum", "adam", "sgd"]:
        for learning_rate in [1e-1,1e-2,1e-3]:
            for ls_choice in [0,1,2,3]:
                for lr_decay in [1e-1,1e-2,1e-3]:
                    img_id += 1
                    param = {"opt_name":opt_name,
                            "lr":learning_rate,
                            "ls_choice":ls_choice,
                            "lr_decay":lr_decay,
                            "img_id":img_id}
                    
                    logger.info(str(param))
                    cnt = 0
                    cltot=0
                    acctot=0
                    tot = 0
                    sg_id=0
                    for fdict in saved_fdict:
                        x_batch=fdict[content]
                        fdict[LEARNING_RATE]=learning_rate
                        fdict[loss_choice]=ls_choice
                        fdict[LR_DECAY_RATE]=lr_decay
                        train_op = train_ops[opt_name]
                        grad_attack()
                    
                        _loss,_acc = sess.run([content_loss_y, acc_y],fdict)
                        for sid in range(BATCH_SIZE):
                            if _loss[sid] < 500 and _acc[sid]==0:
                                cnt += 1
                            acctot += _acc[sid]
                            cltot += _loss[sid]
                            tot += 1

                        for i in range(BATCH_SIZE):
                            sg_id+=1
                            gan_out = sess.run(adv_img, feed_dict=fdict)
                            save_out = np.concatenate(
                                (gan_out[i], x_batch[i], np.abs(gan_out[i]-x_batch[i])))
                            sz = TRAINING_IMAGE_SHAPE[1]
                            full_path = os.path.join(
                                "advimg"+data_set_name, "%d" % img_id,  "%d.jpg" % sg_id)
                            os.makedirs(os.path.join(
                                "advimg"+data_set_name, "%d" % img_id), exist_ok=True)
                            save_out = np.reshape(save_out, newshape=[sz*3, sz, 3])
                            save_rgb_img(save_out, path=full_path)

                    logger.info("avg content %.2f, acc %.2f, succ_attk %.2f\r\n" % 
                        (float(cltot)/tot, float(acctot)/tot , float(cnt)/tot))



            """if batch % 1 == 0:

            elapsed_time = datetime.now() - start_time
            _adv_img, _content_loss, _adv_acc, _adv_loss, _loss, = sess.run([adv_img, content_loss, adv_acc , adv_loss, loss],
                                                                  feed_dict=fdict)
            #_normal_loss, _normal_acc = sess.run([normal_loss, norm_acc],
            #                                     feed_dict=fdict)
            l2_loss = (_adv_img - x_batch) /255
            l2_loss = np.sum(l2_loss*l2_loss)/8
            li_loss = np.mean( np.amax(np.abs(_adv_img - x_batch) / 255, axis=-1))
            l1_loss = np.mean(np.sum(np.abs(_adv_img - x_batch) / 255, axis=-1))
            print("l2_loss", l2_loss, "li_loss", li_loss, "l1_loss", l1_loss)
            print('step: %d,  total loss: %.3f,  elapsed time: %s' % (step, _loss, elapsed_time))
            print('content loss: %.3f' % (_content_loss))
            print('adv loss  : %.3f,  weighted adv loss: %.3f , adv acc %.3f' %
                  (_adv_loss, adv_weight * _adv_loss, _adv_acc))
            #print('normal loss : %.3f normal acc: %.3f\n' %
            #      (_normal_loss, _normal_acc))
            """

    ###### Done Training & Save the model ######
    #saver.save(sess, model_save_path)

    if debug:
        elapsed_time = datetime.now() - start_time
        print('Done training! Elapsed time: %s' % elapsed_time)
        #print('Model is saved to: %s' % model_save_path)

