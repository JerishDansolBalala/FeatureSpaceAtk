# Train the Style Transfer Net

from __future__ import print_function

import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import tensorflow as tf
import os
import settings

"""
data_set = "imagenet"#"imagenet"
model_name = "imagenet_denoise"
decoder_name = "imagenet"
"""

tid = 2

task = "trades_cifar10_adv"
task_name = "test"+task
if task == "imagenet_denoise_untargeted":
    data_set = "imagenet"  # "imagenet"
    model_name = "imagenet_denoise"
    decoder_name = "imagenet_shallow"
    exec(open("base.py").read())

elif task == "imagenet_denoise_untargeted_ns":
    data_set = "imagenet"  # "imagenet"
    model_name = "imagenet_denoise"
    decoder_name = "imagenet"
    exec(open("base.py").read())

elif task == "imagenet_denoise_targeted":
    data_set = "imagenet"  # "imagenet"
    model_name = "imagenet_denoise"
    decoder_name = "imagenet"
    exec(open("base.py").read())
    base_dir_model = os.path.join(base_dir_model,"target_attack")

elif task =="madry_cifar10_adv":
    data_set = "cifar10"  # "imagenet"
    model_name = "cifar10_adv"
    decoder_name = "cifar10"
    exec(open("base.py").read())

elif task =="trades_cifar10_adv":
    data_set = "cifar10"  # "imagenet"
    model_name = "cifar10_trades"
    decoder_name = "cifar10_balance"
    exec(open("base.py").read())    

elif task == "madry_cifar10_adv_balance":
    data_set = "cifar10"  # "imagenet"
    model_name = "cifar10_adv"
    decoder_name = "cifar10_balance"
    exec(open("base.py").read())

#base_dir_model = os.path.join(base_dir_model, "target_attack")

from style_transfer_net import StyleTransferNet, StyleTransferNet_adv
from utils import get_train_images
from cifar10_class import Model
import cifar10_input
from PIL import Image
from adaptive_instance_norm import normalize
from pgd_attack import LinfPGDAttack
from trade_interface import build_model as build_model_trades

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
    for i in range(ITER):
        #_,  acc, aloss, closs, closs1, sigma, mean, sigmaS, meanS = sess.run(
        #    [train_op,  adv_acc, adv_loss, content_loss_y, content_loss, stn.sigmaC, stn.meanC, stn.sigmaS, stn.meanS], feed_dict=fdict)
        _, acc, aloss, closs, = sess.run([train_op,  adv_acc, adv_loss, content_loss_y],  feed_dict=fdict)
        sess.run(stn.style_bound, feed_dict = fdict)
        if i%1==0 :
            print(i,acc,"advl",aloss,"contentl",closs)
        if acc == 0 and np.all(np.less_equal(closs,2*128)):
            break
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


class np_dictionary():
    def __init__(self, attrs, data=None):
        self.attrs = attrs
        if data is None:
            data = {}
            for attr in self.attrs:
                data[attr] = None


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
    generated_img, generated_img_adv = stn.transform(content, p=1.05)
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
        if model_name in ["cifar10_nat", "cifar10_adv"]:
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
            normal_loss = classifier.target_loss
            norm_acc = classifier.accuracy
            acc_y = tf.cast(classifier.correct_prediction, tf.float32)
            #classifier._build_model(img, label, reuse=True)
            classifier = build_model_trades(img, label, conf=0.1)
            decode_acc_y = tf.cast(classifier.correct_prediction, tf.float32)
        else:
            assert False
    elif data_set == "imagenet":
        classifier = build_imagenet_model(adv_img_bgr, label, conf=1)
        adv_acc = classifier.accuracy
        adv_acc_y = classifier.acc_y
        adv_acc_y_5 = classifier.acc_y_5
        adv_loss = classifier.target_loss5
        content_bgr = tf.reverse(
            content, axis=[-1])  # switch RGB to BGR
        classifier = build_imagenet_model(content_bgr, label, reuse=True)
        normal_loss = - classifier.target_loss5#classifier.target_loss
        norm_acc = classifier.accuracy
        acc_y = classifier.acc_y
        acc_y_5 = classifier.acc_y_5
        normal_label =classifier.label
        normal_loss = classifier.target_loss5
        #adv_loss = target_loss5
        target_label = tf.placeholder(tf.float32, shape=[None])
        normal_xent = classifier.xent #- tf.nn.sparse_softmax_cross_entropy_with_logits(
            #logits=classifier.logits, labels=target_label)  # classifier.xent
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
    if tid == 2:
        loss = tf.reduce_sum(
            tf.reduce_mean(tf.square(enc_gen_adv - stn.norm_features), axis=[1, 2]))  # enc_gen_adv - stn.norm_features
    else:
        loss = content_loss + tf.reduce_sum(adv_loss * BATCH_SIZE * adv_weight)# style_weight * style_loss

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
    train_op = gradient(tf.train.GradientDescentOptimizer(learning_rate), vars=stn_vars, loss=loss)

    #train_op = tf.train.AdamOptimizer(learning_rate).minimize(
    #    loss, var_list=stn_vars, global_step=global_step)  

    sess.run(tf.global_variables_initializer())
    if data_set == "cifar10":
        if model_name == "cifar10_adv":
            classifier_saver = tf.train.Saver(classifier_vars, max_to_keep=1)
            classifier_saver.restore(sess, settings.config["hardened_model"])
        elif model_name == "cifar10_nat":
            classifier_saver = tf.train.Saver(classifier_vars, max_to_keep=1)
            classifier_saver.restore(sess, settings.config["pretrained_model"])
        elif model_name == "cifar10_trades":
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



    def merge_dict(dict_tot, dict1):
        for k,v in dict1.items():
            if k in dict_tot:
                dict_tot[k] = np.concatenate([dict_tot[k], dict1[k]])
            else:
                dict_tot[k] = dict1[k]
        return dict_tot
    def get_np_dict():
        np_dict = {}
        for sf in ["", "_pgd_pgd_linf"]:
            for i in range(2, 100):
                np_file_path = os.path.join(
                    base_dir_model, "saved_samples%s%d.npy" % (sf,i))
                if os.path.exists(np_file_path):
                    _np_dict = np.load(np_file_path).item()
                    merge_dict(np_dict, _np_dict)
        return np_dict
    def l2_dist(x1,x2,mask=None):
        l=len(x1.shape)
        samples = min(x1.shape[0],x2.shape[0])
        x1 = x1[:samples]
        x2 = x2[:samples]

        diff = np.multiply( (x1-x2), (x1-x2))
        dist = np.sum(diff,axis=tuple(range(1,l)))
        dist = np.sqrt(dist)
        #print(dist)
        if mask is None:
            dist = np.mean(dist)
        else:
            dist = np.sum(dist*mask)/np.sum(mask)
        return dist

    def linf_dist(x1,x2,mask=None):
        l = len(x1.shape)
        samples = min(x1.shape[0],x2.shape[0])
        x1=x1[:samples]
        x2 =x2[:samples]

        diff = np.abs((x1-x2))
        dist = np.max(diff, axis=tuple(range(1, l)))

        #print(dist)
        if mask is None:
            dist = np.mean(dist)
        else:
            dist = np.sum(dist*mask)/np.sum(mask)
        return dist


    def l1_dist(x1, x2,mask=None):
        l = len(x1.shape)
        samples = min(x1.shape[0], x2.shape[0])
        x1 = x1[:samples]
        x2 = x2[:samples]

        diff = np.abs((x1-x2))
        dist = np.sum(diff, axis=tuple(range(1, l)))

        #print(dist)
        if mask is None:
            dist = np.mean(dist)
        else:
            dist = np.sum(dist*mask)/np.sum(mask)
        return dist

    def save_img_gen(a,b):
        for i in range(BATCH_SIZE):
            save_out = np.concatenate(
                (a[i], b[i], np.abs(a[i]-b[i])))
            sz = TRAINING_IMAGE_SHAPE[1]
            full_path = os.path.join(
                "temp", "%d.jpg" % i)
            os.makedirs(os.path.join("temp"), exist_ok=True)
            save_out = np.reshape(save_out, newshape=[sz*3, sz, 3])
            save_rgb_img(save_out, path=full_path)

    # 0 evaluate the statistics
    # 1 evaluate pgd l_inf attack

    if tid == 0:
        np_dict = get_np_dict()
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
        l_inf_adv = np_dict["adv_image_pgd_linf"]
        l_inf_acc = np_dict["acc_y_pgd_linf"]
        l_inf_acc_5 = np_dict["acc_y_5_pgd_linf"]
        print("l2 dist adv", l2_dist(np_adv_image_arr, np_benign_image_arr))
        print("l2 dist pgd", l2_dist(l_inf_adv, np_benign_image_arr))
        print("linf dist adv", linf_dist(np_adv_image_arr, np_benign_image_arr))
        print("linf dist pgd", linf_dist(l_inf_adv, np_benign_image_arr))
        print("l1 dist adv", l1_dist(np_adv_image_arr, np_benign_image_arr))
        print("l1 dist pgd", l1_dist(l_inf_adv, np_benign_image_arr))

        print("acc_5", np.mean(np_acc_5_arr))
        print("acc_1", np.mean(np_acc_arr))
        print("adv_acc_5", np.mean(np_acc_attack_5_arr))
        print("adv_acc", np.mean(np_acc_attack_arr))
        print("decode_acc_5", np.mean(np_decode_acc_5_arr))
        print("decode_acc", np.mean(np_decode_acc_arr))
        print("linf acc", np.mean(l_inf_acc))
        print("linf acc5", np.mean(l_inf_acc_5))
    elif tid==1:

        for attack_method in ["linf","l2"]:
            if attack_method == "linf":
                pgd_attack = LinfPGDAttack(normal_loss, content, label,
                                    epsilon=8.0, num_steps=50, step_size=2.0, random_start=True)
                atk_func = pgd_attack.perturb
                suffix = "_pgd_linf"
            elif attack_method == "l2":
                pgd_attack = LinfPGDAttack(normal_loss, content, label,
                                    epsilon=7.0*255, num_steps=50, step_size=64, random_start=True) # 255 normally
                atk_func = pgd_attack.perturb_l2
                suffix = "_pgd_l2"

            assert len(suffix)>2
            for i in range(1, 100):
                np_file_path = os.path.join(
                    base_dir_model, "saved_samples%d.npy" % i)
                np_new_path = os.path.join(
                    base_dir_model, "saved_samples_pgd%s%d.npy" % (suffix,i))
                if os.path.exists(np_file_path) and not os.path.exists(np_new_path):
                    np_dict = np.load(np_file_path).item()
                    
                    np_benign_image_arr = np_dict["benign_image"]
                    np_label_arr = np_dict["label"]
                    np_adv_arr = np_dict["adv_image"]
                    num = np_benign_image_arr.shape[0]
                    bs = num//BATCH_SIZE
                    np_acc_y = []
                    np_acc_y_5 = []
                    np_arr_img = []

                    for j in range(bs):
                        _x_batch = np_benign_image_arr[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
                        _y_batch = np_label_arr[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
                        _adv_batch = np_adv_arr[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
                        #_x_batch, _y_batch = get_data(sess)
                        #_target_label = sess.run(normal_label, feed_dict={
                        #        content: np_adv_arr, label: _y_batch})
                        perturbed_x = atk_func(_x_batch, _y_batch,sess)
                        if data_set == "imagenet":
                            _acc_y, _acc_y_5 = sess.run([acc_y, acc_y_5], feed_dict={content:perturbed_x, label:_y_batch})
                            np_acc_y.append(_acc_y)
                            np_acc_y_5.append(_acc_y_5)
                            np_arr_img.append(perturbed_x)
                            print(_acc_y_5)
                            print(_acc_y)
                            _np_acc_y= np.concatenate(np_acc_y)
                            _np_acc_y_5 = np.concatenate(np_acc_y_5)
                            _np_arr_img = np.concatenate(np_arr_img)
                            
                            adv_np_dict = {"acc_y" + suffix : _np_acc_y,
                                        "acc_y_5" + suffix : _np_acc_y_5,
                                        "adv_image" + suffix : _np_arr_img,
                            }
                        elif data_set == "cifar10":
                            _acc_y = sess.run([acc_y], feed_dict={
                                                        content: perturbed_x, label: _y_batch})
                            np_acc_y.append(_acc_y)
                            np_arr_img.append(perturbed_x)
                            print(_acc_y)
                            _np_acc_y = np.concatenate(np_acc_y)
                            _np_arr_img = np.concatenate(np_arr_img)

                            adv_np_dict = {"acc_y" + suffix: _np_acc_y,
                                           "adv_image" + suffix: _np_arr_img,
                                           }
                        np.save(np_new_path,adv_np_dict)

    elif tid==2:
        np_dict = get_np_dict()

        benign_image = np_dict["benign_image"]
        adv_image = np_dict["adv_image"]
        pgd_linf_image = np_dict["adv_image_pgd_linf"]
        label_arr = np_dict["label"]

        l_inf_acc = np_dict["acc_y_pgd_linf"]
        np_adv_acc = np_dict["acc_attack"]

        decoder_image = []
        sp = min(benign_image.shape[0],
                 adv_image.shape[0], 
                 pgd_linf_image.shape[0])
        bs = sp // BATCH_SIZE  
        sp = bs * BATCH_SIZE
        benign_image = benign_image[:sp]
        adv_image = adv_image[:sp]
        label_arr = label_arr[:sp]
        pgd_linf_image = pgd_linf_image[:sp]

        assert benign_image.shape[0] == adv_image.shape[0] and benign_image.shape[0] == pgd_linf_image.shape[0]

              
        TRAIN_DECODER = False
        #np_label_arr = np_dict["label"]
        embeds = []
        for i in range(bs):
            bt_benign_image = benign_image[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            bt_adv_image = adv_image[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            bt_pgd_image = pgd_linf_image[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            bt_label = label_arr[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            fd_benign = {content: bt_benign_image, label:bt_label}
            fd_adv = {content: bt_adv_image}           
            fd_pgd = {content: bt_pgd_image}           

            if TRAIN_DECODER:
                sess.run(stn.init_style, feed_dict=fd_benign)
                sess.run(global_step.initializer)
                for i in range(10):
                    #_,  acc, aloss, closs, closs1, sigma, mean, sigmaS, meanS = sess.run(
                    #    [train_op,  adv_acc, adv_loss, content_loss_y, content_loss, stn.sigmaC, stn.meanC, stn.sigmaS, stn.meanS], feed_dict=fdict)
                    _, acc, aloss, closs, = sess.run(
                        [train_op,  adv_acc, adv_loss, content_loss_y],  feed_dict=fd_benign)
                    sess.run(stn.style_bound, feed_dict=fd_benign)
                print(i, acc, "advl", aloss, "contentl", closs)
                bt_decoder_image, embed_benign = sess.run([adv_img, stn.normalized], feed_dict=fd_benign)
            else:
                bt_decoder_image, embed_benign = sess.run([img, stn.normalized ], feed_dict=fd_benign)
            save_img_gen(bt_decoder_image, bt_benign_image)
            decoder_image.append(bt_decoder_image)
            fd_decoder = {content: bt_decoder_image}

            embed_adv = sess.run(stn.normalized, feed_dict=fd_adv)
            embed_decoder = sess.run(stn.normalized, feed_dict=fd_decoder)
            embed_pgd = sess.run(stn.normalized, feed_dict=fd_pgd)

            embeds.append([embed_benign,embed_adv,embed_decoder,embed_pgd])
        
        decoder_image = np.concatenate(decoder_image)
        rsts = []
        for j in range(4):
            rst = []
            for embed in embeds:
                rst.append(embed[j])
            rsts.append(np.concatenate(rst))

        print("l2 dist decoder", l2_dist(
            decoder_image, benign_image)/255)
        print("linf dist decoder", linf_dist(
            decoder_image, benign_image)/255)
        print("l1 dist decoder", l1_dist(
            decoder_image, benign_image)/255)

        title = ["benign", "adv", "decoder","pgd"]
        
        l_inf_acc = np.concatenate(l_inf_acc)
        #np_adv_acc = np.concatenate(np_adv_acc)
        masks = [None, 1-np_adv_acc, 1-l_inf_acc, None]
        dist_desc = ["l2","l1","linf"]
        func = [l2_dist, l1_dist, linf_dist]
        
        for rs1 in range(4):
            for rs2 in range(rs1,4):
                print("Results between %s and %s:"%(title[rs1],title[rs2]))
                for df,dd in zip(func,dist_desc):
                    print("%s distance is %f"%(dd,df(rsts[rs1],rsts[rs2],masks[rs2])))

    elif tid==3:
        np_dict = get_np_dict()
        benign_image = np_dict["benign_image"]
        adv_image = np_dict["adv_image"]
        pgd_linf_image = np_dict["adv_image_pgd_linf"]
        label_arr = np_dict["label"]
        decoder_image = []
        sp = min(benign_image.shape[0],
                 adv_image.shape[0],
                 pgd_linf_image.shape[0])
        bs = sp // BATCH_SIZE
        sp = bs * BATCH_SIZE
        benign_image = benign_image[:sp]
        adv_image = adv_image[:sp]
        label_arr = label_arr[:sp]
        pgd_linf_image = pgd_linf_image[:sp]
        for j in range(16):
            for i in range(BATCH_SIZE):
                print(j+1,i+1,label_arr[j*BATCH_SIZE + i])
            
    ###### Done Training & Save the model ######
    #saver.save(sess, model_save_path)

    if debug:
        elapsed_time = datetime.now() - start_time
        print('Done training! Elapsed time: %s' % elapsed_time)
        #print('Model is saved to: %s' % model_save_path)

