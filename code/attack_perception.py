# Train the Style Transfer Net

from __future__ import print_function

import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import tensorflow as tf
import os
import settings

task_name = "imperceptability"
data_set = "imagenet"  # "imagenet"
model_name = "imagenet_normal"
decoder_name = "imagenet_shallowest"

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
    if decoder_name == "imagenet_shallowest_smooth":
        LEARNING_RATE = 1e-3
    else:
        LEARNING_RATE = 1e-2
    LR_DECAY_RATE = 0 # 5e-5
    DECAY_STEPS = 1.0
    adv_weight = 10
    ITER=1000
    CLIP_NORM_VALUE = 1000.0

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
        [adv_img, content_loss_y, adv_acc_y_5, stn.meanS, stn.sigmaS],  feed_dict=fdict)
    _ITER = ITER
    flag=True
    last_update= [0 for i in range(BATCH_SIZE)]
    min_val = [1e10 for i in range(BATCH_SIZE)]
    for i in range(_ITER):
        #_,  acc, aloss, closs, closs1, sigma, mean, sigmaS, meanS = sess.run(
        #    [train_op,  adv_acc, adv_loss, content_loss_y, content_loss, stn.sigmaC, stn.meanC, stn.sigmaS, stn.meanS], feed_dict=fdict)
        _, _l2_g = sess.run([train_op, l2_norm_g],  feed_dict=fdict)
        sess.run(stn.style_bound, feed_dict = fdict)
        flag1 = True
        _adv_img, acc, aloss, closs, _mean, _sigma = sess.run( [adv_img, adv_acc_y_5, adv_loss, content_loss_y, stn.meanS, stn.sigmaS],  feed_dict = fdict)
        ups=[]
        for j in range(BATCH_SIZE):
            if  aloss[j]<=min_val[j]*0.95:
                min_val[j] = aloss[j]
                last_update[j] = i

            if acc[j]<rst_acc[j] or (acc[j]==rst_acc[j] and closs[j]<rst_loss[j]):
                rst_img[j]=_adv_img[j]
                rst_acc[j] = acc[j]
                rst_loss[j] = closs[j]
                rst_mean[j] = _mean[j]
                rst_sigma[j] = _sigma[j]
                last_update[j] = i

            if i-last_update[j]<=200:
                flag1 = False
                #ups.append(stn.init_style_rand[j])
                #print("\treset %d"%j,end="\t")
                #last_update[j] = i
        if flag1:
            break

        if len(ups)>0:
            sess.run(ups,feed_dict=fdict)
        if i>_ITER:
            break
        if flag and np.mean(acc)==0:
            _ITER=i+500
            flag=False
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
            
            print(i, acc, "advl", aloss, "contentl", closs, "norm", _l2_g)
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
    global l2_norm_g
    gradients, variables = zip(*opt.compute_gradients(loss,vars))
    l2_norm_g = tf.norm(gradients)
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

    mgt = tf.get_variable(dtype=tf.float32, shape =[], name="magnititude")
    mgt_ph = tf.placeholder(tf.float32, shape= [])
    mgt_asgn = tf.assign (mgt,mgt_ph)
    # create the style transfer net
    stn = StyleTransferNet_adv(encoder_path)

    # pass content and style to the stn, getting the generated_img
    generated_img , generated_img_adv = stn.transform(content, p=mgt)
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
        adv_loss = - classifier.target_loss
        adv_acc = classifier.accuracy
        classifier._build_model(img, label, reuse=True)
        normal_loss = - classifier.target_loss
        norm_acc = classifier.accuracy
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


    # compute the content loss
    bar=3000/64/128
    content_loss_y = tf.reduce_sum(
        tf.reduce_mean(tf.square(enc_gen_adv - target_features), axis=[1, 2]),axis=-1)
    content_loss = tf.reduce_sum(tf.reduce_mean(
        tf.square(enc_gen_adv - target_features), axis=[1, 2]))
    #
    #content_loss += tf.reduce_sum(tf.reduce_mean(
    #    tf.square(enc_gen - stn.norm_features), axis=[1, 2]))

    # compute the style loss
    
    style_layer_loss = []

    # compute the total loss
    # adv_loss * adv_weight
    loss = tf.reduce_sum((1-adv_acc_y_5) * content_loss_y)
    loss += tf.reduce_sum(adv_loss * BATCH_SIZE * adv_weight)# style_weight * style_loss

    l2_embed = normalize(enc_gen)[0] - normalize(stn.norm_features)[0]
    l2_embed = tf.reduce_mean(
        tf.sqrt(tf.reduce_sum((l2_embed * l2_embed), axis=[1, 2, 3])))


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



    uid = 0

    data_set = "imagenet"  # "imagenet"
    model_name = "imagenet_normal_backup"
    decoder_name = "imagenet_shallowest_smooth"
    base_dir_model_old = base_dir_model
    base_dir_model = os.path.join(
         "store", data_set, decoder_name, model_name)


    def merge_dict(dict_tot, dict1):
        for k, v in dict1.items():
            if k in dict_tot:
                dict_tot[k] = np.concatenate([dict_tot[k], dict1[k]])
            else:
                dict_tot[k] = dict1[k]
        return dict_tot


    def get_np_dict():
        np_dict = {}
        for sf in ["", "_pgd_pgd_linf"]:
            for i in range(1, 100):
                np_file_path = os.path.join(
                    base_dir_model, "saved_samples%s%d.npy" % (sf, i))  # "target_attack",
                if os.path.exists(np_file_path):
                    _np_dict = np.load(np_file_path, allow_pickle=True).item()
                    merge_dict(np_dict, _np_dict)
        return np_dict


    class np_dictionary():
        def __init__(self, attrs, data=None):
            self.attrs = attrs
            if data is None:
                data = {}
                for attr in self.attrs:
                    data[attr] = None


    tot_succ = 0
    cnt = 50
    #for i, (im, file_name) in enumerate(dataset_loader):
    np_dict = get_np_dict()
    print(np_dict.keys())
    np_label_arr_tot = np_dict["label"]
    np_benign_image_arr_tot = np_dict["benign_image"]
    idx = np_dict["index"]

    base_dir_model = base_dir_model_old
    report_batch = 2
    assert len(np_benign_image_arr_tot) >= 21*8*8
    for batch in range(1,8+1):
        #x_batch, y_batch = get_data(sess)
        x_batch = np_benign_image_arr_tot[21*8*(batch-1):21*8*(batch-1)+8]
        y_batch = np_label_arr_tot[21*8*(batch-1):21*8*(batch-1)+8]

        fdict = {content: x_batch, label: y_batch}

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
            np_mgt = []
            np_index = []

        start=1.0
        end=2.0
        divides=40
        for j in range(0,25,2):
            mgt_val = (start*(divides-j)+end*j)/divides
            sess.run(mgt_asgn, feed_dict={mgt_ph:mgt_val})

        # run the training step
        

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
                _adv_img, _loss_y, _adv_acc_y, _adv_acc_y_5, _acc_y, _acc_y_5, _decode_acc_y, _decode_acc_y_5 = sess.run([
                    adv_img, content_loss_y, adv_acc_y, adv_acc_y_5, acc_y, acc_y_5, decode_acc_y, decode_acc_y_5], feed_dict=fdict)
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
                np_mgt . append(BATCH_SIZE*[mgt_val])
                np_index.append([batch*BATCH_SIZE+k for k in range(BATCH_SIZE)])

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
            np_mgt_arr = np.concatenate(np_mgt)
            np_index_arr = np.concatenate(np_index)

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
                        "magnititude": np_mgt_arr,
                        "index":np_index_arr}

            np.save(os.path.join(base_dir_model, "saved_samples%d.npy" %
                                 (batch//report_batch)), saved_dict)

    ###### Done Training & Save the model ######
    #saver.save(sess, model_save_path)

    if debug:
        elapsed_time = datetime.now() - start_time
        print('Done training! Elapsed time: %s' % elapsed_time)
        #print('Model is saved to: %s' % model_save_path)

