# Train the Style Transfer Net


from __future__ import print_function

import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import tensorflow as tf
import os
import settings
import json
task_name = "attack"

"""
data_set = "cifar10"  # "imagenet"
model_name = "cifar10_adv"
decoder_name = "cifar10_balance"
"""
data_set = "imagenet"  # "imagenet"
model_name = "imagenet_denoise"
#decoder_name = "imagenet"



BATCH_SIZE=8
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





def save_rgb_img( img, path):
    img = img.astype(np.uint8)
    #img=np.reshape(img,[28,28])
    Image.fromarray(img, mode='RGB').save(path)


def get_scope_var(scope_name):
    var_list = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)
    assert (len(var_list) >= 1)
    return var_list



#model_save_path= "./transform.ckpt"
debug=True
logging_period=100
if debug:
    from datetime import datetime
    start_time = datetime.now()

def grad_attack_selfaug():
    sess.run(_d["init_style"], feed_dict=fdict)
    sess.run(global_step.initializer)
    rst_img, rst_loss, rst_acc,rst_mean,rst_sigma = sess.run(
        [adv_img, content_loss_y, adv_acc_y, meanS, sigmaS],  feed_dict=fdict)
    
    for i in range(ITER):
        #_,  acc, aloss, closs, closs1, sigma, mean, sigmaS, meanS = sess.run(
        #    [train_op,  adv_acc, adv_loss, content_loss_y, content_loss, stn.sigmaC, stn.meanC, stn.sigmaS, stn.meanS], feed_dict=fdict)
        _ = sess.run([train_op],  feed_dict=fdict)
        sess.run(stn.style_bound, feed_dict = fdict)
        _adv_img, acc, aloss, closs, _mean, _sigma = sess.run(
            [adv_img, adv_acc_y, adv_loss, content_loss_y, meanS, sigmaS],  feed_dict=fdict)
        for j in range(BATCH_SIZE):
            if acc[j]<rst_acc[j] or (acc[j]==rst_acc[j] and closs[j]<rst_loss[j]):
                rst_img[j]=_adv_img[j]
                rst_acc[j] = acc[j]
                rst_loss[j] = closs[j]
                rst_mean[j] = _mean[j]
                rst_sigma[j] = _sigma[j]

        if i%50==0 :
            acc=np.mean(acc)
            closs=np.mean(closs)
            print(i,acc,"advl",aloss,"contentl",closs)
    sess.run(asgn, feed_dict={meanS_ph: rst_mean, sigmaS_ph: rst_sigma})
    #return rst_img


def grad_attack_polygon():
    sess.run(store_normalize, feed_dict=fdict)
    sess.run(global_step.initializer)
    sess.run(tf.variables_initializer(train_obj.variables()))
    sess.run(regulate)
    rst_acc, rst_loss, rst_coef = sess.run(
        [adv_acc_y, content_loss_y, coef],  feed_dict=fdict)
    for i in range(ITER):
        #_,  acc, aloss, closs, closs1, sigma, mean, sigmaS, meanS = sess.run(
        #    [train_op,  adv_acc, adv_loss, content_loss_y, content_loss, stn.sigmaC, stn.meanC, stn.sigmaS, stn.meanS], feed_dict=fdict)
        _, acc, aloss, closs, _coef = sess.run(
            [train_op,  adv_acc_y, adv_loss, content_loss_y , coef],  feed_dict=fdict)
        sess.run(stn.regulate)

        for j in range(BATCH_SIZE):
            if acc[j] < rst_acc[j] or (acc[j] == rst_acc[j] and closs[j] < rst_loss[j]):

                rst_acc[j] = acc[j]
                rst_loss[j] = closs[j]
                rst_coef[j] = _coef[j]

        
        if i % 50 == 0:
            acc = np.mean(acc)
            closs = np.mean(closs)
            print(i, acc, "advl", aloss, "contentl", closs)

    sess.run(coef_asgn, feed_dict={coef_ph: rst_coef})
    #return rst_img

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



def gradient(opt, vars, loss ):
    global update_mean, update_sigma
    assert vars[0].name.find("mean_S") >= 0 and vars[1].name.find(
        "sigma_S") >= 0 , vars

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
    gradients = []
    variables_r = []
    if update_mean:
        gradients.append(tf.stack(g1_list, axis=0))
        variables_r.append(variables[0])
    if update_sigma:
        gradients.append(tf.stack(g2_list, axis=0))
        variables_r.append(variables[1])
    #gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    opt = opt.apply_gradients(zip(gradients, variables_r), global_step=global_step)
    return opt

def gradient1(opt, vars, loss ):

    gradients, variables = zip(*opt.compute_gradients(loss,vars))
    g_split = [tf.unstack(g, BATCH_SIZE, axis=0) for g in gradients]
    assert (len(g_split)==1)
    g1_list=[]
    DIM = settings.config["DECODER_DIM"][-1]
    limit = 10/np.sqrt(DIM)    
    for g1 in g_split[0]:
        #(g1, g2), _ = tf.clip_by_global_norm([g1, g2], CLIP_NORM_VALUE)
        g1 = tf.clip_by_value(g1,-1/np.sqrt(limit),1/np.sqrt(limit))
        g1_list.append(g1)
    gradients = []
    variables_r = []
    gradients.append(tf.stack(g1_list, axis=0))
    variables_r.append(variables[0])
    #gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    opt = opt.apply_gradients(zip(gradients, variables_r), global_step=global_step)
    return opt

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
            meanC_pop = _mean_all[y*real_num:(y+1)*real_num]
            res_mean.append(meanC_pop)
            sigmaCi = _sigmaC[i: i+1]
            sigmaC_pop = _sigma_all[y*real_num:(y+1)*real_num]
            res_sigma.append(sigmaC_pop)
    return np.stack(res_mean), np.stack(res_sigma)


prefetch = []
motive_batch = 2

def save_img_gen(lst, prefix = ""):
    global step
    def rs(x):
        nonlocal i, cnt
        cnt += 1
        full_path = os.path.join(
            "motiveimg1",  decoder_name, aname1, "%d" %
            batch, "%s_%d_%d.jpg" % (prefix, i, cnt))
        #print(x.shape)
        x = np.reshape(x, newshape=[sz, sz, 3])
        save_rgb_img(x, path=full_path)

    os.makedirs(os.path.join("motiveimg1",  decoder_name , aname1 , "%d" %
                                        batch), exist_ok=True)
    for i in range(BATCH_SIZE):
        cnt = 0
        sz = TRAINING_IMAGE_SHAPE[1]

        for j in range(len(lst)):
            rs(lst[j][i])



import importlib

Good_Example =[(15,3),(17,3),(17,7), (11,3), (4,1), (4,0),(2,1), (14,3), (14,6), (14,5) , (6,0), (12,4)]

size = 224, 224
import cv2

x_batch = []

for i,j in Good_Example:
    im = cv2.imread(os.path.join("motiveimg",  "imagenet", "nat", "%d" % i 
                                 , "_%d_1.jpg" % j), cv2.IMREAD_COLOR)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)                                 
    x_batch . append(im)

x_batch_s = x_batch + x_batch
x_batch_s = np.stack(x_batch_s)
print(len(x_batch_s))



first_import = True
for decoder_name in ["imagenet_shallowest", "imagenet_shallow", "imagenet"]:

    exec(open('base.py').read())
    tf.reset_default_graph()

    import style_transfer_net
    #if not first_import:
    #    importlib.reload(style_transfer_net)
    StyleTransferNet = style_transfer_net.StyleTransferNet
    StyleTransferNet_adv = style_transfer_net.StyleTransferNet_adv
    from utils import get_train_images
    from cifar10_class import Model
    import cifar10_input
    from PIL import Image
    from adaptive_instance_norm import normalize
    first_import =False
    from pgd_attack import LinfPGDAttack

    INTERPOLATE_NUM = settings.config["INTERPOLATE_NUM"]

    STYLE_LAYERS = settings.config["STYLE_LAYERS"]

    # (height, width, color_channels)
    TRAINING_IMAGE_SHAPE = settings.config["IMAGE_SHAPE"]
    INCLUDE_SELF=True

    # get the traing image shape
    HEIGHT, WIDTH, CHANNELS = TRAINING_IMAGE_SHAPE
    INPUT_SHAPE = (None, HEIGHT, WIDTH, CHANNELS)

    if data_set == "cifar10":
        raw_cifar = cifar10_input.CIFAR10Data("cifar10_data")

        def get_data(sess):
            global subset
            if subset == "eval":
                x_batch, y_batch = raw_cifar.eval_data.get_next_batch(
                    batch_size=BATCH_SIZE, multiple_passes=True)
            elif subset == "train":
                x_batch, y_batch = raw_cifar.train_data.get_next_batch(
                    batch_size=BATCH_SIZE, multiple_passes=True)

            return x_batch, y_batch
    elif data_set == "imagenet":
        inet = imagenet(BATCH_SIZE, "val")

        def get_data(sess):
            x_batch, y_batch = inet.get_next_batch(sess)

            return x_batch, y_batch


    EPSILON = 1e-5
    BATCH_SIZE = settings.config["BATCH_SIZE"]
    if data_set == "cifar10":
        LEARNING_RATE = 1e-2
        LR_DECAY_RATE = 1e-4  # 5e-5
        DECAY_STEPS = 1.0
        adv_weight = 256
        ITER = 100
        CLIP_NORM_VALUE = 10.0
    else:
        if model_name == "imagenet_shallowest":
            LEARNING_RATE = 5e-2
        else:
            LEARNING_RATE = 1e-2
        LR_DECAY_RATE = 1e-3  # 5e-5
        DECAY_STEPS = 1.0
        adv_weight = 128
        ITER = 500
        CLIP_NORM_VALUE = 10.0

    style_weight = 1

    subset = "eval"
    encoder_path = ENCODER_WEIGHTS_PATH
    #if len(prefetch)==0:
    #    for i in range(30):
    #        prefetch.append(get_data(None))
    # create the graph
    tf_config = tf.ConfigProto()
    #tf_config.gpu_options.per_process_gpu_memory_fraction=0.5
    tf_config.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:

        p = tf.Variable(initial_value=0.0)
        p_ph = tf.placeholder(tf.float32,shape=None)
        p_asgn = tf.assign(p,p_ph)
        content = tf.placeholder(tf.float32, shape=INPUT_SHAPE, name='content')
        style = tf.placeholder(tf.float32, shape=INPUT_SHAPE, name='style')
        label = tf.placeholder(tf.int64, shape =None, name="label")
        #style   = tf.placeholder(tf.float32, shape=INPUT_SHAPE, name='style')
        content_bgr = tf.reverse(
                content, axis=[-1])  # switch RGB to BGR    
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.inverse_time_decay(
            LEARNING_RATE, global_step, DECAY_STEPS, LR_DECAY_RATE)

        # create the style transfer net
        stn = StyleTransferNet_adv(encoder_path)

        op_d = {"pgd":{}}
        def switch(name):
            assert name in op_d
            _d=op_d[name]
            for k,v in _d.items():
                code = "global %s" % k
                code += "\r\n" + "%s=_d[\"%s\"]" % (k, k)
                exec(code)


        pgd_attack = None
        # pass content and style to the stn, getting the generated_img
        if data_set =="cifar10":
            classifier = Model("eval", raw_cifar.train_images)
        else:
            pass
        first_use=True

        update_mean = True
        update_sigma = True

        report_batch = 50
        aname1="nat"
        os.makedirs("motiveimg1",exist_ok=True)

        for batch in range(0, motive_batch):
            step = batch
            x_batch = x_batch_s[batch*8:(batch+1)*8]
            out_img = []
            out_img.append(x_batch)
            save_img_gen(out_img)

        for aname1 in ["selfaugall"]:#,"selfaug_mean", "selfaug_sigma", "polygon"]:
            aname = aname1
            if aname == "selfaugall":
                aname = "selfaug"
                update_mean = True
                update_sigma = True
            if aname == "selfaug_mean":
                aname = "selfaug"
                update_mean = True
                update_sigma = False
            elif aname == "selfaug_sigma":
                aname = "selfaug"
                update_mean = False
                update_sigma = True
            _d={}
            with tf.variable_scope(aname1):
                if aname == "selfaug":
                    _d["generated_img"] , _d["generated_img_adv"] = stn.transform(content,p=p)
                    _d["meanS_ph"] = stn.meanS_ph
                    _d["sigmaS_ph"] = stn.sigmaS_ph
                    _d["asgn"] = stn.asgn
                elif aname =="polygon":
                    _d["generated_img"], _d["generated_img_adv"] = stn.transform_from_internal_poly(
                        content)
                    _d["internal_sigma"] = stn.internal_sigma
                    _d["internal_mean"] = stn.internal_mean
                    _d["regulate"] = stn.regulate
                    _d["store_normalize"] = stn.store_normalize
                    _d["coef_asgn"] = stn.coef_asgn
                    _d["coef_ph"] = stn.coef_ph
                    _d["coef"] = stn.coef
                    _d["meanC"] = stn.meanC
                    _d["sigmaC"] = stn.sigmaC

                _d["target_features"] = stn.target_features
                _d["init_style"] = stn.init_style
                _d["style_bound"] = stn.style_bound
                _d["meanS"] = stn.meanS
                _d["sigmaS"] = stn.sigmaS


                _d["adv_img"] = _d["generated_img_adv"]
                _d["img"] = _d["generated_img"]

                _d["stn_vars"] = get_scope_var(aname1+"/"+"transform")
                # get the target feature maps which is the output of AdaI

                # pass the generated_img to the encoder, and use the output compute loss
                _d["generated_img_adv"] = tf.reverse(_d["generated_img_adv"], axis=[-1])  # switch RGB to BGR
                adv_img_bgr = _d["generated_img_adv"]
                _d["generated_img_adv"] = stn.encoder.preprocess(
                    _d["generated_img_adv"])  # preprocess image
                _d["enc_gen_adv"], _d["enc_gen_layers_adv"] = stn.encoder.encode(
                    _d["generated_img_adv"])
        


            if data_set == "cifar10":
                classifier._build_model(
                    _d["adv_img"], label, reuse=not first_use, conf=5)
                _d["adv_acc_y"] = tf.cast(classifier.correct_prediction, tf.float32)
            else:
                classifier=build_imagenet_model(adv_img_bgr, label, conf=5, reuse = not first_use)
                _d["adv_acc_y"] = classifier.acc_y
            first_use=False
            _d["adv_loss"] = - classifier.target_loss
            _d["adv_acc"] = classifier.accuracy
            
            if data_set == "cifar10":
                classifier._build_model(content, label, reuse=True, conf=5)
                _d["acc_y"] = tf.cast(
                    classifier.correct_prediction, tf.float32)
            else:
                classifier=build_imagenet_model(content_bgr, label, conf=5, reuse = not first_use)
                _d["acc_y"] = classifier.acc_y
                _d["prediction"] = classifier.label

            _d["normal_loss"] = - classifier.target_loss
            _d["norm_acc"] = classifier.accuracy


            _d["content_loss_y"] = tf.reduce_sum(
                tf.reduce_mean(tf.square(_d["enc_gen_adv"] - _d["target_features"]), axis=[1, 2]),axis=-1)
            _d["content_loss"] = tf.reduce_sum(
                tf.reduce_mean(tf.square(_d["enc_gen_adv"] - _d["target_features"]), axis=[1, 2]))
            
            _d["loss"] = _d["content_loss"] + tf.reduce_sum(_d["adv_loss"] *
                                        BATCH_SIZE * adv_weight)  # style_weight * style_loss
            if aname == "selfaug":
                _d["train_obj"] = tf.train.AdamOptimizer(
                    learning_rate, beta1=0.5)
                _d["train_op"] = gradient(
                    _d["train_obj"], vars=_d["stn_vars"], loss=_d["loss"])
            else:
                _d["train_obj"] = tf.train.AdamOptimizer(
                    learning_rate, beta1=0.5)
                learning_rate1 = tf.train.inverse_time_decay(
                            1e-1, global_step, 1, 5e-3)
                _d["train_op"] = gradient1(
                    _d["train_obj"], vars=_d["stn_vars"], loss=_d["loss"])

            op_d[aname1] = _d

            if pgd_attack is None:
                if data_set == "cifar10":
                    pgd_attack = LinfPGDAttack(-_d["normal_loss"], content, label,
                                        epsilon=16.0, num_steps=50, step_size=2.0, random_start=True)
                elif data_set == "imagenet":
                    pgd_attack = LinfPGDAttack(-_d["normal_loss"], content, label,
                                            epsilon=16.0, num_steps=50, step_size=2.0, random_start=True)
                atk_func=pgd_attack.perturb
        #content_loss += tf.reduce_sum(tf.reduce_mean(
        #    tf.square(enc_gen - stn.norm_features), axis=[1, 2]))

        # compute the style loss
    
        style_layer_loss = []

        # compute the total loss
        # adv_loss * adv_weight
        #content_loss

        if data_set == "cifar10":
            classifier_vars = get_scope_var("model")
        decoder_vars = get_scope_var("decoder")
        # Training step
        
        
        #tf.train.AdamOptimizer(learning_rate).minimize(  # MomentumOptimizer(learning_rate, momentum=0.9) tf.train.GradientDescentOptimizer(learning_rate)
        #train_op = tf.train.AdamOptimizer(learning_rate).minimize(
        #    loss, var_list=stn_vars, global_step=global_step)

        ##gradient clipping
        

        #train_op = tf.train.AdamOptimizer(learning_rate).minimize(
        #    loss, var_list=stn_vars, global_step=global_step)  

        sess.run(tf.global_variables_initializer())
        if data_set == "cifar10":
            classifier_saver = tf.train.Saver(classifier_vars, max_to_keep=1)
            if model_name=="cifar10_adv":
                classifier_saver.restore(sess, settings.config["hardened_model"])
            elif model_name=="cifar10_nat":
                classifier_saver.restore(sess, settings.config["pretrained_model"])
            else:
                assert False
        elif data_set == "imagenet":
            restore_parameter(sess)
        

        # saver
        saver = tf.train.Saver(decoder_vars, max_to_keep=1)
        saver.restore(sess,Decoder_Model)
        ###### Start Training ######
        
        mean_file = "polygon_mean_%s.npy" % decoder_name
        sigma_file = "polygon_sigma_%s.npy" % decoder_name
        if os.path.exists(mean_file) and os.path.exists(sigma_file):
            _mean_all = np.load(mean_file)
            _sigma_all = np.load(sigma_file)
        else:
            ## Populate polygon point
            if data_set=="cifar10":
                class_num = 10
                dp = datapair(class_num, batch_size=64,
                            stack_num=INTERPOLATE_NUM-1)
            else:
                class_num =1000
                dp = datapair(class_num, batch_size=8, stack_num=INTERPOLATE_NUM-1)
            
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

        if debug:
            elapsed_time = datetime.now() - start_time
            start_time = datetime.now()
            print('\nElapsed time for preprocessing before actually train the model: %s' % elapsed_time)
            print('Now begin to train the model...\n')



        uid = 0

        def additem(name,val):
            if name in rst_dict:
                rst_dict[name].append(val)
            else:
                rst_dict[name]=[val]
        def concatitem():
            itm =  rst_dict.items()
            for k,v in itm:
                rst_dict[k]=np.concatenate(rst_dict[k])

        p_val = 1.50
        
        for aname1 in ["selfaugall"]:#["selfaug_mean",  "polygon", "selfaug_sigma", "pgd", ]:
            
            aname = aname1
            if aname in ["selfaug_mean", "selfaug_sigma", "selfaugall"]:
                aname = "selfaug"
            switch(aname1)
            
            print(aname1)
            print(op_d[aname1])
            report_batch = 50
            for batch in range(0, motive_batch):
                sess.run(p_asgn, feed_dict={p_ph: p_val})  # +0.03*batch
                step=batch
                x_batch=x_batch_s[batch*8:(batch+1)*8]
                y_batch = sess.run(prediction, feed_dict={content:x_batch})
                out_img=[]
        
                if aname == "selfaug":                
                    fdict = {content: x_batch, label: y_batch}
                    grad_attack_selfaug()
                    rst_img = sess.run(adv_img, feed_dict=fdict)

                elif aname == "polygon":
                    fdict={content: x_batch, label: y_batch}
                    _meanC, _sigmaC=sess.run([meanC, sigmaC], feed_dict={
                        content: x_batch})
                    _meanC, _sigmaC=popoulate_data(
                        _meanC, _sigmaC, y_batch, include_self=INCLUDE_SELF)

                    fdict={internal_mean: _meanC, internal_sigma: _sigmaC,
                        label: y_batch, content: x_batch}
                    grad_attack_polygon()
                    rst_img = sess.run(adv_img, feed_dict=fdict)

                elif aname == "pgd":
                    rst_img=atk_func(x_batch, y_batch, sess)

                else:
                    assert False

                out_img.append(rst_img)

                save_img_gen(out_img)
                full_path = os.path.join(
                    "motiveimg1",  decoder_name, aname1, "%d" %
                    batch, "acc.txt")                                
                _acc_y = sess.run(acc_y, feed_dict={content: rst_img, label: y_batch})
                open(full_path, "w").write(json.dumps(_acc_y.tolist()))


            ###### Done Training & Save the model ######
            #saver.save(sess, model_save_path)

            if debug:
                elapsed_time = datetime.now() - start_time
                print('Done training! Elapsed time: %s' % elapsed_time)
                #print('Model is saved to: %s' % model_save_path)


