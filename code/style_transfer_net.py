# Style Transfer Network
# Encoder -> AdaIN -> Decoder

import tensorflow as tf

from encoder import Encoder
from decoder import Decoder
from adaptive_instance_norm import AdaIN, AdaIN_adv, normalize, AdaIN_rand
import settings

class StyleTransferNet(object):

    def __init__(self, encoder_weights_path):
        self.encoder = Encoder(encoder_weights_path)
        self.decoder = Decoder()

    def deocde_to_real(self, img):
        # deprocess image
        generated_img = self.encoder.deprocess(img)

        # switch BGR back to RGB
        generated_img = tf.reverse(generated_img, axis=[-1])

        # clip to 0..255
        generated_img = tf.clip_by_value(generated_img, 0.0, 255.0)
        return generated_img

    def transform(self, content, style):
        # switch RGB to BGR
        content = tf.reverse(content, axis=[-1])
        style   = tf.reverse(style,   axis=[-1])

        # preprocess image
        content = self.encoder.preprocess(content)
        style   = self.encoder.preprocess(style)

        # encode image
        enc_c, enc_c_layers = self.encoder.encode(content)
        enc_s, enc_s_layers = self.encoder.encode(style)

        self.encoded_content_layers = enc_c_layers
        self.encoded_style_layers   = enc_s_layers

        self.norm_features = enc_c
        # pass the encoded images to AdaIN
        if "TRAIN_SMOOTH_ENABLE" in settings.config:
            Batch_SZ = settings.config["BATCH_SIZE"]
            self.mix_rate_content = tf.random.uniform([Batch_SZ,1,1,1], minval=0, maxval=1)
            with tf.variable_scope("transform"):
                target_features = AdaIN_rand(
                    enc_c, enc_s, self.mix_rate_content)
        else:
            with tf.variable_scope("transform"):
                target_features = AdaIN(enc_c, enc_s)
        self.target_features = target_features
        #self.target_features = target_features

        # decode target features back to image
        generated_adv_img = self.deocde_to_real(self.decoder.decode(target_features))
        generated_img = self.deocde_to_real(self.decoder.decode(enc_c))

        return generated_img, generated_adv_img



class StyleTransferNet_adv(object):

    def __init__(self, encoder_weights_path):
        self.encoder = Encoder(encoder_weights_path)
        self.decoder = Decoder()

    def deocde_to_real(self, img):
        # deprocess image
        generated_img = self.encoder.deprocess(img)

        # switch BGR back to RGB
        generated_img = tf.reverse(generated_img, axis=[-1])

        # clip to 0..255
        generated_img = tf.clip_by_value(generated_img, 0.0, 255.0)
        return generated_img

    def transform(self, content , p=1.5):
        # switch RGB to BGR
        content = tf.reverse(content, axis=[-1])
        #style = tf.reverse(style,   axis=[-1])

        # preprocess image
        content = self.encoder.preprocess(content)
        #style = self.encoder.preprocess(style)

        # encode image
        enc_c, enc_c_layers = self.encoder.encode(content)
        #enc_s, enc_s_layers = self.encoder.encode(style)

        self.encoded_content_layers = enc_c_layers
        #self.encoded_style_layers = enc_s_layers

        self.norm_features = enc_c
        # pass the encoded images to AdaIN
        with tf.variable_scope("transform"):
            target_features, self.init_style, self.style_bound, self.sigmaS, self.meanS, self.meanC, self.sigmaC, self.init_style_rand, self.normalized = AdaIN_adv(enc_c, p=p)
        bs = settings.config["BATCH_SIZE"]
        self.meanS_ph= tf.placeholder(tf.float32, [bs]+ self.meanS.shape.as_list()[1:])
        self.sigmaS_ph = tf.placeholder(
            tf.float32, [bs] + self.sigmaS.shape.as_list()[1:])
        self.asgn = [tf.assign(self.meanS, self.meanS_ph),
                     tf.assign(self.sigmaS, self.sigmaS_ph)]
        self.target_features = target_features
        #self.target_features = target_features

        # decode target features back to image
        generated_adv_img = self.deocde_to_real(
            self.decoder.decode(target_features))
        generated_img = self.deocde_to_real(self.decoder.decode(enc_c))

        return generated_img, generated_adv_img

    def transform_from_internal(self, content, store_var, sigma, mean):
        content = tf.reverse(content, axis=[-1])
        #style = tf.reverse(style,   axis=[-1])

        # preprocess image
        content = self.encoder.preprocess(content)
        #style = self.encoder.preprocess(style)

        # encode image
        enc_c, enc_c_layers = self.encoder.encode(content)

        self.normalized, self.meanC, self.sigmaC = normalize(enc_c)

        self.store_normalize = tf.assign(store_var, self.normalized)
        self.restored_internal = store_var * sigma + mean
        self.target_features = self.restored_internal

        self.loss_l1 = tf.reduce_sum(tf.abs(enc_c - self.target_features))
        
        generated_adv_img = self.deocde_to_real(
            self.decoder.decode(self.target_features))
        generated_img = self.deocde_to_real(self.decoder.decode(enc_c))

        return generated_img, generated_adv_img

    def transform_from_internal_poly(self, content):
        content = tf.reverse(content, axis=[-1])
        #style = tf.reverse(style,   axis=[-1])

        # preprocess image
        content = self.encoder.preprocess(content)
        #style = self.encoder.preprocess(style)

        # encode image
        enc_c, enc_c_layers = self.encoder.encode(content)
        self.normalized, self.meanC, self.sigmaC = normalize(enc_c)

        INTERPOLATE_NUM = settings.config["INTERPOLATE_NUM"]
        BATCH_SIZE = settings.config["BATCH_SIZE"]
        DIM = settings.config["DECODER_DIM"]
        STORE_SHAPE = [BATCH_SIZE] + DIM

        self.store_var = tf.Variable(
            tf.zeros(STORE_SHAPE), dtype=tf.float32, trainable=False)
        self.internal_sigma = tf.placeholder(
            tf.float32, shape=(BATCH_SIZE, INTERPOLATE_NUM, 1, 1, DIM[2]))
        self.internal_mean = tf.placeholder(tf.float32, shape=(
            BATCH_SIZE, INTERPOLATE_NUM, 1, 1, DIM[2]))

        self.coef_ph = tf.placeholder(tf.float32, shape=[BATCH_SIZE, INTERPOLATE_NUM])
        
        with tf.variable_scope("transform"):
            self.coef = tf.get_variable("coef", shape=[BATCH_SIZE, INTERPOLATE_NUM],
                                        initializer=tf.ones_initializer())
            self.coef_asgn = tf.assign(self.coef, self.coef_ph)
            method = "relu"
            if method == "relu":
                postive_coef = tf.nn.relu(self.coef)
                sum_coef = tf.reduce_sum(postive_coef, axis=1, keepdims=True)
                coef_poss = postive_coef / (sum_coef + 1e-7)
                self.regulate = tf.assign(self.coef, coef_poss)
                coef_poss = tf.reshape(
                    coef_poss, shape=[BATCH_SIZE, INTERPOLATE_NUM, 1, 1, 1])
            elif method =="softmax":
                coef = self.coef * 2 # control the gradient not to be too large
                coef_poss = tf.nn.softmax(coef, axis=-1)
                coef_poss = tf.reshape(coef_poss, shape=[BATCH_SIZE, INTERPOLATE_NUM, 1, 1, 1])
                self.regulate = []

        self.store_normalize = [tf.assign(self.store_var, self.normalized), self.coef.initializer]
        self.sigma_poly = tf.reduce_sum(self.internal_sigma*coef_poss, axis=1)
        self.mean_poly = tf.reduce_sum(self.internal_mean*coef_poss, axis=1)
        self.restored_internal = self.store_var * self.sigma_poly + self.mean_poly
        self.target_features = self.restored_internal

        self.loss_l1 = tf.reduce_sum(tf.abs(enc_c - self.target_features))

        generated_adv_img = self.deocde_to_real(
            self.decoder.decode(self.target_features))
        generated_img = self.deocde_to_real(self.decoder.decode(enc_c))

        return generated_img, generated_adv_img
