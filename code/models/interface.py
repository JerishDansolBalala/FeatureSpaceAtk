import tensorflow as tf
from . import resnet_slim
slim = tf.contrib.slim

def get_scope_var(scope_name):
    var_list = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)
    assert (len(var_list) >= 1)
    return var_list

def restore_parameter(sess):
    file_path = "imagenet_resnet_v1_50.ckpt"
    var_list = get_scope_var("resnet_v1")
    saver = tf.train.Saver(var_list)
    saver.restore(sess,file_path)



class container:
    def __init__(self):
        pass


def compute_loss_and_error(logits, label, label_smoothing=0.):
    if label_smoothing == 0.:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=label)
    else:
        nclass = logits.shape[-1]
        loss = tf.losses.softmax_cross_entropy(
            tf.one_hot(label, nclass),
            logits, label_smoothing=label_smoothing,
            reduction=tf.losses.Reduction.NONE)
    loss = tf.reduce_mean(loss, name='xentropy-loss')

    def prediction_incorrect(logits, label, topk=1, name='incorrect_vector'):
        with tf.name_scope('prediction_incorrect'):
            x = tf.logical_not(tf.nn.in_top_k(logits, label, topk))
        return tf.cast(x, tf.float32, name=name)

    wrong_1 = prediction_incorrect(logits, label, 1, name='wrong-top1')

    wrong_5 = prediction_incorrect(logits, label, 5, name='wrong-top5')
    return loss, wrong_1, wrong_5

def build_imagenet_model(image, label, reuse=False, conf=1, shrink_class = 1000):
    cont = container()

    with slim.arg_scope(resnet_slim.resnet_arg_scope()):
        logits, desc = resnet_slim.resnet_v1_50(image, num_classes=shrink_class, is_training= False, reuse=reuse)
    loss, wrong_1, wrong_5 = compute_loss_and_error(logits,label,)
    cont.logits = logits
    cont.label = tf.argmax(cont.logits, axis=-1)
    cont.acc_y = 1-wrong_1
    cont.acc_y_5 = 1-wrong_5
    cont.accuracy = tf.reduce_mean(1-wrong_1)  # wrong_5
    cont.rev_xent = tf.reduce_mean(tf.log(
        1 - tf.reduce_sum(tf.nn.softmax(logits) *
                          tf.one_hot(label, depth=shrink_class), axis=-1)
    ))
    cont.poss_loss = 1 - tf.reduce_mean(
        tf.reduce_sum(tf.nn.softmax(logits) *
                      tf.one_hot(label, depth=shrink_class), axis=-1)
    )

    label_one_hot = tf.one_hot(label, depth=shrink_class)
    wrong_logit = tf.reduce_max(
        logits * (1-label_one_hot) - label_one_hot * 1e7, axis=-1)
    true_logit = tf.reduce_sum(logits * label_one_hot, axis=-1)
    cont.target_loss = - tf.nn.relu(true_logit - wrong_logit + conf)

    cont.xent_filter = tf.reduce_mean((1.0-wrong_1) *
                                           tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits), axis=-1)

    cont.xent = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=label, logits=logits), axis=-1)
    return cont
