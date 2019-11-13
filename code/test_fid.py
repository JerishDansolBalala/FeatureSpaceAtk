import tensorflow as tf
import os
import functools
import numpy as np
import time
import settings
from tensorflow.python.ops import array_ops
tfgan = tf.contrib.gan

session = tf.InteractiveSession()
# A smaller BATCH_SIZE reduces GPU memory usage, but at the cost of a slight slowdown
BATCH_SIZE = 64

# Run images through Inception.
inception_images = tf.placeholder(tf.float32, [None, 3, None, None])
activations1 = tf.placeholder(
    tf.float32, [None, None], name='activations1')
activations2 = tf.placeholder(
    tf.float32, [None, None], name='activations2')
fcd = tfgan.eval.frechet_classifier_distance_from_activations(
    activations1, activations2)


def inception_activations(images=inception_images, num_splits=1):
    images = tf.transpose(images, [0, 2, 3, 1])
    size = 299
    images = tf.image.resize_bilinear(images, [size, size])
    generated_images_list = array_ops.split(
        images, num_or_size_splits=num_splits)
    activations = tf.map_fn(
        fn=functools.partial(tfgan.eval.run_inception,
                             output_tensor='pool_3:0'),
        elems=array_ops.stack(generated_images_list),
        parallel_iterations=1,
        back_prop=False,
        swap_memory=True,
        name='RunClassifier')
    activations = array_ops.concat(array_ops.unstack(activations), 0)
    return activations


activations = inception_activations()


def get_inception_activations(inps):
    n_batches = int(np.ceil(float(inps.shape[0]) / BATCH_SIZE))
    act = np.zeros([inps.shape[0], 2048], dtype=np.float32)
    for i in range(n_batches):
        inp = inps[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] / 255. * 2 - 1
        act[i * BATCH_SIZE: i * BATCH_SIZE + min(BATCH_SIZE, inp.shape[0])] = session.run(
            activations, feed_dict={inception_images: inp})
    return act


def activations2distance(act1, act2):
     return session.run(fcd, feed_dict={activations1: act1, activations2: act2})


def get_fid(images1, images2):
    assert(type(images1) == np.ndarray)
    assert(len(images1.shape) == 4)
    assert(images1.shape[1] == 3)
    assert(np.min(images1[0]) >= 0 and np.max(images1[0])
           > 10), 'Image values should be in the range [0, 255]'
    assert(type(images2) == np.ndarray)
    assert(len(images2.shape) == 4)
    assert(images2.shape[1] == 3)
    assert(np.min(images2[0]) >= 0 and np.max(images2[0])
           > 10), 'Image values should be in the range [0, 255]'
    assert(images1.shape ==
           images2.shape), 'The two numpy arrays must have the same shape'
    print('Calculating FID with %i images from each distribution' %
          (images1.shape[0]))
    start_time = time.time()
    act1 = get_inception_activations(images1)
    act2 = get_inception_activations(images2)
    fid = activations2distance(act1, act2)
    print('FID calculation time: %f s' % (time.time() - start_time))
    return fid

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
        for i in range(2, 30):
            np_file_path = os.path.join(
                base_dir_model, "saved_samples%s%d.npy" % (sf, i))
            if os.path.exists(np_file_path):
                _np_dict = np.load(np_file_path).item()
                merge_dict(np_dict, _np_dict)
    return np_dict

def bhwc2bchw(x):
    return np.transpose(x, [0, 3, 1, 2])

if __name__=="__main__":

    tid = 2

    task = "imagenet_denoise_targeted"
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
        base_dir_model = os.path.join(base_dir_model, "target_attack")

    elif task == "madry_cifar10_adv":
        data_set = "cifar10"  # "imagenet"
        model_name = "cifar10_adv"
        decoder_name = "cifar10"
        exec(open("base.py").read())

    elif task == "trades_cifar10_adv":
        data_set = "cifar10"  # "imagenet"
        model_name = "cifar10_trades"
        decoder_name = "cifar10_balance"
        exec(open("base.py").read())

    elif task == "madry_cifar10_adv_balance":
        data_set = "cifar10"  # "imagenet"
        model_name = "cifar10_adv"
        decoder_name = "cifar10_balance"
        exec(open("base.py").read())

    np_dict = get_np_dict()


    benign_image = np_dict["benign_image"]
    adv_image = np_dict["adv_image"]
    pgd_linf_image = np_dict["adv_image_pgd_linf"]
    label_arr = np_dict["label"]

    benign_image = bhwc2bchw(benign_image)
    adv_image = bhwc2bchw(adv_image)
    pgd_linf_image = bhwc2bchw(pgd_linf_image)

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

    d_benign_adv = get_fid(benign_image, adv_image)
    d_benign_pgd = get_fid(benign_image, pgd_linf_image)

    print("benign adv",d_benign_adv)
    print("benign pgd",d_benign_pgd)
