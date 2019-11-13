# Train the Style Transfer Net

from __future__ import print_function

import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import tensorflow as tf
import os
import settings


data_set = "imagenet"#"imagenet"
model_name = "imagenet_denoise"
decoder_name = "imagenet"


"""
data_set = "cifar10"  # "imagenet"
model_name = "cifar10_trades"
decoder_name = "cifar10_balance"
"""

"""
data_set = "cifar10"  # "imagenet"
model_name = "cifar10_nat"
decoder_name = "cifar10"
"""

"""
data_set = "cifar10"  # "imagenet"
model_name = "cifar10_adv"
decoder_name = "cifar10"
"""

"""
data_set = "cifar10"  # "imagenet"
model_name = "cifar10_adv"
decoder_name = "cifar10_balance"
"""

"""
data_set = "imagenet"  # "imagenet"
model_name = "imagenet_normal"
decoder_name = "imagenet_shallow"
"""

task_name = "test"

exec(open("base.py").read())
BATCH_SIZE = settings.config["BATCH_SIZE"]
TRAINING_IMAGE_SHAPE = settings.config["IMAGE_SHAPE"]

from style_transfer_net import StyleTransferNet, StyleTransferNet_adv
from utils import get_train_images
from cifar10_class import Model
import cifar10_input
from PIL import Image
from adaptive_instance_norm import normalize
from pgd_attack import LinfPGDAttack

def l2_dist(x1, x2):
    samples = min(x1.shape[0], x2.shape[0])
    x1 = x1[:samples]
    x2 = x2[:samples]

    diff = np.multiply((x1-x2), (x1-x2))
    dist = np.sum(diff, axis=(1, 2, 3))

    #print(dist)
    dist = np.mean(np.sqrt(dist)) /255
    return dist

def linf_dist(x1, x2):
    samples = min(x1.shape[0], x2.shape[0])
    x1 = x1[:samples]
    x2 = x2[:samples]

    diff = np.abs((x1-x2))
    dist = np.max(diff, axis=(1, 2,3))

    #print(dist)
    dist = np.mean(dist) / 255
    return dist


def l1_dist(x1, x2):
    samples = min(x1.shape[0], x2.shape[0])
    x1 = x1[:samples]
    x2 = x2[:samples]

    diff = np.abs((x1-x2))
    dist = np.sum(diff, axis=(1, 2, 3))

    #print(dist)
    dist = np.mean(dist) / 255
    return dist

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
        for i in range(1, 100):
            np_file_path = os.path.join(
                base_dir_model, "saved_samples%s%d.npy" % (sf, i))  # "target_attack",
            if os.path.exists(np_file_path):
                _np_dict = np.load(np_file_path).item()
                merge_dict(np_dict, _np_dict)
    return np_dict
tid=2

class np_dictionary():
    def __init__(self,attrs,data=None):
        self.attrs=attrs
        if data is None:
            data ={}
            for attr in self.attrs:
                data[attr] = None


def save_rgb_img(img, path):
    img = img.astype(np.uint8)
    #img=np.reshape(img,[28,28])
    Image.fromarray(img, mode='RGB').save(path)

def save_img_gen(a, b, p=1 , prefix=""):

    diff = np.abs(a-b) * p
    diff = np.clip(diff,0,255)
    save_out = np.concatenate(
        (a, b, diff))
    sz = TRAINING_IMAGE_SHAPE[1]
    full_path = os.path.join(
        "temp", prefix ,"%d.jpg" % i)
    os.makedirs(os.path.join("temp", prefix), exist_ok=True)
    save_out = np.reshape(save_out, newshape=[sz*3, sz, 3])
    save_rgb_img(save_out, path=full_path)


def save_img_gen_1(a,s, prefix=""):

    sz = TRAINING_IMAGE_SHAPE[1]
    full_path = os.path.join(
        "temp", prefix, "%s.jpg" % s)
    os.makedirs(os.path.join("temp", prefix), exist_ok=True)
    save_out = np.reshape(a, newshape=[sz, sz, 3])
    save_rgb_img(save_out, path=full_path)

tid=2
if tid ==-1:
    np_dict = get_np_dict()
    print("l2 dist adv", l2_dist(np_dict["adv_image"], np_dict["benign_image"]))
    print("linf dist adv", linf_dist(np_dict["adv_image"], np_dict["benign_image"]))
    print("l1 dist adv", l1_dist(np_dict["adv_image"], np_dict["benign_image"]))

    print("acc_1", np.mean(np_dict["acc"]))
    print("adv_acc", np.mean(np_dict["acc_attack"]))
    print("decode_acc", np.mean(np_dict["decode_acc"]))
    if "adv_image_pgd_linf" in np_dict:
        print("pgd acc", np.mean(np_dict["acc_y_pgd_linf"]))
        print("l2 dist pgd", l2_dist(
            np_dict["adv_image_pgd_linf"], np_dict["benign_image"]))
        print("linf dist pgd", linf_dist(
            np_dict["adv_image_pgd_linf"], np_dict["benign_image"]))
        print("l1 dist pgd", l1_dist(
            np_dict["adv_image_pgd_linf"], np_dict["benign_image"]))
        print("linf acc", np.mean(np_dict["acc_y_pgd_linf"]))
    
if tid == 0 :
    #base_dir_model = os.path.join(base_dir_model, "target_attack")
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
    if "adv_image_pgd_linf" in np_dict:
        l_inf_adv = np_dict["adv_image_pgd_linf"]
        l_inf_acc = np_dict["acc_y_pgd_linf"]
        l_inf_acc_5 = np_dict["acc_y_5_pgd_linf"]
        print("l2 dist pgd", l2_dist(l_inf_adv, np_benign_image_arr))
        print("linf dist pgd", linf_dist(l_inf_adv, np_benign_image_arr))
        print("l1 dist pgd", l1_dist(l_inf_adv, np_benign_image_arr))
        print("linf acc", np.mean(l_inf_acc))
        print("linf acc5", np.mean(l_inf_acc_5))
        
    if "succ_attack" in np_dict:
        print("succ attack", np.mean(np_dict["succ_attack"]))
        if "adv_image_pgd_linf" in np_dict:
            print("succ attack", np.mean(np_dict["succ_attack_pgd_linf"]))

    print("l2 dist adv", l2_dist(np_adv_image_arr, np_benign_image_arr))
    print("linf dist adv", linf_dist(np_adv_image_arr, np_benign_image_arr))
    print("l1 dist adv", l1_dist(np_adv_image_arr, np_benign_image_arr))

    print("acc_5", np.mean(np_acc_5_arr))
    print("acc_1", np.mean(np_acc_arr))
    print("adv_acc_5", np.mean(np_acc_attack_5_arr))
    print("adv_acc", np.mean(np_acc_attack_arr))
    print("decode_acc_5", np.mean(np_decode_acc_5_arr))
    print("decode_acc", np.mean(np_decode_acc_arr))

elif tid==1:
    base_dir_model = os.path.join(base_dir_model, "noise_detect")
    np_dict = get_np_dict()
    detect_normal = np_dict["detect_normal"]
    detect_pgd = np_dict["detection_pgd"]
    detect_adv = np_dict["detection_adv"]

    print("detect pgd",np.mean(detect_pgd))
    print("detect adv",np.mean(detect_adv))
    print("detect normal", np.mean(detect_normal))

elif tid == 2:
    np_dict = get_np_dict()
    benign_image = np_dict["benign_image"]
    adv_image = np_dict["adv_image"]
    pgd_linf_image = np_dict["adv_image_pgd_linf"]
    #success = np_dict["succ_attack_pgd_linf"]
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
    BATCH_SIZE=1024
    for j in range(1):
        for i in range(BATCH_SIZE):
            print(j+1, i+1, label_arr[j*BATCH_SIZE + i])
            #if success[j*BATCH_SIZE + i]:
                #save_img_gen(pgd_linf_image[j*BATCH_SIZE + i],  # benign_image[j*BATCH_SIZE + i],
                #         adv_image[j*BATCH_SIZE + i],
                #         p=2, prefix=str(j))
            save_img_gen_1(pgd_linf_image[j*BATCH_SIZE + i],"%d_%s"%(j*BATCH_SIZE + i,"pgd"),"1")
            save_img_gen_1(adv_image[j*BATCH_SIZE + i],"%d_%s"%(j*BATCH_SIZE + i,"fs"),"1")
            save_img_gen_1(benign_image[j*BATCH_SIZE + i],"%d_%s"%(j*BATCH_SIZE + i,"n"),"1")
            #print("linf:", linf_dist(
            #    benign_image[j*BATCH_SIZE + i:j*BATCH_SIZE + i+1], adv_image[j*BATCH_SIZE + i:j*BATCH_SIZE + i+1]))
            #print("l2:", l2_dist(
            #    benign_image[j*BATCH_SIZE + i:j*BATCH_SIZE + i+1], adv_image[j*BATCH_SIZE + i:j*BATCH_SIZE + i+1]))
