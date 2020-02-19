import tensorflow as tf
import logging
import sys

def init_settings(data_set,suffix="",task_dir=""):
    global config
    config={}
    assert data_set in ["cifar10", "imagenet",
                        "imagenet_shallow", "imagenet_shallowest", "imagenet_quality", "cifar10_balance", "imagenet_smooth", "imagenet_shallow_smooth",
                        "imagenet_shallowest_smooth"]
    config["style_weight"]=1
    config["data_mode"] = 2
    if data_set=="cifar10":
        config["IMAGE_SHAPE"] = [32,32,3]
        config["DECODER_DIM"] = [16, 16, 512]

        config["BATCH_SIZE"] = 64
        config["ENCODER_LAYERS"] = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1')
        config["DECODER_LAYERS"] = ('conv2_1', 'conv1_2', 'conv1_1')
        config["upsample_indices"] = (0, )
        config["STYLE_LAYERS"] = ('relu1_1', 'relu2_1')

        config["pretrained_model"] = "pretrained.ckpt"
        config["hardened_model"] = "hardened.ckpt"
        config["model_save_path"] = "./cifar10transform%d.ckpt" % (config["style_weight"])

    elif data_set == "cifar10_balance":
        config["IMAGE_SHAPE"] = [32, 32, 3]
        config["DECODER_DIM"] = [16, 16, 128]
        config["INTERPOLATE_NUM"] = 100 + 1
        config["BATCH_SIZE"] = 64
        config["ENCODER_LAYERS"] = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1')
        config["DECODER_LAYERS"] = ('conv2_1', 'conv1_2', 'conv1_1')
        config["upsample_indices"] = (0, )
        config["STYLE_LAYERS"] = ('relu1_1', 'relu2_1')

        config["pretrained_model"] = "pretrained.ckpt"
        config["hardened_model"] = "hardened.ckpt"
        config["model_save_path"] = "./cifar10transform_balance%d.ckpt" % (
            config["style_weight"])
        config["balance_weight"] = 1.0

    elif data_set=="imagenet":
        config["INTERPOLATE_NUM"] = 50 + 1
        config["IMAGE_SHAPE"] = [224, 224, 3]
        config["DECODER_DIM"] = [28, 28, 512]
        config["BATCH_SIZE"] = 8
        config["ENCODER_LAYERS"] = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1','conv2_2', 'relu2_2', 'pool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
            'conv4_1', 'relu4_1')
        config["DECODER_LAYERS"] = ('conv4_1',
                                    'conv3_4', 'conv3_3', 'conv3_2', 'conv3_1', 
                                    'conv2_2', 'conv2_1', 
                                    'conv1_2', 'conv1_1')
        config["upsample_indices"] = (0, 4, 6)
        config["STYLE_LAYERS"] = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1')

        config["pretrained_model"] = "imagenet_pretrained.ckpt"
        config["hardened_model"] = "imagenet_hardened.ckpt"
        config["model_save_path"] = "./imagenettransform%d.ckpt" % (
            config["style_weight"])

    elif data_set == "imagenet_shallow":
        config["balance_weight"] = 1000
        config["Decoder_Layer"] = "deconv"
        config["INTERPOLATE_NUM"] = 50 + 1
        config["DECODER_DIM"] = [56, 56, 256]
        config["IMAGE_SHAPE"] = [224, 224, 3]
        config["BATCH_SIZE"] = 8
        config["ENCODER_LAYERS"] = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
            'conv3_1', 'relu3_1', )
        config["DECODER_LAYERS"] = ('conv3_1',
                                    'conv2_2', 'conv2_1',
                                    'conv1_2', 'conv1_1')
        config["upsample_indices"] = (1, 3)
        config["STYLE_LAYERS"] = ('relu1_1', 'relu2_1', 'relu3_1')

        config["pretrained_model"] = "imagenet_pretrained.ckpt"
        config["hardened_model"] = "imagenet_hardened.ckpt"
        config["model_save_path"] = "./imagenetshallowtransform%d.ckpt" % (
            config["style_weight"])

    elif data_set == "imagenet_shallowest":
        config["INTERPOLATE_NUM"] = 50 + 1
        config["DECODER_DIM"] = [112, 112, 128]
        config["IMAGE_SHAPE"] = [224, 224, 3]
        config["BATCH_SIZE"] = 8
        config["ENCODER_LAYERS"] = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2')
        config["DECODER_LAYERS"] = ('conv2_2','conv2_1', 'conv1_2', 'conv1_1')
        config["upsample_indices"] = (1, )
        config["STYLE_LAYERS"] = ('relu1_1', 'relu2_1')

        config["pretrained_model"] = "imagenet_pretrained.ckpt"
        config["hardened_model"] = "imagenet_hardened.ckpt"
        config["model_save_path"] = "./imagenetshallowesttransform%d.ckpt" % (
            config["style_weight"])
        config["Decoder_Layer"] = "deconv"
        config["balance_weight"] = 100

    elif data_set == "imagenet_shallowest_smooth":
        config["INTERPOLATE_NUM"] = 50 + 1
        config["DECODER_DIM"] = [112, 112, 128]
        config["IMAGE_SHAPE"] = [224, 224, 3]
        config["BATCH_SIZE"] = 8
        config["ENCODER_LAYERS"] = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1')
        config["DECODER_LAYERS"] = ('conv2_1', 'conv1_2', 'conv1_1')
        config["upsample_indices"] = (0, )
        config["STYLE_LAYERS"] = ('relu1_1', 'relu2_1')

        config["pretrained_model"] = "imagenet_pretrained.ckpt"
        config["hardened_model"] = "imagenet_hardened.ckpt"
        config["model_save_path"] = "./imagenetshallowesttransform_smooth%d.ckpt" % (
            config["style_weight"])
        config["balance_weight"] = 1
        config["Decoder_Layer"] = "deconv"
        #config["TRAIN_SMOOTH_ENABLE"] = "Key_Exists"

    elif data_set == "imagenet_quality":
        config["INTERPOLATE_NUM"] = 50 + 1
        config["IMAGE_SHAPE"] = [224, 224, 3]
        config["DECODER_DIM"] = [28, 28, 512]
        config["BATCH_SIZE"] = 8
        config["ENCODER_LAYERS"] = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
            'conv4_1', 'relu4_1')
        config["DECODER_LAYERS"] = ('conv4_1',
                                    'conv3_4', 'conv3_3', 'conv3_2', 'conv3_1',
                                    'conv2_2', 'conv2_1',
                                    'conv1_2', 'conv1_1')
        config["Decoder_Layer"]="deconv"
        config["upsample_indices"] = (0, 4, 6)
        config["STYLE_LAYERS"] = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1')

        config["pretrained_model"] = "imagenet_pretrained.ckpt"
        config["hardened_model"] = "imagenet_hardened.ckpt"
        config["model_save_path"] = "./imagenettransform_new%d.ckpt" % (
            config["style_weight"])
        config["balance_weight"] = 0.1

    elif data_set == "imagenet_smooth":
        config["INTERPOLATE_NUM"] = 50 + 1
        config["IMAGE_SHAPE"] = [224, 224, 3]
        config["DECODER_DIM"] = [28, 28, 512]
        config["BATCH_SIZE"] = 8
        config["ENCODER_LAYERS"] = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
            'conv4_1', 'relu4_1')
        config["DECODER_LAYERS"] = ('conv4_1',
                                    'conv3_4', 'conv3_3', 'conv3_2', 'conv3_1',
                                    'conv2_2', 'conv2_1',
                                    'conv1_2', 'conv1_1')
        config["Decoder_Layer"] = "deconv"
        config["upsample_indices"] = (0, 4, 6)
        config["STYLE_LAYERS"] = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1')

        config["pretrained_model"] = "imagenet_pretrained.ckpt"
        config["hardened_model"] = "imagenet_hardened.ckpt"
        config["model_save_path"] = "./imagenettransform_smooth%d.ckpt" % (
            config["style_weight"])
        config["balance_weight"] = 0
        config["TRAIN_SMOOTH_ENABLE"] = "Key_Exists"
        

    elif data_set == "imagenet_shallow_smooth":
        config["INTERPOLATE_NUM"] = 50 + 1
        config["DECODER_DIM"] = [56, 56, 256]
        config["IMAGE_SHAPE"] = [224, 224, 3]
        config["BATCH_SIZE"] = 8
        config["ENCODER_LAYERS"] = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
            'conv3_1', 'relu3_1', )
        config["DECODER_LAYERS"] = ('conv3_1',
                                    'conv2_2', 'conv2_1',
                                    'conv1_2', 'conv1_1')
        config["Decoder_Layer"] = "deconv"
        config["upsample_indices"] = (0, 2)
        config["STYLE_LAYERS"] = ('relu1_1', 'relu2_1', 'relu3_1')

        config["pretrained_model"] = "imagenet_pretrained.ckpt"
        config["hardened_model"] = "imagenet_hardened.ckpt"
        config["model_save_path"] = "./imagenetshallowtransform_smooth%d.ckpt" % (
            config["style_weight"])

        config["balance_weight"] = 1
        #config["TRAIN_SMOOTH_ENABLE"] = "Key_Exists"

    global logger

    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT,
                        filename=task_dir+"log.log")
    logger = logging.getLogger()
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
