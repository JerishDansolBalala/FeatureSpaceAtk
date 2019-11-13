


# data_set, model_name, decoder_name

assert data_set in ["imagenet", "cifar10"]
ENCODER_WEIGHTS_PATH = 'vgg19_normalised.npz'
base_dir_data = os.path.join("store", data_set)
base_dir_decoder = os.path.join("store", data_set, decoder_name)
base_dir_model = os.path.join("store", data_set, decoder_name, model_name)
task_dir = os.path.join("store", data_set, decoder_name, model_name, task_name)
os.makedirs(task_dir, exist_ok=True)
if data_set == "cifar10":
    assert model_name in ["cifar10_nat", "cifar10_adv", "cifar10_trades"]
    assert decoder_name in ["cifar10","cifar10_balance"]
    settings.init_settings(decoder_name, task_dir=task_dir)
    if decoder_name == "cifar10":
        Decoder_Model = "./cifar10transform1.ckpt"
    else:
        Decoder_Model = "./cifar10transform_balance1.ckpt"
elif data_set == "imagenet":
    assert model_name in ["imagenet_denoise", "imagenet_normal"]
    assert decoder_name in ["imagenet",
                            "imagenet_shallow", "imagenet_shallowest"]
    settings.init_settings(decoder_name, task_dir = task_dir)
    import gaussiansmooth.interfacegaussian as gaussdetect
    from imagenetmod.interface import imagenet
    if model_name == "imagenet_denoise":
        from imagenetmod.interface import build_imagenet_model, restore_parameter
    elif model_name == "imagenet_normal":
        from models.interface import build_imagenet_model, restore_parameter

    if decoder_name == "imagenet_shallowest":
        Decoder_Model = "./imagenetshallowesttransform1.ckpt.mode2"
    elif decoder_name == "imagenet_shallow":
        # "./trans_pretrained/imagenetshallowtransform1.ckpt-104000"
        Decoder_Model = "./imagenetshallowtransform1.ckpt.mode2"
    elif decoder_name == "imagenet":
        Decoder_Model = "./imagenettransform1.ckpt.mode2"

