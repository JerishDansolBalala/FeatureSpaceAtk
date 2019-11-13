#!/usr/bin/env python3

import os
import numpy as np
import tqdm
import math
import scipy.stats
from absl import logging
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression
import itertools as itt
from types import SimpleNamespace

from tensorflow.python.platform import flags

FLAGS = flags.FLAGS
logging.set_verbosity(logging.INFO)


def collect_statistics(x_train, y_train, x_ph=None, sess=None, latent_and_logits_fn_th=None, 
    latent_x_tensor=None, logits_tensor=None, nb_classes=None, weights=None, cuda=True, 
    targeted=False, noise_eps=8e-3, noise_eps_detect=None, num_noise_samples=256, batch_size=256,
     clip_min=-1., clip_max=1., p_ratio_cutoff=20., save_alignments_dir=None, load_alignments_dir=None, debug_dict=None, 
     debug=False, clip_alignments=True, pgd_train=None, fit_classifier=False, just_detect=False):
    assert len(x_train) == len(y_train)
    if pgd_train is not None:
        assert len(pgd_train) == len(x_train)

    if x_ph is not None:
        import tensorflow as tf
        backend = 'tf'
        assert sess is not None
        assert latent_and_logits_fn_th is None
        assert latent_x_tensor is not None
        assert logits_tensor is not None
        assert nb_classes is not None
        assert weights is not None


    if debug:
        logging.set_verbosity(logging.DEBUG)

    try:
        len(noise_eps)
        if isinstance(noise_eps, str):
            raise TypeError()
    except TypeError:
        noise_eps = [noise_eps]

    if noise_eps_detect is None:
        noise_eps_detect = noise_eps

    try:
        len(noise_eps_detect)
        if isinstance(noise_eps_detect, str):
            raise TypeError()
    except TypeError:
        noise_eps_detect = [noise_eps_detect]

    noise_eps_all = set(noise_eps + noise_eps_detect)

    n_batches = math.ceil(x_train.shape[0] / batch_size)

    if len(y_train.shape) == 2:
        y_train = y_train.argmax(-1)

    if backend == 'tf':
        y_ph = tf.placeholder(tf.int64, [None])
        loss_tensor = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_tensor, labels=y_ph))
        pgd_gradients = tf.gradients(loss_tensor, x_ph)[0]
        preds_tensor = tf.arg_max(logits_tensor, -1)


    def get_noise_samples(x, num_samples, noise_eps, clip=False):
        if isinstance(noise_eps, float):
            kind = 'u'
            eps = noise_eps
        else:
            kind, eps = noise_eps[:1], float(noise_eps[1:])

        if isinstance(x, np.ndarray):
            if kind == 'u':
                noise = np.random.uniform(-1., 1., size=(num_samples,) + x.shape[1:])
            elif kind == 'n':
                noise = np.random.normal(0., 1., size=(num_samples,) + x.shape[1:])
            elif kind == 's':
                noise = np.random.uniform(-1., 1., size=(num_samples,) + x.shape[1:])
                noise = np.sign(noise)
            x_noisy = x + noise * eps
            if clip:
                x_noisy = x_noisy.clip(clip_min, clip_max)
        elif backend == 'tf':
            shape = (num_samples,) + tuple(s.value for s in x.shape[1:])
            if kind == 'u':
                noise = tf.random_uniform(shape=shape, minval=-1., maxval=1.)
            elif kind == 'n':
                noise = tf.random_normal(shape=shape, mean=0., stddev=1.)
            elif kind == 's':
                noise = tf.random_uniform(shape=shape, minval=-1., maxval=1.)
                noise = tf.sign(noise)
            x_noisy = x + noise * eps
            if clip:
                x_noisy = tf.clip_by_value(x_noisy, clip_min, clip_max)
        elif backend == 'th':
            if kind == 'u':
                noise = x.new_zeros((num_samples,) + x.shape[1:]).uniform_(-1., 1.)
            elif kind == 'n':
                noise = x.new_zeros((num_samples,) + x.shape[1:]).normal_(0., 1.)
            elif kind == 's':
                noise = x.new_zeros((num_samples,) + x.shape[1:]).uniform_(-1., 1.)
                noise.sign_()
            x_noisy = x + noise * eps
            if clip:
                x_noisy.clamp_(clip_min, clip_max)
        return x_noisy


    def get_latent_and_pred(x):
        if backend == 'tf':
            return sess.run([latent_x_tensor, preds_tensor], {x_ph: x})


    x_preds_clean = []
    x_train_pgd = []
    x_preds_pgd = []
    latent_clean = []
    latent_pgd = []

    if not load_alignments_dir:
        for b in tqdm.trange(n_batches, desc='creating adversarial samples'):
            x_batch = x_train[b*batch_size:(b+1)*batch_size]
            lc, pc = get_latent_and_pred(x_batch)
            x_preds_clean.append(pc)
            latent_clean.append(lc)


        x_preds_clean, latent_clean = map(np.concatenate, (x_preds_clean, latent_clean))
        if not just_detect:
            x_train_pgd, x_preds_pgd, latent_pgd = map(np.concatenate, (x_train_pgd, x_preds_pgd, latent_pgd))

        valid_idcs = []
        if not just_detect:
            for i, (pc, pp, y) in enumerate(zip(x_preds_clean, x_preds_pgd, y_train)):
                if y == pc and pc != pp:
                # if y == pc:
                    valid_idcs.append(i)
        else:
            valid_idcs = list(range(len(x_preds_clean)))

        logging.info('valid idcs ratio: {}'.format(len(valid_idcs) / len(y_train)))
        if targeted:
            for i, xpp in enumerate(x_preds_pgd.T):
                logging.info('pgd success class {}: {}'.format(i, (xpp == i).mean()))

        x_train, y_train, x_preds_clean, latent_clean = (a[valid_idcs] for a in (x_train, y_train, x_preds_clean, latent_clean))
        if not just_detect:
            x_train_pgd, x_preds_pgd, latent_pgd = (a[valid_idcs] for a in (x_train_pgd, x_preds_pgd, latent_pgd))

    if backend == 'tf':
        weights = tf.transpose(weights, (1, 0))
        weights_np = sess.run(weights)

    big_memory = weights.shape[0] > 20
    logging.info('BIG MEMORY: {}'.format(big_memory))
    if not big_memory:
        wdiffs = weights[None, :, :] - weights[:, None, :]
        wdiffs_np = weights_np[None, :, :] - weights_np[:, None, :]

    if backend == 'tf':
        # lat_ph = tf.placeholder(tf.float32, [weights.shape[-1]])
        # pred_ph = tf.placeholder(tf.int64)
        # if big_memory:
            # wdiffs_relevant = weights[pred_ph, None] - weights
        # else:
            # wdiffs_relevant = wdiffs[:, pred_ph]
        # lat_diff_tensor = lat_ph[None] - latent_x_tensor
        # alignments_tensor = tf.matmul(lat_diff_tensor, wdiffs_relevant, transpose_b=True)

        # def _compute_neps_alignments(x, lat, pred, idx_wo_pc, neps):
            # x_noisy = get_noise_samples(x[None], num_noise_samples, noise_eps=neps, clip=clip_alignments)
            # return sess.run(alignments_tensor, {x_ph: x_noisy, lat_ph: lat, pred_ph: pred})[:, idx_wo_pc]
        lat_ph = tf.placeholder(tf.float32, [weights.shape[-1]])
        wdiffs_relevant_ph = tf.placeholder(tf.float32, [weights.shape[-1], nb_classes])
        lat_diff_tensor = lat_ph[None] - latent_x_tensor
        alignments_tensor = tf.matmul(lat_diff_tensor, wdiffs_relevant_ph)

        def _compute_neps_alignments(x, lat, pred, idx_wo_pc, neps):
            if big_memory:
                wdiffs_relevant = weights_np[pred, None] - weights_np
            else:
                wdiffs_relevant = wdiffs_np[:, pred]
            x_noisy = get_noise_samples(x[None], num_noise_samples, noise_eps=neps, clip=clip_alignments)
            # return sess.run(alignments_tensor, {x_ph: x_noisy, lat_ph: lat, wdiffs_relevant_ph:wdiffs_relevant.T})[:, idx_wo_pc]
            lat_x = sess.run(latent_x_tensor, {x_ph: x_noisy})
            lat_diffs = lat[None] - lat_x
            return np.matmul(lat_diffs, wdiffs_relevant.T)[:, idx_wo_pc]



    if debug_dict is not None:
        debug_dict['weights'] = weights_np
        debug_dict['wdiffs'] = wdiffs_np

    def _compute_alignments(x, lat, pred, source=None, noise_eps=noise_eps_all):
        if source is None:
            idx_wo_pc = [i for i in range(nb_classes) if i != pred]
            assert len(idx_wo_pc) == nb_classes - 1
        else:
            idx_wo_pc = source

        alignments = OrderedDict()
        for neps in noise_eps:
            alignments[neps] = _compute_neps_alignments(x, lat, pred, idx_wo_pc, neps)
            # if debug_dict is not None:
                # debug_dict.setdefault('lat', []).append(lat)
                # debug_dict.setdefault('lat_noisy', []).append(lat_noisy)
                # debug_dict['weights'] = weights
                # debug_dict['wdiffs'] = wdiffs
        return alignments, idx_wo_pc

    def _collect_wdiff_stats(x_set, latent_set, x_preds_set, clean, save_alignments_dir=None, load_alignments_dir=None):
        if clean:
            wdiff_stats = {(tc, tc, e): [] for tc in range(nb_classes) for e in noise_eps_all}
            name = 'clean'
        else:
            wdiff_stats = {(sc, tc, e): [] for sc in range(nb_classes) for tc in range(nb_classes) for e in noise_eps_all if sc != tc}
            name = 'adv'

        def _compute_stats_from_values(v, raw=False):
            if not v.shape:
                return None
            v = v.mean(1)
            if debug:
                v = np.concatenate([v, v*.5, v*1.5])
            if clean or not fit_classifier:
                if v.shape[0] < 3:
                    return None
                return v.mean(0), v.std(0)
            else:
                return v

        for neps in noise_eps_all:
            neps_keys = {k for k in wdiff_stats.keys() if k[-1] == neps}
            loading = load_alignments_dir
            if loading:
                for k in neps_keys:
                    fn = 'alignments_{}_{}.npy'.format(name, str(k))
                    load_fn = os.path.join(load_alignments_dir, fn)
                    if not os.path.exists(load_fn):
                        loading = False
                        break
                    v = np.load(load_fn)
                    wdiff_stats[k] = _compute_stats_from_values(v)
                logging.info('loading alignments from {} for {}'.format(load_alignments_dir, neps))
            if not loading:
                for x, lc, pc, pcc in tqdm.tqdm(zip(x_set, latent_set, x_preds_set, x_preds_clean), total=len(x_set), desc='collecting stats for {}'.format(neps)):
                    if len(lc.shape) == 2:
                        alignments = []
                        for i, (xi, lci, pci) in enumerate(zip(x, lc, pc)):
                            if i == pcc:
                                continue
                            alignments_i, _ = _compute_alignments(xi, lci, i, source=pcc, noise_eps=[neps])
                            for e, a in alignments_i.items():
                                wdiff_stats[(pcc, i, e)].append(a)
                    else:
                        alignments, idx_wo_pc = _compute_alignments(x, lc, pc, noise_eps=[neps])
                        for e, a in alignments.items():
                            wdiff_stats[(pcc, pc, e)].append(a)

                saving = save_alignments_dir and not loading
                if saving:
                    logging.info('saving alignments to {} for {}'.format(save_alignments_dir, neps))
                if debug:
                    some_v = None
                    for k in neps_keys:
                        some_v = some_v or wdiff_stats[k]
                    for k in neps_keys:
                        wdiff_stats[k] = wdiff_stats[k] or some_v

                for k in neps_keys:
                    wdsk = wdiff_stats[k]
                    if len(wdsk):
                        wdiff_stats[k] = np.stack(wdsk)
                    else:
                        wdiff_stats[k] = np.array(None)
                    if saving:
                        fn = 'alignments_{}_{}.npy'.format(name, str(k))
                        save_fn = os.path.join(save_alignments_dir, fn)
                        os.makedirs(os.path.dirname(save_fn), exist_ok=True)
                        wds = wdiff_stats[k]
                        np.save(save_fn, wds)
                    wdiff_stats[k] = _compute_stats_from_values(wdiff_stats[k])
        return wdiff_stats

    save_alignments_dir_clean = os.path.join(save_alignments_dir, 'clean') if save_alignments_dir else None
    save_alignments_dir_pgd = os.path.join(save_alignments_dir, 'pgd') if save_alignments_dir else None
    load_alignments_dir_clean = os.path.join(load_alignments_dir, 'clean') if load_alignments_dir else None
    load_alignments_dir_pgd = os.path.join(load_alignments_dir, 'pgd') if load_alignments_dir else None
    if load_alignments_dir:
        load_alignments_dir_clean, load_alignments_dir_pgd = map(lambda s: '{}_{}'.format(s, 'clip' if clip_alignments else 'noclip'), (load_alignments_dir_clean, load_alignments_dir_pgd))
    if save_alignments_dir:
        save_alignments_dir_clean, save_alignments_dir_pgd = map(lambda s: '{}_{}'.format(s, 'clip' if clip_alignments else 'noclip'), (save_alignments_dir_clean, save_alignments_dir_pgd))
    wdiff_stats_clean = _collect_wdiff_stats(x_train, latent_clean, x_preds_clean, clean=True, save_alignments_dir=save_alignments_dir_clean, load_alignments_dir=load_alignments_dir_clean)

    if debug_dict is not None and False:
        esizes = OrderedDict((k, []) for k in noise_eps_all)
        for k, (mc, sc) in wdiff_stats_clean.items():
            mp, sp = wdiff_stats_pgd[k]
            esizes[k[-1]].append(np.abs(mp - mc) / ((sp + sc) / 2.))
        debug_dict['effect_sizes'] = OrderedDict((k, np.array(v)) for k, v in esizes.items())

    wdiff_stats_clean_detect = [np.stack([wdiff_stats_clean[(p, p, eps)] for eps in noise_eps_detect]) for p in range(nb_classes)]
    wdiff_stats_clean_detect = [s.transpose((1, 0, 2)) if len(s.shape) == 3 else None for s in wdiff_stats_clean_detect]
    wdiff_stats_pgd_classify = []

    for i in range(nb_classes):
        if (wdiff_stats_clean_detect[i] is None):
            print (i, "has no stat")
    batch = yield

    while batch is not None:
        batch_latent, batch_pred = get_latent_and_pred(batch)
        if debug_dict is not None:
            debug_dict.setdefault('batch_pred', []).append(batch_pred)
        corrected_pred = []
        detection = []
        for b, lb, pb in zip(batch, batch_latent, batch_pred):
            b_align, idx_wo_pb = _compute_alignments(b, lb, pb)
            b_align_det = np.stack([b_align[eps] for eps in noise_eps_detect])
            b_align = np.stack([b_align[eps] for eps in noise_eps])

            wdsc_det_pb = wdiff_stats_clean_detect[pb]
            if wdsc_det_pb is None:
                print("Assert False in gaussian",str(pb))
                z_hit = False
            else:
                wdm_det, wds_det = wdsc_det_pb
                z_clean = (b_align_det - wdm_det[:, None]) / wds_det[:, None]
                z_clean_mean = z_clean.mean(1)
                z_cutoff = scipy.stats.norm.ppf(p_ratio_cutoff)
                z_hit = z_clean_mean.mean(0).max(-1) > z_cutoff

            if z_hit:
                detection.append(True)
            else:
                detection.append(False)
            if debug_dict is not None:
                debug_dict.setdefault('b_align', []).append(b_align)
                # debug_dict.setdefault('stats', []).append((wdm_det, wds_det, wdmp, wdsp))
                # debug_dict.setdefault('p_ratio', []).append(p_ratio)
                # debug_dict.setdefault('p_clean', []).append(p_clean)
                # debug_dict.setdefault('p_pgd', []).append(p_pgd)
                debug_dict.setdefault('z_clean', []).append(z_clean)
                # debug_dict.setdefault('z_conf', []).append(z_conf)
                # debug_dict.setdefault('z_pgdm', []).append(z_pgdm)
                # debug_dict.setdefault('z_pgd', []).append(z_pgd)
            corrected_pred.append(pb)
        if debug_dict is not None:
            debug_dict.setdefault('detection', []).append(detection)
            debug_dict.setdefault('corrected_pred', []).append(corrected_pred)
        batch = yield np.stack((corrected_pred, np.array(detection,dtype=np.float32)), -1)

def build_detect(sess,batch_size,base_dir,input,logits,x_train,y_train,dataset="imagenet"):
    global predictor
    if dataset == "imagenet":
        noise_eps_detect = flags.DEFINE_string('noise_eps_detect', 's3.0,s5.0,s7.0', '')
        flags.DEFINE_string('noise_eps', 's3.0,s5.0,s7.0', '')
        flags.DEFINE_integer('n_collect', 10000, '')
    elif dataset == "cifar10":
        #noise_eps_detect=s0.05,s0.08,s0.1
        noise_eps_detect = flags.DEFINE_string(
            'noise_eps_detect', 's12.75,s20.4,s25.5', '')
        flags.DEFINE_string('noise_eps', 's12.75,s20.4,s25.5', '')
        flags.DEFINE_integer('n_collect', 20000, '')

    base_dir=os.path.join(".",base_dir)
    print(base_dir)
    if os.path.exists(base_dir): #,"alignment"
        load_alignments_dir = base_dir
        save_alignments_dir = None
    else:
        save_alignments_dir = base_dir
        load_alignments_dir = None
    preds = logits
    x=input
    logits_op = preds.op
    eval_params = {'batch_size': batch_size}
    pgd_params = {
        # ord: ,
        'eps': FLAGS.eps,
        'eps_iter': (FLAGS.eps / 5),
        'nb_iter': 10,
        'clip_min': 0,
        'clip_max': 255
    }
    logits_op = preds.op
    while logits_op.type != 'MatMul':
        logits_op = logits_op.inputs[0].op
    latent_x_tensor, weights = logits_op.inputs
    logits_tensor = preds
    nb_classes = weights.shape[-1].value
    
    noise_eps = FLAGS.noise_eps.split(',')
    if FLAGS.noise_eps_detect is None:
        FLAGS.noise_eps_detect = FLAGS.noise_eps
    noise_eps_detect = FLAGS.noise_eps_detect.split(',')
    predictor = collect_statistics(x_train[:FLAGS.n_collect], y_train[:FLAGS.n_collect], x, sess, 
                                                logits_tensor=logits_tensor, latent_x_tensor=latent_x_tensor, weights=weights, nb_classes=nb_classes, p_ratio_cutoff=FLAGS.p_ratio_cutoff, noise_eps=noise_eps, noise_eps_detect=noise_eps_detect,  save_alignments_dir=save_alignments_dir,
                load_alignments_dir=load_alignments_dir, clip_min=pgd_params['clip_min'], clip_max=pgd_params['clip_max'], batch_size=batch_size, num_noise_samples=FLAGS.num_noise_samples, debug_dict=None, debug=FLAGS.debug, targeted=False, pgd_train=None, fit_classifier=FLAGS.fit_classifier, clip_alignments=FLAGS.clip_alignments, just_detect=True)
    next(predictor)

def detect(x_batch,y_batch,batch_size):
    global predictor
    b=0
    
    p_set, p_det = np.concatenate([predictor.send(
                x_batch[b*batch_size:(b+1)*batch_size]) for b in tqdm.trange(1)]).T
    print("pset",p_set)
    print("pdet",p_det)
    print(y_batch[:len(p_set)])
    acc = np.equal(p_set, y_batch[:len(p_set)]).mean()
    print("acc",acc)
    return y_batch[:len(p_set)],p_set,p_det

flags.DEFINE_bool('backprop_through_attack', False,
                    ('If True, backprop through adversarial example '
                    'construction process during adversarial training'))

flags.DEFINE_float('p_ratio_cutoff', .999, '')
flags.DEFINE_float('eps', 8., '')
#flags.DEFINE_string('noise_eps', 'n18.0,n24.0,n30.0', '')
#flags.DEFINE_string('noise_eps_detect', 'n30.0', '')
flags.DEFINE_bool('debug', False, 'for debugging')
flags.DEFINE_integer('test_size', 10000, '')
flags.DEFINE_bool('save_alignments', False, '')
flags.DEFINE_bool('load_alignments', False, '')
flags.DEFINE_integer('num_noise_samples', 256, '')
flags.DEFINE_integer('rep', 0, '')
flags.DEFINE_bool('save_debug_dict', False, '')
flags.DEFINE_bool('save_pgd_samples', False, '')
flags.DEFINE_string('load_pgd_train_samples', None, '')
flags.DEFINE_string('load_pgd_test_samples', None, '')
flags.DEFINE_bool('fit_classifier', True, '')
flags.DEFINE_bool('clip_alignments', True, '')
flags.DEFINE_string('attack', 'pgd', '')
flags.DEFINE_bool('passthrough', False, '')
flags.DEFINE_integer('cw_steps', 300, '')
flags.DEFINE_integer('cw_search_steps', 20, '')
flags.DEFINE_float('cw_lr', 1e-1, '')
flags.DEFINE_float('cw_c', 1e-4, '')
flags.DEFINE_bool('just_detect', False, '')
flags.DEFINE_integer('mean_samples', 16, '')
flags.DEFINE_float('mean_eps', .1, '')
flags.DEFINE_bool('load_adv_trained', False, '')

