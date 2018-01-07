import os
import argparse
import itertools

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(color_codes=True)

from tqdm import tqdm, trange
from keras.preprocessing.image import array_to_img

import models
import dataloaders


def eval_vdp(model, dataloader, args):

    def visualize_and_save(out_prefix, sample_data):
        ref, dist, pmap, q, pred_pmap, pred_q = sample_data
        images = np.concatenate((ref, dist), axis=1)
        pmaps = np.concatenate((pmap, pred_pmap), axis=1)
        pmaps = np.concatenate([pmaps,]*3, axis=2) # make it RGB
        preview = np.concatenate((images, pmaps), axis=0)
        preview = array_to_img(preview)
        out_fname = '{}_Q_{:4.2f}_predQ_{:4.2f}.png'.format(out_prefix, 100 * q[0], 100 * pred_q[0])
        preview.save(out_fname)

    test_generator, test_iterations = dataloader.test_generator(batch_size=4)
    img_paths = dataloader.test['Distorted'].values

    if not os.path.exists(args.out):
        os.makedirs(args.out)
    
    i = 0
    test_generator = iter(test_generator)
    for _ in trange(test_iterations):
        x, y = next(test_generator)
        p = model.predict_on_batch(x)
        batch_data = zip(*(x+y+p))  # unzipping all the stuff in the couples
        for sample_data in batch_data:
            # out_prefix example: /path/to/dir/img_2351_jpeg_30
            out_prefix = os.path.join(args.out, img_paths[i][:-4])
            visualize_and_save(out_prefix, sample_data)
            i += 1
            # break at the end of samples (even in the middle of a batch)
            if i == len(img_paths):
                return


def eval_q(model, dataloader, args):
    test_generator, test_iterations = dataloader.test_generator(batch_size=32)
    img_paths = dataloader.test['Distorted'].values
    
    i = 0
    test_generator = (x[0] for x in tqdm(test_generator, total=test_iterations))
    pred_qs = model.predict_generator(test_generator, steps=test_iterations)
    pred_qs = pred_qs.squeeze()[:len(dataloader.test)]*100
    dataloader.test['PredQ'] = pd.Series(pred_qs, index=dataloader.test.index)
    
    if not os.path.exists(args.out):
        os.makedirs(args.out)
        
    csv_path = os.path.join(args.out, 'predictions.csv')
    dataloader.test.to_csv(csv_path)
    
    gt = dataloader.test['Q'].values
    diffs = pred_qs - gt
    print 'MSE:', np.mean((diffs)**2)
    print 'MAE:', np.mean(np.abs(diffs))
    sns.distplot(diffs, kde=True, rug=True)
    hist_path = os.path.join(args.out, 'error_hist.png')
    plt.savefig(hist_path)
    print 'Error Hist.:', hist_path
        

def eval_driim(model, dataloader, args):

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    def visualize_and_save(out_prefix, sample_data):
        ref, dist, driim, p75, p95, pred_driim, pred_p75, pred_p95 = sample_data
        # import pdb; pdb.set_trace()
        
        # bring HDR back to [0,1] only for visualization purposes
        max_lum = max(ref.max(), dist.max())
        ref /= max_lum
        dist /= max_lum
        
        # divide ALR maps
        driim = np.split(driim, 3, axis=2)
        pred_driim = np.split(pred_driim, 3, axis=2)
        
        top = np.concatenate([ref,] + driim, axis=1)
        bottom = np.concatenate([dist,] + pred_driim, axis=1)
        preview = np.concatenate((top, bottom), axis=0)
        preview = array_to_img(preview)
        
        
        p75 *= 100.0
        p95 *= 100.0
        pred_p75 *= 100.0
        pred_p95 *= 100.0
        prob_format = 'p75_{:4.2f}_{:4.2f}_p95_{:4.2f}_{:4.2f}'
        out_format = '{}_a_' + prob_format + '_l_' + prob_format + '_r_' + prob_format + '.png'
        
        probs = itertools.chain.from_iterable(zip(p75, pred_p75, p95, pred_p95))
        format_args = [out_prefix,] + list(probs)
        out_fname = out_format.format(*format_args)
        preview.save(out_fname)

    test_generator, test_iterations = dataloader.test_generator(batch_size=4)
    img_paths = dataloader.test['Distorted'].values
    
    i = 0
    test_generator = iter(test_generator)
    for _ in trange(test_iterations):
        x, y = next(test_generator)
        p = model.predict_on_batch(x)
        batch_data = zip(*(x+y+p))  # unzipping all the stuff in the couples
        for sample_data in batch_data:
            # out_prefix example: /path/to/dir/img_2351_jpeg_30
            out_prefix = os.path.join(args.out, img_paths[i][:-4])
            visualize_and_save(out_prefix, sample_data)
            i += 1
            # break at the end of samples (even in the middle of a batch)
            if i == len(img_paths):
                return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate PMap and Q predictor')
    parser.add_argument('data', help='Directory containing image data')
    parser.add_argument('metric', help='Metric to evaluate, one of: (vdp, q, driim)')
    parser.add_argument('-w', '--weights', default='ckpt/weights.29-0.19.hdf5', help='Path to HDF5 weights file')
    parser.add_argument('-o', '--out', default='out/', help='Where to save predictions')
    parser.add_argument('-a', '--arch', type=str, default='normal', help='The network architecture ([normal] | fixed_res | small)')

    args = parser.parse_args()
    data_label = os.path.basename(os.path.normpath(args.data))
    
    net = getattr(models, '{}Net'.format(args.metric.upper()))()
    input_shape = (512, 512, 1) if args.metric == 'driim' else (512, 512, 3)
    model = net.create_model(img_shape=input_shape, architecture=args.arch)
    model.load_weights(args.weights)

    dataloader = '{}DataLoader'.format(args.metric.upper())
    dataloader = getattr(dataloaders, dataloader)
    
    dataloader_kwargs = dict(random_state=42)
    # some dataset comes in groups of augmented samples,
    # using 'group' we are not separating those samples between splits
    if data_label == 'driim-tmo':
        dataloader_kwargs['group'] = 7
    elif data_label == 'driim-comp':
        dataloader_kwargs['group'] = 77
    elif data_label == 'vdp-comp':
        dataloader_kwargs['group'] = 42
        dataloader_kwargs['hdr'] = True
    
    dataloader = dataloader(args.data, **dataloader_kwargs)

    eval_fn = globals()['eval_{}'.format(args.metric)]
    eval_fn(model, dataloader, args)
    
