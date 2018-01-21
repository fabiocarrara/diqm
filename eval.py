import os
import argparse
import itertools

import numpy as np
import pandas as pd
import glob2 as glob
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(color_codes=True)

from tqdm import tqdm, trange
from keras.preprocessing.image import array_to_img

from models import get_model_for, get_best_checkpoint
from dataloaders import get_dataloader


def eval_vdp(model, dataloader, args):

    def visualize_and_save(out_prefix, sample_data, hdr=False):
        ref, dist, pmap, q, pred_pmap, pred_q = sample_data
        
        if hdr:
            # bring HDR back to [0,1] only for visualization purposes
            min_lum = min(ref.min(), dist.min())
            if min_lum < 0:
                ref -= min_lum
                dist -= min_lum
                
            max_lum = max(ref.max(), dist.max())
            ref /= max_lum
            dist /= max_lum
            
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
    is_hdr = dataloader.hdr # whether the dataset has HDR inputs
    test_generator = iter(test_generator)
    for _ in trange(test_iterations):
        x, y = next(test_generator)
        p = model.predict_on_batch(x)
        batch_data = zip(*(x+y+p))  # unzipping all the stuff in the couples
        for sample_data in batch_data:
            # out_prefix example: /path/to/dir/img_2351_jpeg_30
            out_prefix = os.path.join(args.out, img_paths[i][:-4])
            visualize_and_save(out_prefix, sample_data, is_hdr)
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

    def visualize_and_save(out_prefix, sample_data, hdr=True):
        ref, dist, driim, p75, p95, pred_driim, pred_p75, pred_p95 = sample_data
        # import pdb; pdb.set_trace()
        if hdr:
            # bring HDR back to [0,1] only for visualization purposes
            min_lum = min (ref.min(), dist.min())
            if min_lum < 0:
                ref -= min_lum
                dist -= min_lum
                
            max_lum = max(ref.max(), dist.max())
            ref /= max_lum
            dist /= max_lum
        
        # divide ALR maps
        driim = np.split(driim, 3, axis=2)
        pred_driim = np.split(pred_driim, 3, axis=2)
        
        if ref.shape[-1] > 1: #RGB
            driim = [np.concatenate([m,]*3, axis=2) for m in driim]
            pred_driim = [np.concatenate([m,]*3, axis=2) for m in pred_driim]

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
    is_hdr = dataloader.hdr # whether the dataset has HDR inputs
    test_generator = iter(test_generator)
    for _ in trange(test_iterations):
        x, y = next(test_generator)
        p = model.predict_on_batch(x)
        batch_data = zip(*(x+y+p))  # unzipping all the stuff in the couples
        for sample_data in batch_data:
            # out_prefix example: /path/to/dir/img_2351_jpeg_30
            out_prefix = os.path.join(args.out, img_paths[i][:-4])
            visualize_and_save(out_prefix, sample_data, hdr=is_hdr)
            i += 1
            # break at the end of samples (even in the middle of a batch)
            if i == len(img_paths):
                return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate PMap and Q predictor')
    parser.add_argument('data', help='Directory containing image data')
    parser.add_argument('metric', help='Metric to evaluate, one of: (vdp, q, driim)')
    parser.add_argument('-r', '--run_dir', help='Path to run directory (best validation snapshot is selected)')
    parser.add_argument('-w', '--weights', default='.', help='Path to HDF5 weights file (ignored if specified with -r)')
    parser.add_argument('-o', '--out', default='out/', help='Where to save predictions (relative path if specified with -r)')
    parser.add_argument('-a', '--arch', type=str, default='normal', help='The network architecture ([normal] | fixed_res | small)')

    args = parser.parse_args()
    
    dataloader, img_shape = get_dataloader(args.data, args.metric)
    net = get_model_for(args.metric)
    model = net.create_model(img_shape=img_shape, architecture=args.arch)
    
    if args.run_dir: # Select the best model looking to the validation loss
        args.out = os.path.join(args.run_dir, args.out)
        args.weights = get_best_checkpoint(args.run_dir)
        
    print 'Loading weights:', args.weights
    model.load_weights(args.weights)

    eval_fn = globals()['eval_{}'.format(args.metric)]
    eval_fn(model, dataloader, args)
    
