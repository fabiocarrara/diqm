import argparse
import os

import numpy as np
from tqdm import trange
from keras.preprocessing.image import array_to_img

from dataload import DataLoader
from model import UNet


def visualize_and_save(out_prefix, sample_data):
    ref, dist, pmap, q, pred_pmap, pred_q = sample_data
    images = np.concatenate((ref, dist), axis=1)
    pmaps = np.concatenate((pmap, pred_pmap), axis=1)
    pmaps = np.concatenate([pmaps,]*3, axis=2) # make it RGB
    preview = np.concatenate((images, pmaps), axis=0)
    preview = array_to_img(preview)
    out_fname = '{}_Q_{:4.2f}_predQ_{:4.2f}.png'.format(out_prefix, 100 * q[0], 100 * pred_q[0])
    preview.save(out_fname)


def main(args):

    model = UNet().create_model(img_shape=(512, 512, 3), num_class=1, architecture=args.arch)
    model.load_weights(args.weights)

    dataloader = DataLoader(args.data, random_state=42)
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
    parser.add_argument('-d', '--data', default='data/', help='Directory containing image data')
    parser.add_argument('-w', '--weights', default='ckpt/weights.29-0.19.hdf5', help='Path to HDF5 weights file')
    parser.add_argument('-o', '--out', default='out/', help='Where to save predicted maps')
    parser.add_argument('-a', '--arch', type=str, default='normal', help='The network architecture ([normal] | fixed_res | small)')

    args = parser.parse_args()
    main(args)
