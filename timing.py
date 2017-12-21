import argparse
import os

import numpy as np
from tqdm import trange
from keras.preprocessing.image import array_to_img

from dataload import DataLoader
from model import UNet

def main(args):

    model = UNet().create_model(img_shape=(512, 512, 3), num_class=1, architecture=args.arch)
    model.load_weights(args.weights)

    dataloader = DataLoader(args.data, random_state=42)
    test_generator, test_iterations = dataloader.test_generator(batch_size=args.batch_size)
    print 'Images to process:', len(dataloader.test)
    
    test_generator = iter(test_generator)
    for _ in trange(test_iterations):
        x, _ = next(test_generator)
        p = model.predict_on_batch(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate PMap and Q predictor')
    parser.add_argument('-d', '--data', default='data/', help='Directory containing image data')
    parser.add_argument('-w', '--weights', default='ckpt/weights.29-0.19.hdf5', help='Path to HDF5 weights file')
    parser.add_argument('-a', '--arch', type=str, default='normal', help='The network architecture ([normal] | fixed_res | small)')
    parser.add_argument('-b', '--batch-size', type=int, default=4, help='Batch size')
    args = parser.parse_args()
    main(args)
