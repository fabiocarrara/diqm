import os
import random
import argparse

import numpy as np

random.seed(12451)
np.random.seed(42)

import tensorflow as tf
tf.set_random_seed(342)

from model import QNet
from dataload import DataLoader
from keras.callbacks import Callback, LambdaCallback, TerminateOnNaN, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, CSVLogger


def main(args):
    
    # decide experiment label and base directory
    exp_root_dir = 'runs/q'
    log_root_dir = 'logs/q'
    exp_label = args.label
    exp_dir = os.path.join(exp_root_dir, exp_label)
    i = 1
    while os.path.exists(exp_dir):
        exp_label = '{}.{}'.format(args.label, i)
        exp_dir = os.path.join(exp_root_dir, exp_label)
        i += 1
    
    # create experiment dir and subdirs
    print 'Experiment Directory:', exp_dir
    ckpt_dir = os.path.join(exp_dir, 'ckpt')
    log_dir = os.path.join(log_root_dir, exp_label)
        
    os.makedirs(exp_dir)
    os.makedirs(ckpt_dir)
    os.makedirs(log_dir)

    model = QNet().create_model(img_shape=(512, 512, 3), architecture=args.arch)
    model.compile(loss='mse', optimizer='adam')

    dataloader = DataLoader(args.data, random_state=42, load_vdp=False)

    callbacks = [
        TerminateOnNaN(),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001),
        TensorBoard(log_dir=log_dir),
        CSVLogger(os.path.join(exp_dir, 'training.log')),
        ModelCheckpoint(os.path.join(ckpt_dir, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')),
    ]
    
    train_gen, train_iterations = dataloader.train_generator(batch_size=args.batch_size)
    val_gen, val_iterations = dataloader.val_generator(batch_size=args.batch_size)

    model.fit_generator(
        train_gen, # train generator
        train_iterations, # train steps
        epochs=args.n_epochs, # no. of epochs
        validation_data=val_gen, # val_generator
        validation_steps=val_iterations, # val steps
        workers=1, # no of loading workers (> 1 does not work with generators)
        callbacks=callbacks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Q predictor')
    parser.add_argument('data', default='data/', help='Directory containing image data')
    parser.add_argument('-a', '--arch', type=str, default='normal', help='The network architecture ([normal] | fixed_res | small)')
    parser.add_argument('-e', '--n_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=4, help='Number of training epochs')
    parser.add_argument('-l', '--label', type=str, default=None, help='Run label')
    
    args = parser.parse_args()
    
    if args.label is None:
        args.label = '{0[arch]}_b{0[batch_size]}_e{0[n_epochs]}'.format(vars(args))
    
    main(args)


