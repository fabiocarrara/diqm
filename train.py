import os
import re
import glob
import random
import argparse

import numpy as np

random.seed(12451)
np.random.seed(42)

import tensorflow as tf
tf.set_random_seed(342)

from models import get_model_for
from dataloaders import get_dataloader

from keras.models import load_model
from keras.callbacks import TerminateOnNaN, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, CSVLogger


def main(args):
    # Initialize data loading pipeline
    dataloader, img_shape = get_dataloader(args.data, args.metric)
    train_gen, train_iterations = dataloader.train_generator(batch_size=args.batch_size)
    val_gen, val_iterations = dataloader.val_generator(batch_size=args.batch_size)
    
    # Load dataset-specific model and losses
    net = get_model_for(args.metric)
    loss = net.get_losses()
    
    if args.resume:
        assert os.path.exists(args.resume), 'Training dir for resuming not found: {}'.format(args.resume)
        exp_dir = args.resume
        ckpt_dir = os.path.join(exp_dir, 'ckpt')
        
        # find last checkpoint
        ckpt_files = glob.glob(os.path.join(ckpt_dir, '*.hdf5'))
        last_checkpoint = ckpt_files[0]
        initial_epoch = 0
        for filename in ckpt_files:
            s = re.findall("weights.(\d+)-*", filename)
            epoch = int(s[0]) if s else -1
            if epoch > initial_epoch:
                initial_epoch = epoch
                last_checkpoint = filename
                
        initial_epoch += 1
  
        # load existing model
        print 'Resuming:', last_checkpoint
        model = load_model(last_checkpoint)
        
    else:
        # decide experiment label and base directory
        data_label = os.path.basename(os.path.normpath(args.data))
        exp_root_dir = 'runs/{}'.format(data_label)
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
            
        os.makedirs(exp_dir)
        os.makedirs(ckpt_dir)
        
        # create a new model
        model = net.create_model(img_shape=img_shape, architecture=args.arch)
        model.compile(loss=loss, optimizer='adam')
        initial_epoch = 0

    callbacks = [
        TerminateOnNaN(),
        # ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001),
        TensorBoard(log_dir=exp_dir),
        CSVLogger(os.path.join(exp_dir, 'training.log'), append=True),
        ModelCheckpoint(os.path.join(ckpt_dir, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'), save_best_only=True),
    ]

    model.fit_generator(
        train_gen, # train generator
        train_iterations, # train steps
        epochs=args.n_epochs, # no. of epochs
        validation_data=val_gen, # val_generator
        validation_steps=val_iterations, # val steps
        workers=1, # no of loading workers (> 1 does not work with generators)
        initial_epoch=initial_epoch,
        callbacks=callbacks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Deep Image Metric Predictor.')
    parser.add_argument('data', help='Directory containing image data')
    parser.add_argument('metric', help='The metric to learn, one of (q, vdp, driim)')
    parser.add_argument('-a', '--arch', type=str, default='normal', help='The network architecture ([normal] | fixed_res | small)')
    parser.add_argument('-e', '--n_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=4, help='Number of training epochs')
    parser.add_argument('-l', '--label', type=str, default=None, help='Run label')
    parser.add_argument('-r', '--resume', type=str, default=None, help='Run directory of the training to be resumed')
    
    args = parser.parse_args()
    
    if args.label is None:
        args.label = '{0[arch]}_b{0[batch_size]}_e{0[n_epochs]}'.format(vars(args))
    
    main(args)


