import os
import re
import random
import argparse

import numpy as np
import pandas as pd

random.seed(12451)
np.random.seed(42)

import tensorflow as tf
tf.set_random_seed(342)

from model import UNet
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import Callback, LambdaCallback, TerminateOnNaN, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, CSVLogger
from sklearn.model_selection import train_test_split

# import keras.backend as K


DATA_DIR = 'data'
REFS_DIR = os.path.join(DATA_DIR, 'ref')
DIST_DIR = os.path.join(DATA_DIR, 'stim')
PMAP_DIR = os.path.join(DATA_DIR, 'vdp')

''' Not used ...
def load_pipeline(fnames):
    queue = tf.train.string_input_producer(fnames)
    reader = tf.WholeFileReader()
    _, content = reader.read(queue)
    image = tf.image.decode_png(content)
    float_image = tf.cast(image, dtype=tf.float32)
    return float_image
'''

def load_image_to_tensor(fname, grayscale=False):
    img = load_img(fname, grayscale=grayscale)
    x = img_to_array(img) / 255.0
    x = x.reshape((1,) + x.shape)
    return x


def load_data(data):
    global REFS_DIR
    global DIST_DIR
    global PMAP_DIR

    while True:
        data = data.sample(frac=1)
        for index, sample in data.iterrows():
            ref_fname = os.path.join(REFS_DIR, sample.Reference)
            dist_fname = os.path.join(DIST_DIR, sample.Distorted)
            pmap_fname = os.path.join(PMAP_DIR, sample.Map)

            yield load_image_to_tensor(ref_fname), load_image_to_tensor(dist_fname), load_image_to_tensor(pmap_fname, grayscale=True), sample.Q


def load_batch(data, batch_size=32):
    c = load_data(data)
    while True:
        refs, dists, pmaps, qs = zip(*map(next, [c,] * batch_size))
        refs = np.vstack(refs)
        dists = np.vstack(dists)
        pmaps = np.vstack(pmaps)
        qs = np.array(qs, dtype=np.float32).reshape(-1, 1) / 100.0

        yield [refs, dists], [pmaps, qs]


def main(args):

    model = UNet().create_model(img_shape=(512, 512, 3), num_class=1)
    model.compile(loss=['binary_crossentropy', 'mse'],
                 optimizer='adam') #, metrics=['binary_crossentropy', 'mse'])

    # Development inputs
    # reference_inputs = ['devel-images/reference.png']
    # distorted_inputs = ['devel-images/distorted.png']
    # pmaps_inputs = ['devel-images/P_map.png']
    # q_inputs = np.array([[38.56 / 100]])

    data = pd.read_csv('data/data.csv')
    train, valtest = train_test_split(data, test_size=0.2, random_state=42)
    val, test = train_test_split(valtest, test_size=0.5, random_state=42)

    # reference = load_pipeline(reference_inputs)
    # distorted = load_pipeline(distorted_inputs)
    # pmap = load_pipeline(pmaps_inputs)

    # reference = load_image_to_tensor(reference_inputs[0])
    # distorted = load_image_to_tensor(distorted_inputs[0])
    # pmap = load_image_to_tensor(pmaps_inputs[0], grayscale=True)
    # q = q_inputs

    # print reference
    # print distorted
    # print pmap

    # model.fit([reference, distorted], [pmap, q], batch_size=1, epochs=50)
    bs = 4

#    def tryit(a, b):
#        sample = data.iloc[548]
#        img = load_image_to_tensor(os.path.join(REFS_DIR, sample.Reference))
#        dist_img = load_image_to_tensor(os.path.join(DIST_DIR, sample.Distorted))
#        pmap, q = model.predict([img, dist_img])
#        pmap = array_to_img(pmap.reshape(* pmap.shape[1:]))
#        q = q[0][0] * 100
#        pmap.save('out/' + re.sub(r'Q_.*.png', 'Q_{:.2f}.png'.format(q), sample.Map))

#    asd = LambdaCallback(on_epoch_begin=tryit)

    callbacks = [
        TerminateOnNaN(),
        ModelCheckpoint('ckpt/weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
        TensorBoard(),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001),
        CSVLogger('training.log'),
    ]
    
    train_gen = load_batch(train, batch_size=bs)
    val_gen = load_batch(val, batch_size=bs)

    model.fit_generator(
        train_gen, # train generator
        len(train) / bs, # train steps
        epochs=30, # no. of epochs
        validation_data=val_gen, # val_generator
        validation_steps=len(val) / bs, # val steps
        workers=1, # no of loading workers
        callbacks=callbacks)

    # y, qpred = model.predict([reference, distorted])
    #
    # print 'Q=', qpred
    # img = array_to_img(y[0]*255)
    # img.save('predicted_map.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PMap and Q predictor?')
    # parser.add_argument('data', default='data/', help='Directory containing image data')
    args = parser.parse_args()
    main(args)


