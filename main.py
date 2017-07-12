import argparse
from model import UNet
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import tensorflow as tf
import numpy as np


def load_pipeline(fnames):
    queue = tf.train.string_input_producer(fnames)
    reader = tf.WholeFileReader()
    _, content = reader.read(queue)
    image = tf.image.decode_png(content)
    float_image = tf.cast(image, dtype=tf.float32)
    return float_image


def load_image_to_tensor(fname, grayscale=False):
    img = load_img(fname, grayscale=grayscale)
    x = img_to_array(img) / 255.0
    x = x.reshape((1,) + x.shape)
    return x


def main(args):

    model = UNet().create_model(img_shape=(512, 512, 3), num_class=1)
    model.compile(loss=['binary_crossentropy', 'mse'],
                 optimizer='adam',)
                 # metrics=['binary_crossentropy', 'mse'])

    # Development inputs
    reference_inputs = ['devel-images/reference.png']
    distorted_inputs = [' devel-images/distorted.png']
    pmaps_inputs = [' devel-images/P_map.png']
    q_inputs = np.array([[38.56 / 100]])

    # reference = load_pipeline(reference_inputs)
    # distorted = load_pipeline(distorted_inputs)
    # pmap = load_pipeline(pmaps_inputs)

    reference = load_image_to_tensor(reference_inputs[0])
    distorted = load_image_to_tensor(distorted_inputs[0])
    pmap = load_image_to_tensor(pmaps_inputs[0], grayscale=True)
    q = q_inputs

    # print reference
    # print distorted
    # print pmap

    model.fit([reference, distorted], [pmap, q], batch_size=1, epochs=50)
    y, qpred = model.predict([reference, distorted])

    print 'Q=', qpred
    img = array_to_img(y[0]*255)
    img.save('predicted_map.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PMap and Q predictor ?')
    # parser.add_argument('data', default='data/', help='Directory containing image data')
    args = parser.parse_args()
    main(args)


