import time
import argparse
import numpy as np

from tqdm import trange

from models import get_model_for
from dataloaders import get_dataloader


def dummy_data_generator(batch_shape):
    while True:
        refs = np.empty(batch_shape, dtype=np.float32)
        dists = np.empty(batch_shape, dtype=np.float32)
        yield [refs, dists]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Forward only for timings')
    parser.add_argument('metric', help='The metric to learn, one of (q, vdp, driim)')
    parser.add_argument('-d', '--data', default='dummy', help='Directory containing image data ("dummy" if not given)')
    parser.add_argument('-i', '--input_shape', nargs='+', type=int, help='Input shape for dummy data')
    parser.add_argument('-w', '--weights', help='Path to HDF5 weights file')
    parser.add_argument('-a', '--arch', type=str, default='normal', help='The network architecture ([normal])')
    parser.add_argument('-b', '--batch_size', type=int, default=4, help='Number of training epochs')
    args = parser.parse_args()

    if args.data != 'dummy':
        dataloader, img_shape = get_dataloader(args.data, args.metric)
        print 'Images to process:', len(dataloader.test)
        generator, iterations = dataloader.test_generator(batch_size=args.batch_size)
        generator = (x[0] for x in generator)
    else:
        assert args.input_shape is not None, "You have to provide --input_shape for dummy data."
        img_shape = args.input_shape
        batch_shape = [args.batch_size, ] + img_shape
        generator = dummy_data_generator(batch_shape)
        iterations = 100

    generator = iter(generator)
    net = get_model_for(args.metric)

    model = net.create_model(img_shape=img_shape, architecture=args.arch)
    if args.weights:
        print 'Loading weights:', args.weights
        model.load_weights(args.weights)

    cumulative = 0.0
    reps = 0
    for _ in trange(iterations):
        x = next(generator)
        start = time.time()
        p = model.predict_on_batch(x)
        end = time.time()
        cumulative += (end - start)
        reps += x[0].shape[0]

    avg_time = cumulative / reps
    print 'Average process time per image:', avg_time
    print '{}: {}s'.format(' x '.join(map(str, img_shape)), avg_time)
