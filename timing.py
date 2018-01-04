import os
import argparse

from tqdm import trange

import models
import dataloaders

def main(args):
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
    
    dataloader = dataloader(args.data, **dataloader_kwargs)

    test_generator, test_iterations = dataloader.test_generator(batch_size=args.batch_size)
    print 'Images to process:', len(dataloader.test)
    
    test_generator = iter(test_generator)
    for _ in trange(test_iterations):
        x, _ = next(test_generator)
        p = model.predict_on_batch(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Forward only for timings')
    parser.add_argument('data', help='Directory containing image data')
    parser.add_argument('metric', help='The metric to learn, one of (q, vdp, driim)')
    parser.add_argument('-w', '--weights', default='ckpt/weights.29-0.19.hdf5', help='Path to HDF5 weights file')
    parser.add_argument('-a', '--arch', type=str, default='normal', help='The network architecture ([normal] | fixed_res | small)')
    parser.add_argument('-b', '--batch_size', type=int, default=4, help='Number of training epochs')
    args = parser.parse_args()
    main(args)
