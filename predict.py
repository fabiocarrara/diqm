import os
import argparse
import numpy as np

from tqdm import tqdm
from scipy.io import loadmat
from keras.preprocessing.image import img_to_array, array_to_img, load_img

from models import get_model_for, get_best_checkpoint


def _read_img2tensor(fname, grayscale=False):
    img = load_img(fname, grayscale=grayscale)
    x = img_to_array(img) / 255.0
    x = x.reshape((1,) + x.shape)
    return x


def _read_mat2tensor(fname, log_range=True):
    x = loadmat(fname, verify_compressed_data_integrity=False)['image'].astype(np.float32)
    if log_range: # perform log10(1 + image)
        # np.add(x, 10e-6, out=x)
        # np.log10(x, out=x)
        np.log1p(x, out=x) # base e
        np.divide(x, np.log(10), out=x) # back to base 10
    x = x.reshape((1,) + x.shape)
    
    if x.ndim < 4:
        x = np.expand_dims(x, axis=-1)
        
    return x


def visualize_vdp(y, dist_fname, out_dir):
    pmap = array_to_img(y[0][0])
    base_fname = os.path.splitext(dist_fname)[0]
    out_fname = '{}_vdp.png'.format(base_fname)
    out_fname = os.path.join(out_dir, out_fname)
    pmap.save(out_fname)
    print 'Saved:', out_fname


def visualize_q(y, dist_fname, out_dir):
    print dist_fname, ':', y[0][0]
      

def visualize_driim(y, dist_fname, out_dir):
    base_fname = os.path.splitext(dist_fname)[0]

    driim = np.split(y[0][0], 3, axis=2)
    driim = [array_to_img(pmap) for pmap in driim]
    for pmap, suffix in zip(driim, ('alr')):
        out_fname = '{}_driim_{}.png'.format(base_fname, suffix)
        out_fname = os.path.join(out_dir, out_fname)
        pmap.save(out_fname)
        print 'Saved:', out_fname


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate PMap and Q predictor')
    parser.add_argument('metric', help='Metric to evaluate, one of: (vdp, q, driim)')
    parser.add_argument('ref_image', help='Path to reference image')
    parser.add_argument('dist_images', nargs='+', help='Path to one or more distorted images')
    parser.add_argument('-r', '--run_dir', help='Path to run directory (best validation snapshot is selected)')
    parser.add_argument('-w', '--weights', help='Path to HDF5 weights file (ignored if specified with -r)')
    parser.add_argument('-o', '--out', default='.', help='Where to save predictions')
    parser.add_argument('-a', '--arch', type=str, default='normal', help='The network architecture ([normal] | fixed_res | small)')

    args = parser.parse_args()
    if not (args.run_dir or args.weights):
        parser.error('No weights file provided, please specify at least one between -r and -w.')

    assert os.path.exists(args.ref_image), 'File not found: {}'.format(args.ref_image)
    for i in args.dist_images:
        assert os.path.exists(i), 'File not found: {}'.format(i)
    
    hdr = args.ref_image.lower().endswith('.mat')
    load_image_fn = _read_mat2tensor if hdr else _read_img2tensor
    visualize_fn = globals()['visualize_{}'.format(args.metric)]
    
    ref = load_image_fn(args.ref_image)
    img_shape = ref.shape[1:]
    
    net = get_model_for(args.metric)
    model = net.create_model(img_shape=img_shape, architecture=args.arch)
    
    if args.run_dir: # Select the best model looking to the validation loss
        args.weights = get_best_checkpoint(args.run_dir)
        
    print 'Loading weights:', args.weights
    model.load_weights(args.weights)
    
    for dist_fname in tqdm(args.dist_images):
        dist = load_image_fn(dist_fname)
        x = [ref, dist]
        y = model.predict(x)
        dist_fname = os.path.basename(dist_fname)
        visualize_fn(y, dist_fname, args.out)
    
