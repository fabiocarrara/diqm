import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(color_codes=True)

from tqdm import tqdm

from dataload import DataLoader
from model import QNet

def main(args):

    model = QNet().create_model(img_shape=(512, 512, 3), architecture=args.arch)
    model.load_weights(args.weights)

    dataloader = DataLoader(args.data, random_state=42, load_vdp=False)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Q predictor')
    parser.add_argument('-d', '--data', default='data/', help='Directory containing image data')
    parser.add_argument('-w', '--weights', default='ckpt/weights.29-0.19.hdf5', help='Path to HDF5 weights file')
    parser.add_argument('-o', '--out', default='out/', help='Where to save predicted Qs')
    parser.add_argument('-a', '--arch', type=str, default='normal', help='The network architecture (normal)')

    args = parser.parse_args()
    main(args)
