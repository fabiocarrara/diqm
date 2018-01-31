import os
import argparse
import pandas as pd
import glob2 as glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('paper')
sns.set_style('whitegrid', {
  'legend.frameon': True,
  'legend.fancybox': False,
  'legend.facecolor': '1',
  'legend.edgecolor': '1',
  'grid.color': '.9',
  'grid.linestyle': '--'
})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make plots')
    parser.add_argument('runs_dir', type=str, help='The directory containing runs')
    # parser.add_argument('-', '--', action='store_true', help='')
    # parser.add_argument('-', '--', type=int, default=0, help='')
    
    args = parser.parse_args()
    
    assert os.path.exists(args.runs_dir), "Runs directory not found: {}".format(args.runs_dir)
    
    # Plot training logs
    train_logs = glob.iglob('{}/**/training.log'.format(args.runs_dir))
    for log_fname in train_logs:
        run_dir = os.path.dirname(log_fname)
        log = pd.read_csv(log_fname)
        if 'alr_maps_loss' in log.columns:
           y = ['alr_maps_loss', 'val_alr_maps_loss']
        elif 'pmap_loss' in log.columns:
           y = ['pmap_loss', 'val_pmap_loss']
        else:
           y = ['loss', 'val_loss']
        log.plot(y=y)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='best', fancybox=False, facecolor='1')
        plt.tight_layout()
        sns.despine()
        fig_fname = os.path.join(run_dir, 'loss_plot.pdf')
        plt.savefig(fig_fname)
        print 'Figure saved:', fig_fname
        plt.close()

    # Print statistics on test se predictions
    test_predictions_csv = glob.iglob('{}/**/out/predictions.csv'.format(args.runs_dir))
    stats = pd.DataFrame()
    for test_prediction_csv in test_predictions_csv:
        out_dir = os.path.dirname(test_prediction_csv)
        run_dir = os.path.dirname(out_dir)
        scenario = os.path.basename(os.path.dirname(run_dir))
        
        predictions = pd.read_csv(test_prediction_csv)
        if 'MSE' in predictions.columns:
            mean = predictions['MSE'].describe().loc['mean']
            stats = stats.append(pd.DataFrame([mean,], columns=['MSE'], index=[scenario,]))
        elif 'MSE_A' in predictions.columns:
            map_means = predictions[['MSE_{}'.format(i) for i in 'ALR']].describe().loc['mean']
            mean = map_means.values.mean()
            row = [mean,] + map_means.values.tolist()
            print row
            cols = ['MSE',] + ['MSE_{}'.format(i) for i in 'ALR']
            stats = stats.append(pd.DataFrame([row,], columns=cols, index=[scenario,]))
        else:
            continue
            
    print stats.to_latex()
