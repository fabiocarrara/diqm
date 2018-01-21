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
    
    train_logs = glob.iglob('{}/**/training.log'.format(args.runs_dir))
    for log_fname in train_logs:
        run_dir = os.path.dirname(log_fname)
        log = pd.read_csv(log_fname)
        log.plot(y=['loss', 'val_loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='best', fancybox=False, facecolor='1')
        plt.tight_layout()
        sns.despine()
        fig_fname = os.path.join(run_dir, 'loss_plot.pdf')
        plt.savefig(fig_fname)
        print 'Figure saved:', fig_fname
        plt.close()

