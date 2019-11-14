import argparse
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style('darkgrid')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Error Histogram for DRIIM P75 and P95 prediction')
    parser.add_argument('data', type=str, help='CSV containing predictions and targets')
    parser.add_argument('-o', '--output', type=str, default='hist_p.pdf', help='Output file')
    parser.add_argument('-t', '--title', type=str, default='', help='Plot title')
    args = parser.parse_args()
    
    data = pd.read_csv(args.data)
    p_cols = ['A_P75', 'L_P75', 'R_P75', 'A_P95', 'L_P95', 'R_P95']
    pred_cols = [a + '_Pred' for a in p_cols]
    
    targets = data[p_cols]
    predictions = data[pred_cols]
    predictions.columns = p_cols
        
    errors = (targets - predictions).abs()
    # print(errors.apply([np.mean, np.std]))
    # print(errors.stack().mean(), errors.stack().std())
    
    t = pd.melt(targets, var_name='Metric', value_name='Target')
    p = pd.melt(predictions, var_name='Metric', value_name='Prediction')
    
    d = t.assign(Prediction=p.Prediction)
    
    # print(d)
    g = sns.FacetGrid(d, col='Metric', col_wrap=3, sharex=False, sharey=False)
    g.map(plt.scatter, 'Prediction', 'Target', s=1)
    
    metrics = errors.apply(lambda x: '{:3.2f} ± {:3.2f}'.format(np.mean(x), np.std(x)))
    print(metrics)
    for ax, metric in zip(g.axes.flat, metrics.values):
        ax.set_title('{} (MSE = {})'.format(ax.get_title().replace('Metric = ',''), metric))
        lim = ax.collections[0].get_offsets().max()
        ax.plot((0, lim), (0, lim), '--', c='k', lw=0.5)
        ax.get_xaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax.get_yaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax.grid(b=True, which='minor', linewidth=0.5, linestyle='--')
        ax.set_aspect('equal', adjustable='box')
    
    plt.subplots_adjust(top=0.9, wspace=0.01)
    plt.suptitle('{} Global MSE = {:3.2f} ± {:3.2f}'.format(args.title, errors.stack().mean(), errors.stack().std()))
    plt.savefig(args.output, bbox_inches="tight")
    
    # errors = pd.melt(errors, var_name='Metric', value_name='Error')
    # g = sns.FacetGrid(errors, col='Metric', col_wrap=3, sharex=True, sharey=False)
    # g.map(sns.distplot, 'Error', kde=True, rug=True)
    # plt.savefig(args.output)



