import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set_theme()


def max_df(prefix, metric):
    dfs = [pd.read_csv(name) for name in glob.iglob(f'results/{prefix.lower()}*{metric.lower()}.csv')]
    indexes = np.array([df.mean() for df in dfs]).argmax(axis=0)
    return pd.concat([dfs[j].iloc[:, i] for i, j in enumerate(indexes)], axis=1)


def plot_metric(metric):
    models = ['LR', 'RF', 'GC']
    df = pd.concat([max_df(p, metric).melt(var_name='task', value_name=f'AU{metric}').assign(model=p) for p in models])
    plt.figure(figsize=(14, 5))
    sns.boxplot(data=df, x='task', y=f'AU{metric}', hue='model', width=0.7)

    for i in range(0, 16, 2):
        plt.axvspan(i + 0.5, i + 1.5, color='#dadae2', zorder=0)

    plt.tight_layout()
    plt.savefig(f'figures/{metric.lower()}.pdf')


def analysis():
    plot_metric('ROC')
    plot_metric('PRC')


if __name__ == '__main__':
    analysis()
