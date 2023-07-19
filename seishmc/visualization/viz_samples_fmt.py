# -------------------------------------------------------------------
# Plot samples.
# Full moment tensor (FMT) solutions
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------

from seishmc.utils.find_solutions import find_sol

import seaborn as sns
sns.set_theme(style="ticks", palette=None)

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import pickle


def read_samples_FMT(file_path):
    '''Read samples from pickle file and get the solution'''
    with open(file_path, 'rb') as f:
        samples = pickle.load(f)
        misfit = pickle.load(f)

    samples = np.asarray(samples)
    df = pd.DataFrame({
        'strike': samples[:, 0],
        'dip':    samples[:, 1],
        'rake':   samples[:, 2],
        'Mw':     np.round(samples[:, 3], 3),
        'colatitude': samples[:, 4],
        'lunelongitude': samples[:, 5],
        'misfit': misfit,
    })

    return df, find_sol(df)


def pairplot_samples_FMT(file_path, fig_saving_path=None, init_sol=None, ref_sol=None):
    '''Plot samples in seaborn pairplot (FMT solutions)'''

    df, sol = read_samples_FMT(file_path)
    make_figure_FMT(df, fig_saving_path=fig_saving_path, init_sol=init_sol, ref_sol=ref_sol)


def make_figure_FMT(df, fig_saving_path=None, init_sol=None, nq=6, ref_sol=None):
    '''Make figure based on the dataframe'''

    # determine the solution
    sol = find_sol(df)
    n_sample = len(df)

    # Using Seaborn
    g = sns.PairGrid(df, hue='misfit', corner=True)
    g.map_lower(sns.scatterplot, palette='viridis', marker=".", linewidth=0.5, legend=True)
    g.map_lower(sns.kdeplot, levels=6, color="lightgrey", thresh=.2, alpha=0.75, linewidths=0.5, hue=None)
    g.map_diag(sns.histplot,  color='.15', hue=None)

    for i in range(nq):
        for j in range(nq):
            if i <= j:
                continue
            else:
                # other solution (true in syn. or inverted with MTUQ).
                if ref_sol is not None:
                    g.axes[i, j].scatter(ref_sol[j], ref_sol[i], marker='^', s=200, c='orange', edgecolors='black',
                                         label='True')

                # inverted solution
                g.axes[i, j].scatter(sol[j], sol[i], marker='*', s=300, c='C3', edgecolors='white')

                # initial solution
                if init_sol is not None:
                    g.axes[i, j].scatter(init_sol[j], init_sol[i], marker='o', s=100, c='white', edgecolors='black',
                                         label='Initial')


    # set x-axis and ticks
    # strike angle (degree)
    g.axes[nq - 1, 0].set_xlim([-20, 380])
    g.axes[nq - 1, 0].set_xticks([0, 180, 360])
    g.axes[nq - 1, 0].set_xticklabels([0, 180, 360])
    # dip angle (degree)
    g.axes[nq - 1, 1].set_xlim([-20, 110])
    g.axes[nq - 1, 1].set_xticks([0, 45, 90])
    g.axes[nq - 1, 1].set_xticklabels([0, 45, 90])
    # rake angle (degree)
    g.axes[nq - 1, 2].set_xlim([-110, 110])
    g.axes[nq - 1, 2].set_xticks([-90, 0, 90])
    g.axes[nq - 1, 2].set_xticklabels([-90, 0, 90])
    # co-latitude
    g.axes[nq - 1, 4].set_xlim([-20, 200])
    g.axes[nq - 1, 4].set_xticks([0, 90, 180])
    g.axes[nq - 1, 4].set_xticklabels([0, 90, 180])
    # lunelongitude
    g.axes[nq - 1, 5].set_xlim([-40, 40])
    g.axes[nq - 1, 5].set_xticks([-30, 0, 30])
    g.axes[nq - 1, 5].set_xticklabels([-30, 0, 30])

    # set y-axis and ticks
    # dip
    g.axes[1, 0].set_ylim([-20, 110])
    g.axes[1, 0].set_yticks([0, 45, 90])
    g.axes[1, 0].set_yticklabels([0, 45, 90])
    # rake
    g.axes[2, 0].set_ylim([-110, 110])
    g.axes[2, 0].set_yticks([-90, 0, 90])
    g.axes[2, 0].set_yticklabels([-90, 0, 90])
    # colatitude
    g.axes[4, 0].set_ylim([-20, 200])
    g.axes[4, 0].set_yticks([0, 90, 180])
    g.axes[4, 0].set_yticklabels([0, 90, 180])
    # lunelongitude
    g.axes[5, 0].set_ylim([-40, 40])
    g.axes[5, 0].set_yticks([-30, 0, 30])
    g.axes[5, 0].set_yticklabels([-30, 0, 30])

    # set figure size and layout
    plt.gcf().set_size_inches(12, 10)
    plt.tight_layout()

    # save figures to file
    if fig_saving_path is not None:
        fig_formats = ['png', ]
        for fmt in fig_formats:
            plt.savefig('%s_PairView_N%d.%s' % (fig_saving_path, n_sample, fmt))

    else:
        plt.show()