# -*- coding: utf-8 -*-
"""

Author:   Metin Bicer
email:    m.bicer19@imperial.ac.uk

reproduces results for the dynamical inconsistency analysis in the paper

"""
import numpy as np
import matplotlib.pyplot as plt
import spm1d
import os
from utils import read_dataframes


def diff_dynamic_cons(models, percent_to, plot, save,
                      exp_color='b', synthetic_color='r', spm_color='grey',
                      save_name='SPM', plot_individual=False, alpha=0.2):
    """
    compares dynamical inconsistencies (forces and moments on pelvis
                                        calculated by ID)

    Args:
        models (list): generative model names (has to have results saved
                                               on 'Results/opensim').
        percent_to (bool): True or False to express differences in % of stance
                            and swing phases.
        plot (bool): True or False to plot
        save (bool): True or False to save the plot
        exp_color (str): Exp data color.
        synthetic_color (str): Synthetic data color.
        spm_color (str): Color for shading differences.
        save_name (str): save the plot to 'save_name.png'.
        plot_individual (bool): True if plotting data individually.
                                False if plotting mean-std.
        alpha (float): Color intensity in plots. 0<alpha<1.

    Returns:
        Prints spm differences and plots if plot=True.

    """
    # read exp data
    data_df = read_dataframes(['data/data_1.pickle', 'data/data_2.pickle'])
    excluded_subjects = [2014001, 2014003, 2015042]
    data_df_train = data_df[~data_df.subject.isin(excluded_subjects)].reset_index(drop=True)
    # id results excluding time
    id_exp = np.array([id_[0,:,1:] for id_ in data_df_train['id_gc']])
    for model_i, model in enumerate(models):
        # plot
        if plot:# and axs is None:
            fig, axs = plt.subplots(2, 3)
            mngr = plt.get_current_fig_manager()
            # mngr.window.setGeometry(263, 59, 1340, 912)
            mngr.window.setGeometry(38, 59, 1724, 894)
            axs = axs.flatten()
        print(model)
        print('='*50)
        fold = f'Results/opensim/{model}'
        # get vertical forces to find toeoffs
        vgrfs = np.load(f'{fold}/grf_results.npy')[:,:,1]
        # find toe off times
        mid_cycle = int(vgrfs.shape[1]/2)
        toeoff = np.argmax(vgrfs[:, mid_cycle:] < 10, axis=1) + mid_cycle
        # use its mean
        toeoff = np.mean(toeoff).astype(int)
        before_p, after_p = 1, 1
        if percent_to:
            before_p = 100/toeoff
            after_p = 100/(100-toeoff)
        id_syn = np.load(f'{fold}/id_results.npy')
        id_features = np.load(f'{fold}/id_features.npy')
        print(f'{"Force/Moment":<25s}{"Before_TO":>25s}{"After_TO":>25s}')
        ax_i = 0
        for i, feature in enumerate(id_features):
            printing = f'{feature:<25s}'
            if 'pelvis' in feature and ('force' in feature or 'moment' in feature):
                feature_exp = id_exp[:, :, i]
                feature_syn = id_syn[:, :, i]
                t  = spm1d.stats.ttest2(feature_exp,
                                        feature_syn,
                                        equal_var=False)
                ti = t.inference(alpha=0.05, two_tailed=True, interp=True)
                # get clusters and add significant differences
                # get clusters and add significant differences
                sig_before = 0
                sig_after = 0
                for cluster in ti.clusters:
                    # if toe off is in cluster
                    if cluster.endpoints[0]<toeoff<cluster.endpoints[1]:
                        sig_before += (toeoff-cluster.endpoints[0])*before_p
                        sig_after += (cluster.endpoints[1]-toeoff)*after_p
                    elif toeoff<cluster.endpoints[0]:
                        sig_after += cluster.extent*after_p
                    elif toeoff>cluster.endpoints[1]:
                        sig_before += cluster.extent*before_p
                printing += f'{sig_before:25.1f}{sig_after:25.1f}'
                print(printing)
                # plot
                if plot:
                    ax = axs[ax_i]
                    ax_i += 1
                    # set title and font sizes of ticks
                    plot_name = ' '.join(feature.split('_'))
                    ax.set_title(plot_name, fontsize=20)
                    ax.tick_params(axis='both', which='major', labelsize=15)
                    ax.tick_params(axis='both', which='minor', labelsize=12)
                    # plot zero line
                    ax.axhline(y=0.0, color='k', linestyle='-', linewidth=0.5)
                    # plot toeoff
                    ax.axvline(x=toeoff, color='k', linestyle='--', linewidth=0.5)
                    if plot_individual:
                        # plot individual curves
                        ax.plot(feature_exp.T, color=exp_color,
                                label='Experimental', alpha=alpha)
                        ax.plot(feature_syn.T, color=synthetic_color,
                                label='Synthetic', alpha=alpha)
                    else:
                        # plot mean sd
                        spm1d.plot.plot_mean_sd(feature_exp, ax=ax, linecolor=exp_color,
                                                facecolor=exp_color, edgecolor=exp_color,
                                                label='Experimental')
                        spm1d.plot.plot_mean_sd(feature_syn, ax=ax, linecolor=synthetic_color,
                                                facecolor=synthetic_color, edgecolor=synthetic_color,
                                                label='Synthetic')
                    # indicate significant difference regions
                    for cluster in ti.clusters:
                        ax.axvspan(cluster.endpoints[0], cluster.endpoints[1],
                                   alpha=0.3, color=spm_color)
                    # set font sizes of ticks
                    ax.tick_params(axis='both', which='major', labelsize=15)
                    ax.tick_params(axis='both', which='minor', labelsize=12)
                    ax.set_xlim([0, 100])

        print('\n')
        # after plotting all
        if plot:
            # set y labels for first column
            axs[0].set_ylabel('Moment [Nm]', fontsize=25)
            axs[3].set_ylabel('Force [N]', fontsize=25)
            # set x label for columns
            for i in [3, 4, 5]: axs[i].set_xlabel('Gait Cycle [\%]', fontsize=25)
            # get legend handles and labels
            handles, labels = axs[0].get_legend_handles_labels()
            labels, ids = np.unique(labels, return_index=True)
            handles = np.array(handles)[ids]
            # set legend with its position
            fig.legend(handles, labels, fontsize=20, loc='center',
                       bbox_to_anchor=(0.52,0.52), ncol=2, frameon=False)
            plt.tight_layout(h_pad=3.5)
            if save:
                plt.savefig(f'{save_name}_{model}.png', dpi=400)
            plt.show()


if __name__ == '__main__':
    models = ['wscgan', 'multicgan']
    # prepare figure folder if not existing
    fig_fold = 'Figures'
    if not os.path.isdir(fig_fold):
        os.mkdir(fig_fold)
    diff_dynamic_cons(models, percent_to=False, plot=False, save=False,
                      exp_color='b', synthetic_color='r', spm_color='grey',
                      save_name=f'{fig_fold}/Dynamic_Inconsistency',
                      plot_individual=False, alpha=0.2)
