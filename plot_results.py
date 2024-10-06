# -*- coding: utf-8 -*-
"""

Author:   Metin Bicer
email:    m.bicer19@imperial.ac.uk

"""
import numpy as np
import matplotlib.pyplot as plt
import os


# define title templates
TITLES = {'pelvis_tilt': 'Pelvic anterior (-) / posterior (+) tilt',
          'pelvis_list': 'Pelvic obliquity up(+) / down (-)',
          'pelvis_rotation': 'Pelvic internal (+) / external (-) rotation',
          'hip_flexion_r': 'Hip flexion (+) / extension (-)',
          'hip_adduction_r': 'Hip ab- (+) / adduction (-)',
          'hip_rotation_r': 'Hip internal (+) / external (-) rotation',
          'knee_angle_r': 'Knee flexion (+) / extension (-)',
          'ankle_angle_r': 'Ankle dorsi (+) / plantar (-) flexion',
          'ground_force_1_vx': 'Vertical Ground Reaction Force'}


def plot_dim_speed(trial_tensor, plot_feature, included_names, labels_tensor,
                   unique_labels, toe_offs, direction='y', ylims=None,
                   is_legend=False, colors=None):
    '''
    plots means

    Parameters
    ----------
    trial_tensor : np.ndarray
        generated data of shape == #labels,#sample,#direction,#frames,#included_names.
        #direction = 3 and #frames = 101
    plot_feature : list
        any sublist of included names.
    labels_tensor : np.ndarray
        labels of each generated data (e.g., ['Very Slow', 'Slow' ....].
    included_names : np.ndarray
        the header names for the generated data.
    toe_offs : np.ndarray
        integers containing toe-off for every generated trial.
    direction : str or int.
        'x' (0), 'y' (1) or 'z' (2) defining direction. This value does not
        matter for the ik joint angles.

    Returns
    -------
    None.

    '''
    # find axs from direction
    if type(direction)==str:
        if direction == 'x':
            direction = 0
        elif direction == 'y':
            direction = 1
        else:
            direction = 2
    # define colors (for now just 5 shades of blue for speeds)
    if colors is None:
        colors = ['darkslateblue', 'royalblue', 'dodgerblue', 'deepskyblue', 'cyan']
        colors = ['darkred', 'firebrick', 'red', 'indianred', 'lightcoral']
    # instantiate figure and axes
    nrows = int(np.ceil(len(plot_feature)/3))
    ncols = int(np.ceil(len(plot_feature)/nrows))
    fig, axs = plt.subplots(nrows, ncols)
    if type(axs) != np.ndarray: axs = np.array(axs)
    axs = axs.flatten()
    # figure sizes
    if nrows == 1 and ncols > 1:
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(38, 125, 1752, 535)
    elif nrows > 1 and ncols > 1:
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(494, 193, 1007, 734)
        #mngr.window.showMaximized()
    # gather mean results for every plot_feature
    mean_trials = np.zeros((len(plot_feature), len(unique_labels), 101))
    mean_toeoffs = np.zeros(len(unique_labels))
    # start plotting
    i = 0
    for ax, feature in zip(axs, plot_feature):
        for label_i, (label, color) in enumerate(zip(unique_labels, colors)):
            trials = trial_tensor[labels_tensor==label,
                                  direction, :,
                                  included_names==feature]
            mean_trials[i, label_i] = trials.mean(axis=0)
            ax.plot(mean_trials[i, label_i], c=color, label=label)
            # plot toe-offs
            mean_toeoffs[label_i] = toe_offs[labels_tensor==label].mean()
            ax.axvline(mean_toeoffs[label_i], c=color, lw=0.5)
        # set ax titles, lims and labels
        if feature in TITLES.keys():
            ax.set_title(TITLES[feature], fontsize=26)
        else:
            ax.set_title(feature, fontsize=26)
        if 'force' in feature:
            ax.set_ylabel('Force [N]', fontsize=26)
        # else is kinematics
        else:
            ax.set_ylabel('Angle [Deg]', fontsize=26)
        ax.set_xlabel('Gait Cycle [\%]', fontsize=26)
        ax.set_xlim([0,100])
        if ylims is not None:
            if len(np.array(ylims).shape) > 1:
                # then different lim options is given for axes
                ax.set_ylim(ylims[i])
            else:
                ax.set_ylim(ylims)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=20)
        i = i+1
        if is_legend:
            ax.legend(prop={'size': 12})
    # if any axes left unused, clear
    if i < len(axs):
        for ax in axs[i:]: ax.set_visible(False)
    fig.tight_layout()
    return mean_trials, mean_toeoffs


def save_current_fig(name, fold, dpi=300, fig_format='png'):
    # saves current figure
    # if name has the format, ignore fig_format
    if '.' in name:
        fig_format = ''
        fig_name = os.path.join(fold, name)
    else:
        fig_name = os.path.join(fold, name) + '.' + fig_format
    plt.savefig(fig_name, dpi=dpi)
