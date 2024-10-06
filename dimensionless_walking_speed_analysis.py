# -*- coding: utf-8 -*-
"""

Author:   Metin Bicer
email:    m.bicer19@imperial.ac.uk

reproduces the dimensionless walking speed analysis in the paper
uses Multi-cGAN because this is the only generative model containing both the
leg lenths and walking speeds for all synthetic and experimental subjects/trials
for other models it estimates leng length

Refer to the following publications for calculation of dimensionless walking
speed:
1. Hof, A. L. (1996). Scaling gait data to body size. Gait & posture,
    3(4), 222-223.
2. Schwartz, M. H., Rozumalski, A., & Trost, J. P. (2008).
    The effect of walking speed on the gait of typically developing children.
    Journal of biomechanics, 41(8), 1639-1650.

"""
##### DIMENSIONLESS WALKING SPEED ANALYSIS

import numpy as np
from plot_results import plot_dim_speed, save_current_fig
from utils import read_dataframes, np_r_squared, np_rmse
import torch
import os
from matplotlib import rc
rc('text', usetex=True)


# gen model name
model = 'multicgan'

# defaults
# gravitational acc
g = 9.81
# define colors for plotting
exp_colors = ['darkslateblue', 'royalblue', 'dodgerblue', 'deepskyblue', 'cyan']
synt_colors = ['darkred', 'firebrick', 'indianred', 'lightcoral', 'mistyrose']
# axs limits
ylims_sagittal = [[-30,40], [0,80],[-20,20]]


ylims_lower_limb = [[-15,15], [-10,10], [-15,15],
                    [-40,60],[-15,15], [-30,30],
                    [0,130],[-40,30]]
if 'multi' in model:
    ylims_force = [0, 1.5]
else:
    ylims_force = [0, 1000]

fig_fold = os.path.join(os.getcwd(), 'Figures')
# create figures fold
if not os.path.isdir(fig_fold):
    os.mkdir(fig_fold)


###############################################################################
############## EXPERIMENTAL DATA
############## Load experimental data and calculate vstar
###############################################################################

df = read_dataframes(['data/data_1.pickle', 'data/data_2.pickle'])
excluded_subjects = [2014001, 2014003, 2015042]
df = df[~df.subject.isin(excluded_subjects)].reset_index(drop=True)
bw_exp = df['mass'].values*9.81
# to calculate dimensionless speed
speeds_star = df[df['trial']=='C4'].walking_speed.values
leg_lengths_star = df[df['trial']=='C4'].leglength_static.values/1000 # to m
vstar_free = speeds_star/np.sqrt(g*leg_lengths_star)
# mean and std
vstar_free_m = vstar_free.mean()
vstar_free_s = vstar_free.std()

# ik results and headers
ik_exp = np.array([ik for ik in df.ik_gc])
ik_names_exp = np.array(df.ik_names.values[0])
speeds_exp = df.walking_speed.values
leg_lengths_exp = df.leglength_static.values/1000 # convert to m
dimensionless_speeds_exp = np.zeros_like(speeds_exp)
label_speeds_exp = np.zeros_like(speeds_exp, dtype=object)
for i, (speed, leg_length) in enumerate(zip(speeds_exp, leg_lengths_exp)):
    # dimensionless speed
    trial_vstar = speed/np.sqrt(g*leg_length)
    dimensionless_speeds_exp[i] = trial_vstar
    # the conditions are from Schwartz's paper
    if trial_vstar <= (vstar_free_m-3*vstar_free_s):
        label_speeds_exp[i] = 'Very Slow'
    elif (trial_vstar > (vstar_free_m-3*vstar_free_s)) and (trial_vstar <= (vstar_free_m-1*vstar_free_s)):
        label_speeds_exp[i] = 'Slow'
    elif (trial_vstar > (vstar_free_m-1*vstar_free_s)) and (trial_vstar <= (vstar_free_m+1*vstar_free_s)):
        label_speeds_exp[i] = 'Free'
    elif (trial_vstar > (vstar_free_m+1*vstar_free_s)) and (trial_vstar <= (vstar_free_m+3*vstar_free_s)):
        label_speeds_exp[i] = 'Fast'
    elif (trial_vstar > (vstar_free_m+3*vstar_free_s)):
        label_speeds_exp[i] = 'Very Fast'

# get the toe-off events from the exp data
grf_names_exp = np.array(df.grf_names_2d[0])
grf_verticals_exp = np.array([np.array(grf)[:,grf_names_exp=='ground_force_1_vy']
                                for grf in df.grf_2d_gc]).squeeze()
if 'multi' in model:
    # if multi conditional model, then normalise forces with bw
    grf_verticals_exp /= bw_exp[:,None]
# find max indices for each vertical force
# these will be used to find when the force becomes zero (toe-off)
max_idxs_exp = np.argmax(grf_verticals_exp, axis=1)
# threshold is 2.5% of the max force
toe_offs_exp = np.array([np.argwhere(grf[idx:]<grf.max()*0.025)[0]+idx
                           for idx, grf in zip(max_idxs_exp, grf_verticals_exp)])
# unique labels appended to label_speeds_exp
unique_labels_exp = np.array(['Very Slow', 'Slow', 'Free', 'Fast', 'Very Fast'])
# plot sagittal joint kinematics
plot_feature = np.array(['hip_flexion_r', 'knee_angle_r', 'ankle_angle_r'])
mean_sagittal_exp, mean_toeoffs_exp = plot_dim_speed(ik_exp, plot_feature,
                                                     ik_names_exp,
                                                     label_speeds_exp,
                                                     unique_labels_exp,
                                                     toe_offs_exp, direction=0,
                                                     ylims=ylims_sagittal,
                                                     is_legend=False, colors=exp_colors)
# save figure
save_current_fig('ik_sagittal_experimental', fig_fold)


# plot vertical grfs
plot_feature = np.array(['ground_force_1_vx'])
mean_grf_y_exp, mean_toeoffs_exp = plot_dim_speed(grf_verticals_exp[:,None,:,None],
                                                  plot_feature, plot_feature,
                                                  label_speeds_exp, unique_labels_exp,
                                                  toe_offs_exp, ylims=ylims_force,
                                                  direction=0, is_legend=True,
                                                  colors=exp_colors)
# save figure
save_current_fig('grf_vertical_experimental', fig_fold)


###############################################################################
############## GENERATED TRIALS
############## Synthetic data
###############################################################################
# names
marker_names = np.array(['L_IPS', 'L_IAS', 'R_IPS', 'R_IAS', 'R_FTC',
                         'R_FLE', 'R_FME', 'R_FAX', 'R_TTC', 'R_FAL',
                         'R_TAM', 'R_FCC', 'R_FM1', 'R_FM2', 'R_FM5'])
grf_names = np.array(['ground_force_1_vx', 'ground_force_1_px',
                      'ground_moment_1_mx'])
ik_names = np.array(['pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
                     'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
                     'knee_angle_r', 'ankle_angle_r'])
included_names = np.concatenate((marker_names, grf_names, ik_names))

ik_names_df = np.array(df.ik_names[0])

######### Synthetic data ########
# synthetic data was saved in two parts
synthetic_data = [np.load(f'Results/{model}/dimensionless/synthetic_data_{i}.npy')
                  for i in range(1,3)]
synthetic_data = np.concatenate(synthetic_data, axis=0)

# labels to shape=#samples
n_samples = int(synthetic_data.shape[0]/ik_exp.shape[0])
multi_labels = torch.Tensor(df[['age', 'mass', 'leglength_static',
                                'walking_speed', 'gender_int']].values)
if 'multi' in model:
    speed_labels_untransformed = multi_labels[:,3]
    leg_lens = np.array([leg for leg in multi_labels[:, 2] for i in range(n_samples)])
    all_speeds = np.array([speed for speed in speed_labels_untransformed for i in range(n_samples)])
else:
    # only walking speeds (leg lengths will be estimated)
    multi_labels = torch.Tensor(df['walking_speed'].values)
    speed_labels_untransformed = multi_labels
# prepare arrays for dimensionless_speed, labels and leg lengths
dimensionless_speeds = np.zeros_like(all_speeds)
label_speeds = np.zeros_like(all_speeds, dtype=object)
leg_l_trials = np.zeros_like(all_speeds)
for i, (trial, speed) in enumerate(zip(synthetic_data, all_speeds)):
    if 'multi' in model:
        leg_l_trial = leg_lens[i]/1000
        leg_l_trials[i] = leg_l_trial
    else:
        # leg length is estimated at the 26th frame of gait cycle data
        leg_l_trial = (trial[1,26,included_names=='R_IAS'][0]-trial[1,26,included_names=='R_FCC'][0])/1000
        leg_l_trials[i] = leg_l_trial
    # dimensionless speed
    trial_vstar = speed/np.sqrt(g*leg_l_trial)
    dimensionless_speeds[i] = trial_vstar
    # the conditions are from Schwartz's paper
    if trial_vstar <= (vstar_free_m-3*vstar_free_s):
        label_speeds[i] = 'Very Slow'
    elif (trial_vstar > (vstar_free_m-3*vstar_free_s)) and (trial_vstar <= (vstar_free_m-1*vstar_free_s)):
        label_speeds[i] = 'Slow'
    elif (trial_vstar > (vstar_free_m-1*vstar_free_s)) and (trial_vstar <= (vstar_free_m+1*vstar_free_s)):
        label_speeds[i] = 'Free'
    elif (trial_vstar > (vstar_free_m+1*vstar_free_s)) and (trial_vstar <= (vstar_free_m+3*vstar_free_s)):
        label_speeds[i] = 'Fast'
    elif (trial_vstar > (vstar_free_m+3*vstar_free_s)):
        label_speeds[i] = 'Very Fast'

# get the toe-off events from the generated samples
grf_verticals = synthetic_data[:,1,:,included_names=='ground_force_1_vx'].squeeze()
for grf_i, grf in enumerate(grf_verticals):
    idxs = np.where(grf<20)[0]
    grf_verticals[grf_i, idxs[idxs>40][0]:] = 0
# find max indices for each vertical force
# these will be used to find when the force becomes zero (toe-off)
max_idxs = np.argmax(grf_verticals, axis=1)
# threshold is 2.5% of the max force
toe_offs = np.array([np.argwhere(grf[idx:]<grf.max()*0.025)[0]+idx
                     for idx, grf in zip(max_idxs, grf_verticals)])
if 'multi' in model:
    grf_verticals /= np.array([speed for speed in multi_labels[:, 1]
                               for i in range(n_samples)])[:,None]*9.81
# unique labels appended to label_speeds
unique_labels = np.array(['Very Slow', 'Slow', 'Free', 'Fast', 'Very Fast'])
# plot sagittal joint kinematics
plot_feature = np.array(['hip_flexion_r', 'knee_angle_r', 'ankle_angle_r'])
mean_sagittal, mean_toeoffs = plot_dim_speed(synthetic_data[:,:1],
                                             plot_feature, included_names,
                                             label_speeds, unique_labels,
                                             toe_offs, direction=0,
                                             ylims=ylims_sagittal,
                                             is_legend=False, colors=synt_colors)
# save figure
save_current_fig('ik_sagittal_generated', fig_fold)

# plot all lower limb kinematics
plot_feature = np.array(['pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
                         'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
                         'knee_angle_r', 'ankle_angle_r'])
_ = plot_dim_speed(synthetic_data, plot_feature, included_names,
                               label_speeds, unique_labels, toe_offs,
                               direction=0, ylims=ylims_lower_limb, is_legend=False)
# save figure
save_current_fig('ik_generated', fig_fold)

# plot vertical grfs
plot_feature = np.array(['ground_force_1_vx'])
mean_grf_y, mean_toeoffs = plot_dim_speed(grf_verticals[:,None,:,None],
                                          plot_feature, plot_feature,
                                          label_speeds, unique_labels,
                                          toe_offs, direction=0,
                                          ylims=ylims_force, is_legend=True,
                                          colors=synt_colors)
# save figure
save_current_fig('grf_vertical_generated', fig_fold)


################ PRINTING DIFFERENCES BETWEEN EXP AND SYNTHETIC

def print_r_rmse(data_real, data_synt, plot_feature, feature,
                 unique_labels_exp, label, label_t, rmse_pre=1,
                 r_pre=1):
    r_tr = data_real[plot_feature==feature,
                            unique_labels_exp==label].squeeze()
    r_sy = data_synt[plot_feature==feature,
                      unique_labels_exp==label].squeeze()
    rmse = np_rmse(r_tr,r_sy)
    r_coef = np_r_squared(r_tr,r_sy)

    print(f'{label_t:<25s}{rmse:<15.{rmse_pre}f}{r_coef:<15.{r_pre}f}')


plot_feature = np.array(['hip_flexion_r', 'knee_angle_r', 'ankle_angle_r'])

### kinematics
features = ['hip_flexion_r', 'knee_angle_r', 'ankle_angle_r']
for feature in features:
    print(f'{feature:<25s}{"RMSE":<15s}{"r":<15s}')
    for label in unique_labels_exp:
        label_t = '-'.join(label.split(' '))
        print_r_rmse(mean_sagittal_exp, mean_sagittal, plot_feature, feature,
                     unique_labels_exp, label, label_t)

### vertical grfs
# print some stats (initial, max and final means for each speed)
plot_feature = np.array(['ground_force_1_vx'])
for feature in plot_feature:
    print(f'{feature:<25s}{"RMSE":<15s}{"r":<15s}')
    for label in unique_labels_exp:
        label_t = '-'.join(label.split(' '))
        print_r_rmse(mean_grf_y_exp, mean_grf_y, plot_feature, feature,
                     unique_labels_exp, label, label_t, rmse_pre=2)
