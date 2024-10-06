# -*- coding: utf-8 -*-
"""

Author:   Metin Bicer
email:    m.bicer19@imperial.ac.uk


reproduces results in Section 3.2

Means and standard deviations of conditions in the following papers:

    1. Nigg, B., Fisher, V., Ronsky, J., 1994. Gait characteristics as a
       function of age and gender. Gait & posture 2, 213-220.
       https://doi.org/10.1016/0966-6362(94)90106-6.

    2. Bruening, D.A., Frimenko, R.E., Goodyear, C.D., Bowden, D.R., Fullenkamp,
       A.M., 2015. Sex differences in whole body gait kinematics at preferred speeds.
       Gait & posture 41, 540-545. https://doi.org/10.1016/j.gaitpost.2014.12.011

    3. Weinhandl, J.T., Irmischer, B.S., Sievert, Z.A., 2017. Effects of gait
       speed of femoroacetabular joint forces. Applied Bionics and Biomechanics 2017.
       https://doi.org/10.1155/2017/6432969.

"""
import numpy as np
from scipy.stats import ttest_ind
import pandas as pd
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def print_nigg_angle(sagit_male, sagit_female):
    # roms
    #ax_i = {'Hip Flexion':0, 'Knee Flexion':1, 'Ankle Dorsiflexion':2}
    ax_i = {'Ankle Dorsiflexion': 2}
    for joint, i in ax_i.items():
        male_ank_rom = np.ptp(sagit_male[:,:,i], axis=1)
        female_ank_rom = np.ptp(sagit_female[:,:,i], axis=1)
        male_ank_rom_mean, male_ank_rom_std = male_ank_rom.mean(), male_ank_rom.std()
        female_ank_rom_mean, female_ank_rom_std = female_ank_rom.mean(), female_ank_rom.std()
        print(f'{joint} ROM:')
        print('-'*30)
        print(f'Males: {male_ank_rom_mean:.01f}±{male_ank_rom_std:.01f} deg')
        print(f'Females: {female_ank_rom_mean:.01f}±{female_ank_rom_std:.01f} deg')
        print(f'Diff: {np.abs(male_ank_rom_mean-female_ank_rom_mean):.1f} deg')
        print(f'p: p={ttest_ind(male_ank_rom, female_ank_rom).pvalue:.2f}')
        print('\n')


def print_nigg_force(sagit_male, sagit_female):
    print('First Peak:')
    print('-'*30)
    male_peak = sagit_male[:,:30,-1].max(axis=1)*100
    female_peak = sagit_female[:,:30,-1].max(axis=1)*100
    print(f'Males: {male_peak.mean():.01f}±{male_peak.std():.01f} %BW')
    print(f'Females: {female_peak.mean():.01f}±{female_peak.std():.01f} %BW')
    print(f'Difference: {np.abs(female_peak.mean()-male_peak.mean()):.01f} %BW')
    print(f'p: p={ttest_ind(male_peak, female_peak).pvalue:.2f}')
    print('\n')

    print('Second Peak:')
    print('-'*30)
    male_peak = sagit_male[:,40:60,-1].max(axis=1)*100
    female_peak = sagit_female[:,40:60,-1].max(axis=1)*100
    print(f'Males: {male_peak.mean():.01f}±{male_peak.std():.01f} %BW')
    print(f'Females: {female_peak.mean():.01f}±{female_peak.std():.01f} %BW')
    print(f'Diff: {np.abs(female_peak.mean()-male_peak.mean()):.01f} %BW')
    print(f'p: p={ttest_ind(male_peak, female_peak).pvalue:.2f}')
    print('\n\n')


def print_bruening(sagit_male, sagit_female, n_subjects_male, n_subjects_female):
    print(f'{"Angle":20s}{"Female ROM":<15s}{"Male ROM":<15s}{"Diff":<10s}{"p":<10s}')
    # roms
    joint_angles = ['Pelvic Tilt', 'Pelvic Obliquity', 'Pelvic Rotation',
                    'Hip Flexion', 'Hip Abduction', 'Hip Rotation',
                    'Knee Flexion', 'Ankle Dorsiflexion']
    for i, joint in enumerate(joint_angles):
        # get means of each subjects data
        male_angle = sagit_male[:,:,i]
        female_angle = sagit_female[:,:,i]
        male_rom = np.zeros(n_subjects_male)
        male_n_samp = int(male_angle.shape[0]/n_subjects_male)
        for sub in range(n_subjects_male):
            s = int(sub*male_n_samp)
            e = int(s+male_n_samp)
            male_rom[sub] = np.ptp(male_angle[s:e], axis=1).mean()

        female_rom = np.zeros(n_subjects_female)
        female_n_samp = int(female_angle.shape[0]/n_subjects_female)
        for sub in range(n_subjects_female):
            s = int(sub*female_n_samp)
            e = int(s+female_n_samp)
            female_rom[sub] = np.ptp(female_angle[s:e], axis=1).mean()

        male_rom_mean, male_rom_std = male_rom.mean(), male_rom.std()
        female_rom_mean, female_rom_std = female_rom.mean(), female_rom.std()
        pr = f'{joint:20s}'
        fem = f'{female_rom_mean:.01f}±{female_rom_std:.01f}'
        mal = f'{male_rom_mean:.01f}±{male_rom_std:.01f}'
        dif = f'{np.abs(male_rom_mean-female_rom_mean):.1f}'
        p = f'{ttest_ind(male_rom, female_rom).pvalue:.3f}'
        pr += f'{fem:<15s}{mal:<15s}{dif:<10s}{p:<10s}'
        print(pr)
    print('\n\n')


def anova(vals, names, subjs):
    data = pd.DataFrame({'Value': np.concatenate(vals),
                         'Group': np.tile(subjs, len(vals)),
                         'Change': np.repeat(names, len(vals[0]))})
    # one-way anova
    s, p_value = f_oneway(*vals)
    if p_value < 0.05:
        print(f'There is a significant difference between speeds (p={p_value:.3f} < 0.05).')
    else:
        print(f'There is no significant difference between speeds (p={p_value:.3f} >= 0.05).')

    print('\n')
    # posthoc tukey's test
    tukey = pairwise_tukeyhsd(endog=data['Value'],
                              groups=data['Change'],
                              alpha=0.05)
    print(tukey)
    print('\n\n')

# Nigg et al 1994
print('#'*37)
print('#'*10 + ' NIGG ET AL 1994 ' + '#'*10)
print('#'*37)
print('\n')
nigg_fold = 'Results/papers/nigg/'
male = np.load(f'{nigg_fold}male.npy')
female = np.load(f'{nigg_fold}female.npy')
sagit_male_1 = np.load(f'{nigg_fold}sagit_male_1.npy')
sagit_female_1 = np.load(f'{nigg_fold}sagit_female_1.npy')
sagit_male_4 = np.load(f'{nigg_fold}sagit_male_4.npy')
sagit_female_4 = np.load(f'{nigg_fold}sagit_female_4.npy')

print('AGE 20-39')
print('='*50)
print_nigg_angle(sagit_male_1, sagit_female_1)

print('AGE 70-79')
print('='*50)
print_nigg_angle(sagit_male_4, sagit_female_4)


print('Gender Differences (male vs female for all ages combined)')
print('='*50)
print_nigg_force(male, female)


# Bruening et al 2015
print('#'*42)
print('#'*10 + ' BRUENING ET AL 2015 ' + '#'*10)
print('#'*42)
print('\n')
bru_fold = 'Results/papers/bruening/'
n_subjects_male = 55
n_subjects_female = 36
ik_male_1 = np.load(f'{bru_fold}ik_male_1.npy')
ik_female_1 = np.load(f'{bru_fold}ik_female_1.npy')
print_bruening(ik_male_1, ik_female_1, n_subjects_male, n_subjects_female)

# Weinhandl et al 2015
print('#'*42)
print('#'*10 + ' WEINHANDL ET AL 2017 ' + '#'*10)
print('#'*42)
print('\n')
wei_fold = 'Results/papers/weinhandl/'
# load data
normal1 = np.load(f'{wei_fold}normal1.npy')
slow1 = np.load(f'{wei_fold}slow1.npy')
fast1 = np.load(f'{wei_fold}fast1.npy')
normal2 = np.load(f'{wei_fold}normal2.npy')
slow2 = np.load(f'{wei_fold}slow2.npy')
fast2 = np.load(f'{wei_fold}fast2.npy')
subjs = np.load(f'{wei_fold}subjs.npy')
# analysis
print('\nFirst Peak')
print('-'*20)
print(f'Slower: {slow1.mean():.2f}±{slow1.std():.2f}')
print(f'Normal: {normal1.mean():.2f}±{normal1.std():.2f}')
print(f'Faster: {fast1.mean():.2f}±{fast1.std():.2f}')
print('\n')
anova([slow1, normal1, fast1], ['Slow', 'Normal', 'Fast'], subjs)
print('2nd Peak')
print('-'*20)
print(f'Slower: {slow2.mean():.2f}±{slow2.std():.2f}')
print(f'Normal: {normal2.mean():.2f}±{normal2.std():.2f}')
print(f'Faster: {fast2.mean():.2f}±{fast2.std():.2f}')
print('\n')
anova([slow2, normal2, fast2], ['Slow', 'Normal', 'Fast'], subjs)
