# -*- coding: utf-8 -*-
"""

Author:   Metin Bicer
email:    m.bicer19@imperial.ac.uk

"""
import numpy as np
from scipy import signal, interpolate
import matplotlib.pyplot as plt
import spm1d
import pandas as pd


# define angles and momements
IK = ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
      'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
      'knee_angle_r', 'ankle_angle_r']
ID = [ik + '_moment' for ik in IK]

# define title templates
TITLES = ['Pelvic anterior (-) / posterior (+) tilt',
          'Pelvic obliquity up(+) / down (-)',
          'Pelvic internal (+) / external (-) rotation',
          'Hip flexion (+) / extension (-)',
          'Hip ab- (+) / adduction (-)',
          'Hip internal (+) / external (-) rotation',
          'Knee flexion (+) / extension (-)',
          'Ankle dorsi (+) / plantar (-) flexion']


def read_dataframes(data_df_files):
    '''
    read dataframe saved in a file
    or read and combine dataframes saved in multiple files

    Args:
        data_df_files (str, list): paths to files containing dataframes.

    Raises:
        ValueError: Files can only be saved as pickle or json.

    Returns:
        combined_df (pd.DataFrame): all data.

    '''
    # if a single file path is given
    if isinstance(data_df_files, str):
        data_df_files = [data_df_files]

    dataframes = []

    # loop through given files and read according to the extension (json or pickle)
    for path in data_df_files:
        ext = path.split('.')[1].lower()
        if ext == 'pickle':
            df = pd.read_pickle(path)
        elif ext == 'json':
            df = pd.read_json(path)
        else:
            raise ValueError('Use only json and pickle files')

        dataframes.append(df)
    # combine all dfs
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df


def np_rmse(t1, t2, normalised=True):
    '''
    calculate root mean squared error (rmse) for two nd arrays

    Args:
        t1 (np.ndarray): synthetic data.
        t2 (np.ndarray): experimental data.
        normalised (bool): whether or not to normalise using data t2 range.

    Returns:
        rmse (float): rmse (normalised if normalised=True).

    '''
    # t1 --> synthetic
    # t2 --> experimental
    if len(t1.shape) == 3:
        rmse = np.sqrt(np.mean(np.square(t1 - t2), axis=(0,1))).mean()
        return rmse
    elif len(t1.shape) == 2:
        if normalised:
            n = np.sqrt(np.mean(np.square(t1 - t2), axis=0))/np.ptp(t2,axis=0)
            rmse = np.mean(n)
            return rmse
        # not normalised
        rmse = np.sqrt(np.mean(np.square(t1 - t2), axis=0)).mean()
        return rmse
    elif len(t1.shape) == 1:#just a time array
        rmse = np.sqrt(np.mean(np.square(t1 - t2), axis=0))
        return rmse


def np_r_squared(t1, t2):
    '''
    calculate r for two nd arrays

    Args:
        t1 (np.ndarray): synthetic data.
        t2 (np.ndarray): experimental data.

    Returns:
        r (float or np.ndarray): r value. A single value is returned if arrays
                                are 1d or 2d. If 3d, first dimension is taken as
                                individual data points, so return np.ndarray of
                                length as the first dimension of t1/t2.

    '''
    # if t1.shape = #n_frames, #n_samples
    if len(t1.shape) == 2:
        r_val = 0
        n_samples = t1.shape[1]
        for t1_i, t2_i in zip(t1.T,t2.T):
            r_val += np.corrcoef(t1_i, t2_i)[0,1]
        r = r_val/n_samples
        return r
    elif len(t1.shape) == 1: # time arr
        r = np.corrcoef(t1, t2)[0,1]
        return r
    # else for all 1st dimension (each trial) return an array r values
    r = np.array([np_r_squared(t1_ii, t2_ii) for t1_ii, t2_ii in zip(t1, t2)])
    return r


def normalize_generated_frames(sample, no_frames):
    '''
    time normalisation for generated trials (used for gait cycle normalisation)

    Args:
        sample (np.ndarray): tensor to be used to normalise.
        no_frames (int): number of frames in the output.

    Returns:
        gc (np.ndarray): normalised data.

    '''
    # sample.shape = (3, 101, #included_names)
    # convert numpy if required
    if type(sample) != np.ndarray:
        sample = sample.numpy()
    gc = np.zeros((3, no_frames, sample.shape[-1]))
    for ax_i, ax_data in enumerate(sample):
        # for each included_name (last ax)
        for i, d in enumerate(ax_data.T):
            tck = interpolate.splrep(np.linspace(0, no_frames-1, len(d)),
                                     d, s=0)
            gc[ax_i,:,i] = interpolate.splev(np.linspace(0, no_frames-1,
                                                         num=no_frames),
                                        tck, der=0)
    return gc


def filterGRFTensor(tensor, sf, cutOff, aOrder, threshold, y_comp_i,
                    fp_to_check):
    '''
    filter generated grf tensor

    Args:
        tensor (np.ndarray): grf tensor.
        sf (float): sampling frequency.
        cutOff (float): cut off freq for the filter.
        aOrder (int): order of the filter.
        threshold (float): threshold below which the force is set to zero.
        y_comp_i (int): index of the y component of grf in the tensor.
        fp_to_check (int): index of the force plate to be checked in the tensor.

    Returns:
        filtered_tensor (np.ndarray): filtered grf tensor.

    '''
    # make sure tensor type
    tensor = np.array(tensor)
    # tensor.shape = #samples, 3, 101, #fp*3 (e.g., 780,3,101,3)
    is_single_trial = False if len(tensor.shape)==4 else True
    if is_single_trial: tensor = tensor[None,:,:,:]
    # filter only tensor!=0
    filtered_tensor = np.zeros_like(tensor)
    for i in range(3):
        filtered_tensor[:, i] = filterTensor(tensor[:,i], sf, cutOff, aOrder)
    # threshold forces
    # make everything zero if the y comp is lower than threshold after 50% of
    # find the vertical force in tensor[:,y_comp_i,:,fp_to_check]
    fy = filtered_tensor[:,y_comp_i,:,fp_to_check]
    # find toe off times
    mid_cycle = int(filtered_tensor.shape[2]/2)
    to_indices = np.argmax(fy[:, mid_cycle:] < threshold, axis=1) + mid_cycle
    # after this time everything is zero
    for t in range(filtered_tensor.shape[0]):
        filtered_tensor[t, :, to_indices[t]:, :] = 0
    if is_single_trial: filtered_tensor = filtered_tensor[0]
    return filtered_tensor


def filterTensor(tensor, sf=200.0, cutOff=20, aOrder=2):
    '''
    filter a sensor with a butterworth filter

    Args:
        tensor (np.ndarray): tensor to be filtered.
        sf (float): sampling frequency.
        cutOff (float): cut off freq for the filter.
        aOrder (int): order of the filter.

    Returns:
        tensor (np.ndarray): filtered tensor.

    '''
    # tensor.shape == n_samples, n_frames, n_features
    if len(tensor.shape) == 2:
        tensor = tensor[None,:,:]
    if len(tensor.shape) == 1:
        tensor = tensor[None,:,None]
    # sampling frequency of the measurements
    sf = float(sf)
    nyq = 0.5 * sf
    # Creation of the filter
    fc = cutOff / nyq # Cutoff frequency normalized

    # get the numerator and dominator of the signal function
    b, a = signal.butter(aOrder, fc, btype='lowpass')
    for sample_id, sample in enumerate(tensor):
        for feature_id in range(tensor.shape[-1]):
            tensor[sample_id, :, feature_id] = signal.filtfilt(b, a,
                                                               tensor[sample_id, :, feature_id])
    return tensor


def get_data(df, trial_no=1, subject_nos='all',
             plot_feature='ankle_angle_r', plot_field='ik_gc',
             feature_names_field='ik_names', walking_ids=[],
             divide_by=None):
    '''
    gets data from dataframe following the arguments

    Args:
        df (pd.DataFrame): dataframe containing data.
        trial_no (int, str, list): gets only specified trials.
        subject_nos (int, str, list): gets only specified subjects.
        plot_feature (str): name of the data to be read.
        plot_field (str): name of the field that contains feature.
        feature_names_field (str): name of the df column containing feature list.
        walking_ids (list): gets only specified walking trials.
        divide_by (float): divides the obtained data by this number.

    Returns:
        df_clip (pd.DataFrame): clipped dataframe for the given arguments.
        results (np.ndarray): obtained data.
        means (np.ndarray): data mean.
        plus_std (np.ndarray): mean+std.
        minus_std (np.ndarray): mean-std.

    '''

    # subject nos
    if type(subject_nos)==str:
        if subject_nos == 'all':
            subject_nos = np.unique(df['subject'].values)
        else:
            subject_nos = [subject_nos]
    elif type(subject_nos)==int:
        subject_nos = [subject_nos]
    # trial nos
    if type(trial_no) == int:
        trial_no = [trial_no]
    if type(trial_no) == str:
        if trial_no == 'all':
            trial_no = np.unique(df['trial'].values)
        else:
            trial_no = [trial_no]
    divider = 1
    divider_g = 1
    if divide_by == 'weight':
        divider_g = 9.81
    # feature names for the plotted file (identifiers in the file either ik, id or grf)
    #df = df.dropna()
    #df = df.reset_index()
    # dataframe for the given trial and subjects
    df_clip = df[(df['trial'].isin(trial_no)) & (df['subject'].isin(subject_nos))]
    if len(walking_ids) > 0:
        df_clip = df_clip[df_clip['walking_id'].isin(walking_ids)]
    #df_clip = df_clip.dropna()
    df_clip = df_clip.reset_index(drop=True)
    if len(df_clip) == 0:
        return [None]*5
    # if trial exists
    feature_names = df_clip[feature_names_field].values[0]
    # shape=(#trials, #frames, #coords)
    if divide_by is not False:
        divider = df_clip['mass'].values*divider_g
    results_all = np.stack(df_clip[plot_field].values/divider,
                           axis=0)
    if len(results_all.shape) == 4:
        results_all = results_all.squeeze(axis=1)
    #if np.any(np.array(results_all.shape)==1): results_all.squeeze(axis=1)
    results = results_all[:,:,feature_names==plot_feature].squeeze()
    means = results.mean(axis=0)
    plus_std = means + results.std(axis=0)
    minus_std = means - results.std(axis=0)
    return df_clip, results, means, plus_std, minus_std


def get_real_data(df, trial_no, subject_nos, features, plot_field,
                  feature_names_field, divide_by=False,  df_field=None,
                  llim=None, ulim=None):
    '''
    get real data from the saved dataframe for given subjects with conditions

    Args:
        df (pd.DataFrame): dataframe containing data.
        trial_no (int, str, list): gets only specified trials.
        subject_nos (int, str, list): gets only specified subjects.
        features (list): features to be read.
        plot_field (str): name of the field that contains feature.
        feature_names_field (str): name of the df column containing feature list.
        divide_by (float): divides the obtained data by this number.
        df_field (str): the field name (condition) of df used to clip with given limits.
        llim (float): lower limit of the condition (df_field).
        ulim (float): upper limit of the condition (df_field).

    Returns:
        results (np.ndarray): data as asked by the arguments.
                              results.shape = #examples, #frames, #features

    '''
    results = []
    if df_field is not None:
        df = df[(df[df_field]>=llim)&(df[df_field]<=ulim)].reset_index()
    for feature in features:
        _, res_feature, *_ = get_data(df, trial_no, subject_nos, feature,
                                      plot_field, feature_names_field, [],
                                      divide_by)
        results.append(res_feature[:,:,None])
    results = np.concatenate(results, axis=-1)
    # results.shape = #ex, #frames, #features
    return results


def get_synthetic_grfm(generated_tensor, force_i=15):
    '''
    get ground reaction forces and the y moment

    Args:
        generated_tensor (np.ndarray): data tensor (may contain markers, IK and GRFs).
        force_i (int): index of the grf data in the tensor.

    Returns:
        synthetic_grfm (np.ndarray): grf and moment tensor.

    '''
    #generated_tensor.shape == #samples, 3, 101, 26

    # get forces
    synthetic_forces = np.concatenate([generated_tensor[:,i,:,force_i][:,:,None] for i in range(3)],axis=2)

    # get y moment
    synthetic_moment_y = generated_tensor[:,1,:,force_i+2][:,:,None]

    # combine forces and the y moment
    synthetic_grfm = np.concatenate([synthetic_forces, synthetic_moment_y],
                                    axis=2)
    return synthetic_grfm


def spmInverse(data_1, data_2, analysis_type='ik', plot_toeoff=None, spm_toeoff=None,
               plot=False, save=False, sagittalZoom=False, axs=None,
               data_1_color='b', data_2_color='r', spm_color='grey',
               save_name='SPM', plot_individual=False, alpha=0.2,
               return_comp_diffs=False, plot_sig=True):
    '''
    spm analysis on results of inverse tools (IK and ID)

    Args:
        data_1 (tensor): shape=#ex,#frames,#features.
        data_2 (tensor): shape=#ex,#frames,#features.
        analysis_type (str): 'ik' or 'id'.
        plot_toeoff (bool): plots toe off timings if true.
        spm_toeoff (int): runs spm analysis until given toeoff frame. This is
                          required to avoid errors due to 0s after toeoff.
        plot (bool): plot if true.
        save (bool): save the figure if true.
        sagittalZoom (bool): can make zoom into sagittal plane quantities if true.
        axs (matplotlib.axes): can be drawn into a given axs (None to draw new).
        data_1_color (str): color to data_1 lines.
        data_2_color (str): color to data_2 lines.
        spm_color (str): background shade color for spm differences.
        save_name (str): save figure with this name.
        plot_individual (bool): plot individual trials if True, else mean,std.
        alpha (float): opacity value for plotting.
        return_comp_diffs (bool): returns differences for each component if True, else mean.
        plot_sig (bool): background shading for significant differences if True.

    Returns:
        totalAverageSignificant (float): significant differences averaged over
                                         all trials and all variables involved,
                                         as a percentage of the gait cycle.
        axs (matplotlib.axes): can be used to draw upon.

    '''
    # alpha can be a list for data_1 and data_2
    if not isinstance(alpha, list):
        alpha = [alpha, alpha]
    no_frames = data_1.shape[1]
    # toe off timings
    if spm_toeoff is None:
        spm_toeoff = no_frames
    # define the followings for plots (change them if id)
    plotIndices = [0, 1, 2, 3, 4, 5, 6, 9]
    coord_names = IK
    subplotTitles = TITLES
    ylabel = 'Angle [deg]'
    if analysis_type == 'id':
        coord_names = ID
        ylabel = 'Moment [Nm]'

    # toe-off for spm
    spm_toeoff = int(spm_toeoff)
    # total number of significant region lengths
    significant = [0 for _ in coord_names]
    # plot
    if plot and axs is None:
        fig, axs = plt.subplots(4, 3, figsize=(16, 9))
        axs = axs.flatten()
    # for each coord
    coord_i = 0
    for plotIdx, plotName in zip(plotIndices, subplotTitles):
        # spm1d t-test
        data_1_coord = data_1[:, :, coord_i]
        data_2_coord = data_2[:, :, coord_i]
        t  = spm1d.stats.ttest2(data_1_coord[:, :spm_toeoff],
                                data_2_coord[:, :spm_toeoff],
                                equal_var=False)
        ti = t.inference(alpha=0.05, two_tailed=True, interp=True)
        # get clusters and add significant differences
        for cluster in ti.clusters:
            significant[coord_i] += cluster.extent
        # increase idx
        coord_i = coord_i+1
        # plot
        if plot:
            ax = axs[plotIdx]
            # set title and font sizes of ticks
            ax.set_title(plotName, fontsize=20)
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.tick_params(axis='both', which='minor', labelsize=12)
            # plot zero line
            ax.axhline(y=0.0, color='k', linestyle='-', linewidth=0.5)
            # plot toeoff
            if plot_toeoff is not None:
                ax.axvline(x=plot_toeoff, color='k', linestyle='--', linewidth=0.5)
            if plot_individual:
                # plot individual curves
                ax.plot(data_2_coord.T, color=data_2_color,
                        label='Synthetic', alpha=alpha[1])
                ax.plot(data_1_coord.T, color=data_1_color,
                        label='Experimental', alpha=alpha[0])
            else:
                # plot mean sd
                spm1d.plot.plot_mean_sd(data_1_coord, ax=ax, linecolor=data_1_color,
                                        facecolor=data_1_color, edgecolor=data_1_color,
                                        label='Experimental')
                spm1d.plot.plot_mean_sd(data_2_coord, ax=ax, linecolor=data_2_color,
                                        facecolor=data_2_color, edgecolor=data_2_color,
                                        label='Synthetic')
            # indicate significant difference regions
            if plot_sig:
                for cluster in ti.clusters:
                    ax.axvspan(cluster.endpoints[0], cluster.endpoints[1],
                               alpha=0.3, color=spm_color)
            # set font sizes of ticks
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.tick_params(axis='both', which='minor', labelsize=12)
            ax.set_xlim([0, 100])
    # after plotting all
    if plot:
        # set axs off subplots that are not used
        for i in [7, 8, 10, 11]: axs[i].axis('off')
        # set y labels for first column
        for ax in axs[0::3]: ax.set_ylabel(ylabel, fontsize=25)
        # set x label for columns
        for i in [4, 5, 9]: axs[i].set_xlabel('Gait Cycle [\%]', fontsize=25)
        # get legend handles and labels
        handles, labels = axs[0].get_legend_handles_labels()
        labels, ids = np.unique(labels, return_index=True)
        handles = np.array(handles)[ids]
        # set legend with its position
        axs[7].legend(handles, labels, loc=1, fontsize=20,
                      bbox_to_anchor=(1.5,0.5), frameon=False)
        plt.tight_layout(pad=0.5, w_pad=-5)
        if save:
            plt.savefig(f'{save_name}.png', dpi=400)
        plt.show()
    # significant percentage
    if return_comp_diffs:
        return significant, axs
    # else return total average
    totalAverageSignificant = 100*sum(significant)/(no_frames*len(coord_names))
    return totalAverageSignificant, axs


def spmGRF(real_data, synthetic_data, channel_i=1, plot=False, save=False,
           real_color='b', synthetic_color='r', spm_color='grey', normalized=False,
           plot_individual=False, save_name='SPM_GRF', alpha=0.2,
           return_comp_diffs=False, plot_sig=True):
    '''


    Args:
        real_data (tensor): shape=#ex,#frames,#features.
        synthetic_data (tensor): shape=#ex,#frames,#features.
        channel_i (int): channel index to find toeoffs (y component).
        plot (bool): plot if true.
        save (bool): save the figure if true.
        real_color (str): color to real data lines.
        synthetic_color (str): color to synthetic data lines.
        spm_color (str): background shade color for spm differences.
        normalized (bool): normalise data if True.
        plot_individual (bool): plot individual trials if True, else mean,std.
        save_name (str): save figure with this name.
        alpha (float): opacity value for plotting.
        return_comp_diffs (bool): returns differences for each component if True, else mean.
        plot_sig (bool): background shading for significant differences if True.

    Returns:
        totalAverageSignificant (float): significant differences averaged over
                                         all trials and all variables involved,
                                         as a percentage of the gait cycle.
        toeoff (float): average toeoff times to be used in other analyses.
        axs (matplotlib.axes): can be used to draw upon.

    '''
    # alpha can be a list for real_data and synthetic_data
    if not isinstance(alpha, list):
        alpha = [alpha, alpha]
    axs = None
    if plot:
        fig, axs = plt.subplots(2, 2, figsize=(16, 9))
        axs = axs.flatten()
    # total number of significant region lengths
    significant = [0 for _ in range(real_data.shape[-1])]
    # find toe-offs average (for plotting and grf spm until there)
    toeoff = np.mean([findGeneratedToeOffFrame(r, channel_i) for r in real_data])
    for plot_idx in range(real_data.shape[-1]):
        real_data_i = real_data[:, :, plot_idx]
        synthetic_data_i = synthetic_data[:, :, plot_idx]
        # spm1d t-test
        t  = spm1d.stats.ttest2(real_data_i[:, :int(toeoff)],
                                synthetic_data_i[:, :int(toeoff)],
                                equal_var=False)
        ti = t.inference(alpha=0.05, two_tailed=True, interp=True)
        # get clusters and add significant differences
        for cluster in ti.clusters:
            significant[plot_idx] += cluster.extent
        if plot:
            ax = axs[plot_idx]
            # plot individual trials
            if plot_individual:
                # plot individual curves
                ax.plot(synthetic_data_i.T, color=synthetic_color,
                        label='Synthetic', alpha=alpha[1])
                ax.plot(real_data_i.T, color=real_color,
                        label='Experimental', alpha=alpha[0])
            # plot mean sd
            else:
                spm1d.plot.plot_mean_sd(real_data_i, ax=ax, linecolor=real_color,
                                        facecolor=real_color, edgecolor=real_color,
                                        label='Experimental')
                spm1d.plot.plot_mean_sd(synthetic_data_i, ax=ax, linecolor=synthetic_color,
                                        facecolor=synthetic_color, edgecolor=synthetic_color,
                                        label='Synthetic')
            if plot_sig:
                for cluster in ti.clusters:
                    ax.axvspan(cluster.endpoints[0], cluster.endpoints[1],
                               alpha=0.3, color=spm_color)
    # after plotting all
    ylabels = [r'$F_{x}$ [N]', r'$F_{y}$ [N]', r'$F_{z}$ [N]', r'$T_{y}$ [Nm]']
    #TODO ylims = [[-150,150],[-100,1000],[-80,40],[-3,4]]
    if normalized:
        ylabels = [l.split(' ')[0] + ' [BW]' for l in ylabels]
    if plot:
        # set y labels for first column
        for ax, label in zip(axs, ylabels):
            # plot zero line
            ax.axhline(y=0.0, color='k', linestyle='-', linewidth=0.5)
            # plot toeoff
            ax.axvline(x=toeoff, color='k', linestyle='--', linewidth=0.5)
            ax.set_ylabel(label, fontsize=25)
            # set font sizes of ticks
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.tick_params(axis='both', which='minor', labelsize=12)
            ax.set_xlim([0, 100])
            #TODO ax.set_ylim(ylim)
        for ax in axs[2:]: ax.set_xlabel('Gait Cycle [\%]', fontsize=25)
        # get legend handles and labels
        handles, labels = axs[0].get_legend_handles_labels()
        labels, ids = np.unique(labels, return_index=True)
        handles = np.array(handles)[ids]
        fig.legend(handles, labels, fontsize=20, loc='center',
                   bbox_to_anchor=(0.52,0.05), ncol=2, frameon=False)
        plt.tight_layout(pad=2, w_pad=1)
        if save:
            plt.savefig(f'{save_name}.png', dpi=400)
        plt.show()
    # significant percentage
    if return_comp_diffs:
        return significant, toeoff, axs
    # else return total average
    totalAverageSignificant = 100*sum(significant)/(np.prod(real_data.shape[1:]))
    return totalAverageSignificant, toeoff, axs


def findGeneratedToeOffFrame(grf, channel_i):
    '''
    finds toe-off frame (in gait cycle)

    Parameters
    ----------
    grf : dict
        dictionary containing force plate data.
    fp_no : int
        force plate number.

    Returns
    -------
    toe_off: int
        index of toe-off event in the gait cycle.

    '''
    # grf.shape = #frames, #features
    # find values very close to 0
    close = np.argwhere(grf[:, channel_i]<1e-5).flatten()
    # return the first frame (greater than 40)
    toe_off = close[np.argmax(close>40)]
    return toe_off


def metrics_m_std(arrays, ids):
    '''
    calculate mean and std for performance metrics

    Args:
        arrays (list): contains np.ndarrays.
        ids (TYPE): to calculate mean and std.

    Returns:
        m (float): mean.
        s (float): standard deviation.

    '''
    arr = np.hstack([arrays[i] for i in ids])
    m, s = np.mean(arr), np.std(arr)
    return m, s


def t2test(model_ik, model_grf, real_ik, real_grf, n_samples, toeoff):
    '''
    hotellings_paired test from spm1d between two sets of grf and ik data
    prints the results

    Args:
        model_ik (np.ndarray): synthetic ik.
        model_grf (np.ndarray): synthetic grf.
        real_ik (np.ndarray): experimental ik.
        real_grf (np.ndarray): experimental grf.
        n_samples (int): number of samples in model_ik.
        toeoff (float): average toeoff frame. The test between grfs was carried
                        out until toeoff to eliminate errors caused by comparing
                        arrays of zeros (after toeoff)

    '''
    n_real_data = real_ik.shape[0]
    if n_samples>1:
        # get means
        model_ik = model_ik.reshape((n_real_data, n_samples, 101, 8)).mean(axis=1)
        model_grf = model_grf.reshape((n_real_data, n_samples, 101, 4)).mean(axis=1)
    # ik
    T2 = spm1d.stats.hotellings_paired(real_ik, model_ik)
    T2i = T2.inference(0.05)
    plt.figure()
    T2i.plot()
    # sig in ik
    significant_ik = 'Significant IK differences between '
    for cluster in T2i.clusters:
        ends = cluster.endpoints
        significant_ik += f'"{ends[0]:.1f}-{ends[1]:.1f}" '
    print(significant_ik)
    # grf
    rand1 = np.random.uniform(0, 1e-10, real_grf[:,toeoff:].shape)
    rand2 = np.random.uniform(0, 1e-10, model_grf[:,toeoff:].shape)
    real_grf = np.concatenate([real_grf[:,:toeoff],real_grf[:,toeoff:]+rand1],
                              axis=1)
    model_grf = np.concatenate([model_grf[:,:toeoff],model_grf[:,toeoff:]+rand2],
                               axis=1)
    T2 = spm1d.stats.hotellings_paired(real_grf, model_grf)
    T2i = T2.inference(0.05)
    plt.figure()
    T2i.plot()
    # sig in grf
    significant_grf = 'Significant GRF differences between '
    for cluster in T2i.clusters:
        ends = cluster.endpoints
        significant_grf += f'"{ends[0]:.1f}-{ends[1]:.1f}" '
    print(significant_grf)
