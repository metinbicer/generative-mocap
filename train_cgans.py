# -*- coding: utf-8 -*-
"""

Author:   Metin Bicer
email:    m.bicer19@imperial.ac.uk

"""
import torch.nn as nn
import torch
from torch.autograd import Variable
from datasets import mocapDataLoader
from transformation import HeightChannelScale, labelScale
import numpy as np
import itertools
import pickle as pkl
import time
import os
from itertools import product
import random


### DEFAULTS ###

# Headers for the markers, grfs and ik angles
MARKER_NAMES = np.array(['L_IAS', 'L_IPS', 'R_IPS', 'R_IAS', 'R_FTC',
                         'R_FLE', 'R_FME', 'R_FAX', 'R_TTC', 'R_FAL',
                         'R_TAM', 'R_FCC', 'R_FM1', 'R_FM2', 'R_FM5'])
# the followings are headers only for the x comp of f, m and cop
# because of the dataframe structure
GRF_NAMES = np.array(['force','point','moment'])
IK_NAMES = np.array(['pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
                     'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
                     'knee_angle_r', 'ankle_angle_r'])

# respective column names (check data.pickle)
VALUE_COLS = ['marker_gc', 'grf_3d_gc', 'ik_gc']

# respective names that contain headers (check data.pickle)
# depending on these names, the data from the DATA_DF_FILE will be read
NAMES_COLS = ['marker_names', 'grf_names_3d', 'ik_names']

# json or pickle files that contain the data (check for the structure)
DATA_DF_FILE = ['data/data_1.pickle', 'data/data_2.pickle']

# the column that will be used as a continous label (can be set to None)
LABEL_COL_CONT = ['age']

# the column that will be used as a discrete label (can be set to None)
LABEL_COL_DISCR = None

# excluded subjects from training models (only use for testing)
EXCLUDED_SUBJECTS = [2014001, 2014003, 2015042]

# hyperparameters for model and training
Z_DIM = 20
HIDDEN_DIM = 512
N_EPOCHS = 3000
LR = 0.002
BATHC_SIZE = 128
DISPLAY_STEP = 250
N_SAMPLES = 10

# define labels at which n_samples of data will be generated
# e.g., generate data at ages starting from 15 to 71 in increments of 1.
LABEL_CONTD_LIMS = [[15,71,1]]
LABEL_DISCR_LIMS = None

# to save the results
MODEL_NAME = 'acgan'

# use gpu in training if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setSeed(seed=0):
    """
    Set all seeds for reproducibility

    Parameters
    ----------
    seed (int): seed to be passed to all modules random generators

    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # usig gpu
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def getRandomLatentVec(mu, logvar, device):
    """
    generates a random vector with the given mean and log variability.

    Args:
        mu (torch.Tensor): mean of the latent vector.
        logvar (torch.Tensor): log variability of the latent vector.
        device (torch.device): type of device holding tensors (cpu/gpu).

    Returns:
        z (torch.Tensor): latent vector.

    """
    Tensor = torch.cuda.FloatTensor if device==torch.device('cuda') else torch.FloatTensor
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.shape[0], mu.shape[1]))))
    z = sampled_z * std + mu
    z.to(device)
    return z


class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, data_shape):
        super(Encoder, self).__init__()
        """
        encoder of the network that outputs the mean and logvar of the encoded
        latent vector.

        Args:
            z_dim (int): length of the latent vector.
            hidden_dim (int): hidden layer size.
            data_shape (list): shape of an example.

        """
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(data_shape)), hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mu = nn.Linear(hidden_dim, z_dim)
        self.logvar = nn.Linear(hidden_dim, z_dim)

    def forward(self, data, device):
        """
        forward pass of the network.

        Args:
            data (torch.Tensor): input to the net.
            device (torch.device): type of device holding tensors (cpu/gpu).

        Returns:
            z (torch.Tensor): output of the network.

        """
        # data.shape =
        data = data.view(data.shape[0], -1)
        x = self.model(data)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = getRandomLatentVec(mu, logvar, device)
        return z


class Decoder(nn.Module):
    def __init__(self, z_dim_and_label, hidden_dim, data_shape):
        super(Decoder, self).__init__()
        """
        decoder of the network that decodes the given vector to the original
        data shape.

        Args:
            z_dim_and_label (int): total size of the latent vector and label(s).
            hidden_dim (int): hidden layer size.
            data_shape (list/tuple): shape of an example.

        """
        self.data_shape = data_shape
        self.model = nn.Sequential(
            nn.Linear(z_dim_and_label, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, int(np.prod(data_shape))),
            nn.Sigmoid(),
        )

    def forward(self, z, label):
        """
        forward pass of the network.

        Args:
            z (torch.Tensor): latent vector (noise/encoded).
            label (torch.Tensor): label(s) corresponding to z OR label(s) for
            which sytnhetic data will be generated.

        Returns:
            data (torch.Tensor): output of the network.

        """
        z_and_label = torch.cat((z.float(), label.float()), dim=1)
        decoded = self.model(z_and_label)
        data = decoded.view(decoded.shape[0], *self.data_shape)
        return data


class Discriminator(nn.Module):
    def __init__(self, z_dim_and_label, hidden_dim):
        super(Discriminator, self).__init__()
        """
        network discriminates whether the given latent vector belongs to the
        real or generated samples

        Args:
            z_dim_and_label (int): total size of the latent vector and label(s).
            hidden_dim (int): hidden layer size.

        """
        self.model = nn.Sequential(
            nn.Linear(z_dim_and_label, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, int(hidden_dim / 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(int(hidden_dim / 2), 1),
            nn.Sigmoid(),
        )

    def forward(self, z, label):
        """
        forward pass of the network.

        Args:
            z (torch.Tensor): latent vector (noise/encoded).
            label (torch.Tensor): label(s) corresponding to z OR label(s) for
            which sytnhetic data was generated.

        Returns:
            out (torch.Tensor): Probability of given z and its label being real.

        """
        z_and_label = torch.cat((z.float(), label.float()), dim=1)
        out = self.model(z_and_label)
        return out


def weights_init(m):
    """
    applies xavier uniform weight initialization to the linear layers.

    Args:
        m (torch.nn.Linear): network's linear layer.

    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.01)


def getMinMax_Label(labels):
    return torch.min(labels, dim=0).values, torch.max(labels, dim=0).values


def getMinMax_Tensor(data):
    """
    calculates the min and max of a tensor in two different ways:
        1) channel-wise
        2) column-wise

    Parameters
    ----------
    data : tensor of size (batch_size)x(channel)x..

    Returns
    -------
    channelMin : min along channels
    channelMax : max along channels
    columnMin  : min along columns
    columnMax  : max along columns

    """
    # channel min-max
    channelMin = torch.amin(data, dim=(0, 2, 3))
    channelMax = torch.amax(data, dim=(0, 2, 3))

    # height min-max (for each position of each marker)
    columnMin = torch.amin(data, dim=(0, 2))
    columnMax = torch.amax(data, dim=(0, 2))
    return channelMin, channelMax, columnMin, columnMax


def train(data_df_file, excluded_subjects, input_col, label_col_contd, label_col_discr,
          z_dim, hidden_dim, n_epochs, lr, batch_size, device, display_step):
    """
    training networks (by args) for the data, labels (by args).

    Args:
        data_df_file (str): path to the dataframe (json/pickle) containing the
        real data.
        excluded_subjects (list): subjects can be excluded from training GANs.
        If using all subject's data, this should be set to an empty list.
        input_col (str): dataframe's column name(s) containing data to be generated.
        label_cols (str): dataframe's column name(s) containing label(s).
        z_dim (int): latent vector size.
        hidden_dim (int): hidden layer size of networks.
        n_epochs (int): number of training epochs.
        lr (float): learning rate.
        batch_size (int): batch size for training.
        continuous_conditional (bool): Conditionas/labels are continuous if True.
        device (torch.device): type of device holding tensors/models (cpu/gpu).
        display_step (int): printing metrics at every certain steps.

    Returns:
        models (list): contains trained encoder, decoder and discriminator.
        losses (list): contains generator and discriminator losses.
        transformations (list): contains transformations used for inputs and
        conditions/labels. Includes both way transformations.

    """
    # get dataloader
    dataloader = mocapDataLoader(data_df_file, excluded_subjects,
                                 input_col, label_col_contd, label_col_discr,
                                 batch_size='full', train=False)
    data, label = next(iter(dataloader))
    num_conditions = 0 if len(label)==0 else label.shape[1]
    # get data statistics (mins and maxs) for transformation
    channelMin, channelMax, columnMin, columnMax = getMinMax_Tensor(data)
    # get min-max of labels (for transformation)
    label_old_min, label_old_max = getMinMax_Label(label)
    # back transformation for data
    back_transform_input = HeightChannelScale(oldMin=columnMin*0, oldMax=columnMax*0+1,
                                             newMin=columnMin, newMax=columnMax)
    # load data with applying transformations (both inputs and targets, or labels)
    # both inputs and targets (labels) are transformed into 0-1 range
    transform_inputs = HeightChannelScale(oldMin=columnMin, oldMax=columnMax)
    # back transformation for labels
    # get min-max of labels (for transformation)
    label_old_min, label_old_max = None, None
    if label_col_contd is not None:
        label_old_min, label_old_max = getMinMax_Label(label[:,:len(label_col_contd)])
    # back transformation for data
    back_transform_input = HeightChannelScale(oldMin=columnMin*0, oldMax=columnMax*0+1,
                                             newMin=columnMin, newMax=columnMax)
    # load data with applying transformations (both inputs and targets, or labels)
    # both inputs and targets (labels) are transformed into 0-1 range
    transform_inputs = HeightChannelScale(oldMin=columnMin, oldMax=columnMax)
    # back transformation for labels
    if label_col_contd is not None:
        back_transform_label = labelScale(oldMin=label_old_min*0, oldMax=label_old_max*0+1,
                                          newMin=label_old_min, newMax=label_old_max)
        transform_labels = labelScale(oldMin=label_old_min, oldMax=label_old_max)
    else:
        back_transform_label = None
        transform_labels = None

    dataloader = mocapDataLoader(data_df_file, excluded_subjects,
                                 input_col, label_col_contd, label_col_discr,
                                 batch_size, True,#train=True
                                 transform_inputs, transform_labels)
    # one batch of data
    real, labels = next(iter(dataloader))
    real_ex_shape = real.shape[1:]
    # criterion
    # Use binary cross-entropy loss for discriminator
    bce_loss = torch.nn.BCELoss()
    # l2 norm for generator
    mse_loss = torch.nn.MSELoss()
    # models and optimizers
    encoder = Encoder(z_dim, hidden_dim, real_ex_shape)
    decoder = Decoder(z_dim + num_conditions, hidden_dim, real_ex_shape)
    disc = Discriminator(z_dim + num_conditions, hidden_dim)
    # send to device
    encoder.to(device)
    decoder.to(device)
    disc.to(device)
    # apply weight initialization
    encoder = encoder.apply(weights_init)
    decoder = decoder.apply(weights_init)
    disc = disc.apply(weights_init)
    # Optimizers
    optimizer_G = torch.optim.Adam(itertools.chain(encoder.parameters(),
                                                   decoder.parameters()),
                                   lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
    Tensor = torch.cuda.FloatTensor if device.type=='cuda' else torch.FloatTensor
    generator_losses = []
    discriminator_losses = []
    start_time = time.time()
    i = 0
    for epoch in range(n_epochs):
        # Dataloader returns the batches and the labels
        for real, labels in dataloader:
            # Adversarial ground truths
            valid = Tensor(real.shape[0], 1).fill_(1.0)
            fake = Tensor(real.shape[0], 1).fill_(0.0)
            cur_batch_size = len(real)
            real = real.to(device)
            labels = labels.to(device)

            # -----------------
            #    Train Generator
            # -----------------

            optimizer_G.zero_grad()
            encoded_data = encoder(real, device)
            decoded_data = decoder(encoded_data, labels)
            # mse loss (reconstruction) + bce loss for encoding latent vectors
            gen_loss = 0.999 * mse_loss(decoded_data, real)
            # the following is used to terminate the training and return the
            # latest results if nans appear in the disc outputs
            if torch.any(torch.Tensor.isnan(disc(encoded_data, labels))):
                losses = [generator_losses, 0]
                models = [encoder, decoder, disc]
                transformations = [back_transform_input, transform_inputs,
                                   back_transform_label, transform_labels]
                return models, losses, transformations
            gen_loss += 0.001 * bce_loss(disc(encoded_data, labels), valid)
            gen_loss.backward()
            optimizer_G.step()

            # ---------------------
            #    Train Generator
            # ---------------------

            # ---------------------
            #    Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()
            # Sample noise
            z = torch.randn(cur_batch_size, z_dim).to(device)
            z_labels = (torch.rand_like(labels)).to(device)#.type(torch.LongTensor).to(device)
            # Measure discriminator's ability to classify real from generated samples
            # fool discriminator
            real_loss = bce_loss(disc(z, z_labels), valid)
            fake_loss = bce_loss(disc(encoded_data.detach(), labels), fake)
            disc_loss = 0.5 * (real_loss + fake_loss)
            disc_loss.backward()
            optimizer_D.step()

            # ---------------------
            #    Train Discriminator
            # ---------------------

            # Keep track of the average losses
            discriminator_losses += [disc_loss.item()]
            generator_losses += [gen_loss.item()]
            i += 1
            if i % display_step == 0 and i > 0:
                gen_mean = sum(generator_losses[-display_step:]) / display_step
                disc_mean = sum(discriminator_losses[-display_step:]) / display_step
                print(f"Epoch {epoch+1}, step {i}: Generator loss: {gen_mean}, discriminator loss: {disc_mean}")
    print(f"Training finished in {(time.time()-start_time)/60:.02f} mins")
    # vars to be returned
    losses = [generator_losses, discriminator_losses]
    models = [encoder, decoder, disc]
    transformations = [back_transform_input, transform_inputs,
                       back_transform_label, transform_labels]
    return models, losses, transformations


def save_data_training(decoder, z_dim, n_samples, label_contd, label_discr,
                       back_transform_input, transform_labels, save_file,
                       epoch, seed):
    """
    generates and saves synthetic data during training

    Args:
        decoder (Decoder): decoder/generator model.
        z_dim (int): latent vector dimension.
        n_samples (int): number of synthetic samples to be generated.
        label_contd (TYPE): continous labels.
        label_discr (TYPE): discrete labels.
        back_transform_input (torchvision.transforms or transformation): transformations
        used to convert generated data to original domain.
        transform_labels (torchvision.transforms or transformation): transformations
        used to normalize the labels.
        save_file (str): path to the file to save generated data.
        epoch (int): to append the saved filename.
        seed (int): to append the saved filename.

    Returns:
        None.

    """
    if epoch is not None:
        save_file = save_file + '_' + str(epoch)
    generateds = []
    if label_contd is None and label_discr is None:
        generateds = generate_with_labels(decoder, z_dim, n_samples,
                                         label_contd, label_discr,
                                         back_transform_input,
                                         transform_labels, seed)
    elif label_contd is not None and label_discr is None:
        for label_c in label_contd:
            # generate trials and tranform them
            generateds.append(generate_with_labels(decoder, z_dim, n_samples,
                                                  label_c, label_discr,
                                                  back_transform_input,
                                                  transform_labels, seed))
    elif label_contd is None and label_discr is not None:
        for label_d in label_discr:
            # generate trials and tranform them
            generateds.append(generate_with_labels(decoder, z_dim, n_samples,
                                                  label_contd, label_d,
                                                  back_transform_input,
                                                  transform_labels, seed))
    else:
        # multi conditions (both cont and discr)
        for label_c, label_d in zip(label_contd, label_discr):
            # generate trials and tranform them
            generateds.append(generate_with_labels(decoder, z_dim, n_samples,
                                                  label_c, label_d,
                                                  back_transform_input,
                                                  transform_labels, seed))
    # saving the generated data
    #  if too many samples
    if len(generateds) > 4000:
        for i in range(len(generateds)):
            if i % 2000 == 0:
                save_file = save_file+'_'+str(i)+'.pkl'
                if i+2000<len(generateds):
                    with open(save_file, 'wb') as f:
                        pkl.dump(generateds[i:i+2000], f, protocol=pkl.HIGHEST_PROTOCOL)
                else:
                    with open(save_file, 'wb') as f:
                        pkl.dump(generateds[i:], f, protocol=pkl.HIGHEST_PROTOCOL)
    else:
        with open(save_file+'.pkl', 'wb') as f:
            pkl.dump(generateds, f, protocol=pkl.HIGHEST_PROTOCOL)


def generate_with_labels(gen, z_dim, n_samples, label_contd, label_discr,
                        back_transform_input=None, transform_labels=None,
                        seed=None):
    """
    generate data with the given generator and labels.

    Args:
        gen (torch.Decoder or Model): generator model.
        z_dim (int): latent vector dimension.
        n_samples (int): number of synthetic samples to be generated.
        label_contd (torch.Tensor): continous label(s) for synthetic data.
        label_discr (torch.Tensor): discrete label(s) for synthetic data.
        back_transform_input (torchvision.transforms or transformation):
        transformations used to convert generated data to original domain.
        transform_labels (torchvision.transforms or transformation, optional):
        transformations used to normalize the labels. Defaults to None
        (for continuous).

    Returns:
        generated_data (torch.Tensor): synthetic data.

    """
    if seed is not None:
        setSeed(seed)
    # before passing to the model, make sure all tensors are on the same device
    device_model = next(gen.parameters()).device
    z = torch.randn(n_samples, z_dim).to(device_model)
    z_labels = []
    # if cont labels given
    if label_contd is not None:
        if transform_labels is None:
            z_labels = label_contd#*torch.ones((n_samples, 1))
        else:
            z_labels = transform_labels(label_contd)
    # if discrete labels are given, directly cat to the z_labels (transformed)
    if label_discr is not None:
        if len(z_labels) == 0:
            z_labels = torch.Tensor(z_labels)
        if len(label_discr.shape) == 0:
            label_discr = label_discr.view(1)
        z_labels = torch.cat((z_labels, label_discr))
    # reshape z_labels
    if len(z_labels) > 0:
        z_labels = z_labels*torch.ones((n_samples, 1))
    else:
        z_labels = torch.Tensor(z_labels)
    z_labels = z_labels.to(device_model)
    # eval mode
    gen.eval()
    generated_data = gen(z, z_labels).cpu().detach()
    if back_transform_input is not None:
        for i in range(n_samples):
            generated_data[i] = back_transform_input(generated_data[i])
    return generated_data


def train_cgan(marker_names=MARKER_NAMES, grf_names=GRF_NAMES,
               ik_names=IK_NAMES, value_cols=VALUE_COLS, names_cols=NAMES_COLS,
               label_col_contd=LABEL_COL_CONT, label_col_discr=LABEL_COL_DISCR,
               data_df_file=DATA_DF_FILE, excluded_subjects=EXCLUDED_SUBJECTS,
               z_dim=Z_DIM, hidden_dim=HIDDEN_DIM, n_epochs=N_EPOCHS, lr=LR,
               batch_size=BATHC_SIZE, display_step=DISPLAY_STEP,
               n_samples=N_SAMPLES, label_contd_lims=LABEL_CONTD_LIMS,
               label_discr_lims=LABEL_DISCR_LIMS, model_name=MODEL_NAME,
               device=DEVICE, seed=0):
    """
    all single conditional models can be trained and saved using this function

    Args:
        marker_names (list, optional): name of markers in the order as they saved.
        Defaults to MARKER_NAMES.
        grf_names (list, optional): grf headers. Defaults to GRF_NAMES.
        ik_names (list, optional): ik segment/joint angle names. Defaults to IK_NAMES.
        value_cols (list, optional): names of the "data_df_file" column names
        containing the data. Defaults to VALUE_COLS.
        names_cols (list, optional): controls which features are to be generated.
        Defaults to NAMES_COLS.
        label_cols (lsit, optional): define conditions with column names from
        "data_df_file". Defaults to LABEL_COLS.
        data_df_file (pandas.df, optional): dataframe stroing data. May include
        training and test data. Defaults to DATA_DF_FILE.
        excluded_subjects (TYPE, optional): Exclude subjects, if any, from training.
        Defaults to EXCLUDED_SUBJECTS.
        z_dim (int, optional): Latent vector dimension. Defaults to Z_DIM.
        hidden_dim (int, optional): hidden layer size. Defaults to HIDDEN_DIM.
        n_epochs (int, optional): number of epochs. Defaults to N_EPOCHS.
        lr (float, optional): learning rate. Defaults to LR.
        batch_size (int, optional): batch size. Defaults to BATHC_SIZE.
        display_step (int, optional): for printing losses in training.
        Defaults to DISPLAY_STEP.
        n_samples (int, optional): specify how many samples will be generated
        for each given condition below with "label_lower" and "label_upper"
        and "label_incr". Defaults to N_SAMPLES.
        label_lower (float, optional): lower bound of the condition range that
        will be used to generate synthetic data after the training. Defaults to
        LABEL_LOWER.
        label_upper (float, optional): upper bound of the condition range that
        will be used to generate synthetic data after the training. Defaults to
        LABEL_UPPER.
        label_icnr (float, optional): increaments of condition range. Defaults
        to LABEL_INCR.
        model_name (TYPE, optional): Give a name for saving results.
        Defaults to MODEL_NAME.
        device (TYPE, optional): CPU or GPU for training. Defaults to DEVICE.

    Returns:
        None.

    """
    # define input column names
    included_names = [marker_names, grf_names, ik_names]
    input_col = {'value_cols':value_cols, 'names_cols':names_cols,
                 'included_names':included_names}
    # train
    out = train(data_df_file, excluded_subjects, input_col, label_col_contd,
                label_col_discr, z_dim, hidden_dim, n_epochs, lr,
                batch_size, device, display_step)
    encoder, decoder, disc = out[0]
    gen_loss, disc_loss = out[1]
    transformations = out[-1]
    back_transform_input, transform_inputs = transformations[:2]
    back_transform_label, transform_labels = transformations[2:]
    # generate and save synthetic data
    # define conditions using the ranges given
    # continous conditions
    if label_col_contd is None:
        label_contd = None
    else:
        label_contd = torch.tensor(list(product(*[torch.arange(*lims)
                                                  for lims in label_contd_lims])))
    # discrete conditions
    if label_col_discr is None:
        label_discr = None
    else:
        label_discr = torch.tensor(list(product(*[torch.arange(*lims)
                                                  for lims in label_discr_lims])))
    # combine if both are given
    if label_contd is not None and label_discr is not None:
        # create the new tensor
        expanded = []
        for d in label_discr:
            # expand label_contd by appending each discrete condition to each row
            expanded_row = torch.cat((label_contd, d.expand(label_contd.shape[0], 1)), dim=1)
            expanded.append(expanded_row)
        # Stack expanded rows to get the final tensor
        all_labels = torch.cat(expanded, dim=0)
        # get conditions seperately
        label_contd = all_labels[:, :label_contd.shape[1]]
        label_discr = all_labels[:, label_contd.shape[1]:]
    # save data
    # create folder if not existing with model name
    res_fold = f'Results/{model_name}/'
    if not os.path.isdir(res_fold):
        os.mkdir(res_fold)
    # generate and save
    save_file = f'{res_fold}synthetic_data_after_training'
    save_data_training(decoder, z_dim, n_samples, label_contd, label_discr,
                       back_transform_input, transform_labels, save_file,
                       None, seed)
    # save back transformation, label transformations and generator
    torch.save(decoder.state_dict(),f'{res_fold}generator.pt')
    with open(f'{res_fold}back_transform_input.pkl', 'wb') as f:
        pkl.dump(back_transform_input, f)
    with open(f'{res_fold}transform_labels.pkl', 'wb') as f:
        pkl.dump(transform_labels, f)


### EXAMPLES

'''

def acGAN():
    """
    Trains a generative model only conditioned with age
    (A-cGAN)

    """
    marker_names = np.array(['L_IAS', 'L_IPS', 'R_IPS', 'R_IAS', 'R_FTC',
                             'R_FLE', 'R_FME', 'R_FAX', 'R_TTC', 'R_FAL',
                             'R_TAM', 'R_FCC', 'R_FM1', 'R_FM2', 'R_FM5'])
    grf_names = np.array(['force','point','moment'])
    ik_names = np.array(['pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
                         'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
                         'knee_angle_r', 'ankle_angle_r'])
    # define col names
    value_cols = ['marker_gc', 'grf_3d_gc', 'ik_gc']
    names_cols = ['marker_names', 'grf_names_3d', 'ik_names']
    # labels in label_cols
    label_col_contd = ['age']
    label_col_discr = None
    # file
    data_df_file = ['data/data_1.pickle', 'data/data_2.pickle']
    excluded_subjects = [2014001, 2014003, 2015042]
    # hyperparams
    z_dim = 20
    hidden_dim = 512
    n_epochs = 3000
    lr = 0.002
    batch_size = 128
    display_step = 250
    n_samples = 10
    # define ages at which n_samples of data will be generated
    label_contd_lims = [[15,71,1]]
    label_discr_lims = None
    model_name = 'acgan'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cgan(marker_names, grf_names, ik_names, value_cols,
               names_cols, label_col_contd, label_col_discr,
               data_df_file, excluded_subjects, z_dim, hidden_dim,
               n_epochs, lr, batch_size, display_step, n_samples,
               label_contd_lims, label_discr_lims, model_name, device, 0)


def mcGAN():
    """
    Generates only conditioned with mass
    (M-cGAN)

    Returns
    -------
    Saves n_samples of synthetic trials generated for each class

    """
    marker_names = np.array(['L_IAS', 'L_IPS', 'R_IPS', 'R_IAS', 'R_FTC',
                             'R_FLE', 'R_FME', 'R_FAX', 'R_TTC', 'R_FAL',
                             'R_TAM', 'R_FCC', 'R_FM1', 'R_FM2', 'R_FM5'])
    grf_names = np.array(['force','point','moment'])
    ik_names = np.array(['pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
                         'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
                         'knee_angle_r', 'ankle_angle_r'])
    # define col names
    value_cols = ['marker_gc', 'grf_3d_gc', 'ik_gc']
    names_cols = ['marker_names', 'grf_names_3d', 'ik_names']
    # labels in label_cols
    label_col_contd = ['mass']
    label_col_discr = None
    # file
    data_df_file = ['data/data_1.pickle', 'data/data_2.pickle']
    excluded_subjects = [2014001, 2014003, 2015042]
    # hyperparams
    z_dim = 20
    hidden_dim = 512
    n_epochs = 3000
    lr = 0.002
    batch_size = 128
    display_step = 250
    n_samples = 10
    # define ages at which n_samples of data will be generated
    label_contd_lims = [[45,101,5]]
    label_discr_lims = None
    model_name = 'mcgan'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cgan(marker_names, grf_names, ik_names, value_cols,
               names_cols, label_col_contd, label_col_discr,
               data_df_file, excluded_subjects, z_dim, hidden_dim,
               n_epochs, lr, batch_size, display_step, n_samples,
               label_contd_lims, label_discr_lims, model_name, device, 0)


def llcGAN():
    """
    Generates only conditioned with leg lengths [cm] as measured in static shot
    (LL-cGAN)

    Returns
    -------
    Saves n_samples of synthetic trials generated for each class

    """
    marker_names = np.array(['L_IAS', 'L_IPS', 'R_IPS', 'R_IAS', 'R_FTC',
                             'R_FLE', 'R_FME', 'R_FAX', 'R_TTC', 'R_FAL',
                             'R_TAM', 'R_FCC', 'R_FM1', 'R_FM2', 'R_FM5'])
    grf_names = np.array(['force','point','moment'])
    ik_names = np.array(['pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
                         'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
                         'knee_angle_r', 'ankle_angle_r'])
    # define col names
    value_cols = ['marker_gc', 'grf_3d_gc', 'ik_gc']
    names_cols = ['marker_names', 'grf_names_3d', 'ik_names']
    # labels in label_cols
    label_col_contd = ['leglength_static']
    label_col_discr = None
    # file
    data_df_file = ['data/data_1.pickle', 'data/data_2.pickle']
    excluded_subjects = [2014001, 2014003, 2015042]
    # hyperparams
    z_dim = 20
    hidden_dim = 512
    n_epochs = 3000
    lr = 0.002
    batch_size = 128
    display_step = 250
    n_samples = 10
    # define ages at which n_samples of data will be generated
    label_contd_lims = [[785, 1101, 3]]
    label_discr_lims = None
    model_name = 'llcgan'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cgan(marker_names, grf_names, ik_names, value_cols,
               names_cols, label_col_contd, label_col_discr,
               data_df_file, excluded_subjects, z_dim, hidden_dim,
               n_epochs, lr, batch_size, display_step, n_samples,
               label_contd_lims, label_discr_lims, model_name, device, 0)


def wscGAN():
    """
    Generates only conditioned with walking speed [m/s]
    (WS-cGAN)

    Returns
    -------
    marker_names = np.array(['L_IAS', 'L_IPS', 'R_IPS', 'R_IAS', 'R_FTC',
                             'R_FLE', 'R_FME', 'R_FAX', 'R_TTC', 'R_FAL',
                             'R_TAM', 'R_FCC', 'R_FM1', 'R_FM2', 'R_FM5'])
    grf_names = np.array(['force','point','moment'])
    ik_names = np.array(['pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
                         'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
                         'knee_angle_r', 'ankle_angle_r'])
    # define col names
    value_cols = ['marker_gc', 'grf_3d_gc', 'ik_gc']
    names_cols = ['marker_names', 'grf_names_3d', 'ik_names']
    # labels in label_cols
    label_col_contd = ['walking_speed']
    label_col_discr = None
    # file
    data_df_file = ['data/data_1.pickle', 'data/data_2.pickle']
    excluded_subjects = [2014001, 2014003, 2015042]
    # hyperparams
    z_dim = 20
    hidden_dim = 512
    n_epochs = 3000
    lr = 0.002
    batch_size = 128
    display_step = 250
    n_samples = 10
    # define ages at which n_samples of data will be generated
    label_contd_lims = [[0.16, 2.41, 0.02]]
    label_discr_lims = None
    model_name = 'wscgan'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cgan(marker_names, grf_names, ik_names, value_cols,
               names_cols, label_col_contd, label_col_discr,
               data_df_file, excluded_subjects, z_dim, hidden_dim,
               n_epochs, lr, batch_size, display_step, n_samples,
               label_contd_lims, label_discr_lims, model_name, device, 0)


def gcGAN():
    """
    Generates only conditioned with gender (discrete labels)
    (G-cGAN)

    Returns
    -------
    Saves n_samples of synthetic trials generated for each class

    """
    marker_names = np.array(['L_IAS', 'L_IPS', 'R_IPS', 'R_IAS', 'R_FTC',
                             'R_FLE', 'R_FME', 'R_FAX', 'R_TTC', 'R_FAL',
                             'R_TAM', 'R_FCC', 'R_FM1', 'R_FM2', 'R_FM5'])
    grf_names = np.array(['force','point','moment'])
    ik_names = np.array(['pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
                         'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
                         'knee_angle_r', 'ankle_angle_r'])
    # define col names
    value_cols = ['marker_gc', 'grf_3d_gc', 'ik_gc']
    names_cols = ['marker_names', 'grf_names_3d', 'ik_names']
    # labels in label_cols
    label_col_contd = None
    label_col_discr = ['gender_int']
    # file
    data_df_file = ['data/data_1.pickle', 'data/data_2.pickle']
    excluded_subjects = [2014001, 2014003, 2015042]
    # hyperparams
    z_dim = 20
    hidden_dim = 512
    n_epochs = 3000
    lr = 0.002
    batch_size = 128
    display_step = 250
    n_samples = 10
    # define ages at which n_samples of data will be generated
    label_contd_lims = None
    label_discr_lims = [[0, 1.01, 1]]
    model_name = 'gcgan'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cgan(marker_names, grf_names, ik_names, value_cols,
               names_cols, label_col_contd, label_col_discr,
               data_df_file, excluded_subjects, z_dim, hidden_dim,
               n_epochs, lr, batch_size, display_step, n_samples,
               label_contd_lims, label_discr_lims, model_name, device, 0)


def multicGAN():
    """
    Generates marker trajectories with multi conditions
    (Multi-cGAN)

    Returns
    -------
    Saves n_samples of synthetic trials

    """
    marker_names = np.array(['L_IAS', 'L_IPS', 'R_IPS', 'R_IAS', 'R_FTC',
                             'R_FLE', 'R_FME', 'R_FAX', 'R_TTC', 'R_FAL',
                             'R_TAM', 'R_FCC', 'R_FM1', 'R_FM2', 'R_FM5'])
    grf_names = np.array(['force','point','moment'])
    ik_names = np.array(['pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
                         'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
                         'knee_angle_r', 'ankle_angle_r'])
    # define col names
    value_cols = ['marker_gc', 'grf_3d_gc', 'ik_gc']
    names_cols = ['marker_names', 'grf_names_3d', 'ik_names']
    # labels in label_cols
    label_col_contd = ['age', 'mass', 'leglength_static', 'walking_speed']
    label_col_discr = ['gender_int']
    # file
    data_df_file = ['data/data_1.pickle', 'data/data_2.pickle']
    excluded_subjects = [2014001, 2014003, 2015042]
    # hyperparams
    z_dim = 20
    hidden_dim = 512
    n_epochs = 4000
    lr = 0.002
    batch_size = 128
    display_step = 250
    n_samples = 1
    # define ages at which n_samples of data will be generated
    label_contd_lims = [[15,71,10], [45,101,20],[785,1101,50],[0.16,2.41,0.5]]
    label_discr_lims = [[0, 1.01, 1]]
    model_name = 'multicgan'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cgan(marker_names, grf_names, ik_names, value_cols,
               names_cols, label_col_contd, label_col_discr,
               data_df_file, excluded_subjects, z_dim, hidden_dim,
               n_epochs, lr, batch_size, display_step, n_samples,
               label_contd_lims, label_discr_lims, model_name, device, 0)

'''
