# -*- coding: utf-8 -*-

# %% Directory check cell

import os
import sys

from tqdm import trange

homedir = os.getenv('HOME')
if homedir is None:
    os.chdir("C:/Users/Adrian/Projects/Thesis/Takotsubo/takotsubo_data/HDP-SLDS-GP")
    homedir = "C:/Users/Adrian/Projects/Thesis/Takotsubo"
else:
    try:
        os.chdir(homedir + "/Documents/just-experiments/takotsubo_data/HDP-SLDS-GP")
    except FileNotFoundError:
        aux_dir = input("Please specify directory:\n")
        os.chdir(aux_dir)

cwd = os.getcwd()

# %% Import cell and config cell

import GPI_HDP as hdpgp
import numpy as np
import torch
import pickle as plk
from sklearn.metrics import confusion_matrix
import pickle
import wfdb

dtype = torch.float64
torch.set_default_dtype(dtype)

from get_data import get_data, compute_initial_statistics, take_standard_labels, compute_generation_ratio
from util_plots import plot_models, plot_warp, print_results, plot_MDS
import matplotlib.pyplot as plt

import time
start_ini_time = time.time()
# %% Get data cell
#samples = [50, 130]
#with open("D:/Programs/Workspaces/spyder-workspace/just-experiments/data/takotsubo/data_ntk.pkl",'rb') as inp:
#    data = pickle.load(inp)
ld = int(sys.argv[1])
#ld = 0
print(ld)
#ld = 3
ld_ = ld + 1
lead = "lead"+str(ld_)
ecg = np.load(homedir + "/data/takotsubo/ecg_filepaths/ecg_version_4_leads_filtered.npy")
#ecg = np.load(homedir + "/data/takotsubo/ecg_version_4_leads_filtered.npy")
labels_tak = np.load(homedir + "/data/takotsubo/ecg_filepaths/ecg_version_4_has_takotsubo.npy")
#labels_tak = np.load(homedir + "/data/takotsubo/ecg_version_4_has_takotsubo.npy")
basis_gpmodel_kernel = None
#ran = np.arange(2000)
warp = False
save_models = True
dynamic_params = []
lab_train = []
train_data = []
times = []
for t in range(15438):
    file_path = homedir + "/data/takotsubo/wfdb/tak_" + str(t)
    record = wfdb.rdrecord(file_path, return_res=32, physical=False)
    labels = wfdb.rdann(file_path, 'qrs', return_label_elements=['symbol']).symbol
    annotation = wfdb.rdann(file_path, 'qrs').sample
    qrs_ann = np.where(np.array(labels) == "N")[0]
    data = []
    time_ = []
    fiducial_points = []
    ecg_ = ecg[t,:,ld]
    peaks = wfdb.processing.xqrs_detect(ecg_, fs=500, verbose=False)
    if len(peaks) > 0:
        peaks = wfdb.processing.correct_peaks(ecg_, peaks, 100, 200)
    if len(qrs_ann) == 0:
        annotation = peaks
        for an in peaks:
            if an - 100 > 0 and an + 150 < len(ecg_):
                if len(time_) == 0:
                    time_.append(np.arange(an - 100, an + 150) - an)
                else:
                    if an + time_[0][-1] + 1 < len(ecg_):
                        time_.append(time_[0])
                if len(time_) == 1:
                    data.append(ecg_[np.arange(an - 100, an + 150)])
                else:
                    if an+ time_[0][-1] + 1 < len(ecg_):
                        data.append(ecg_[np.arange(an + time_[0][0], an + time_[0][-1] + 1)])
    elif len(peaks) == 0:
        for an in qrs_ann:
            if len(annotation) > an + 4 and an - 4 > 0:
                if len(time_) == 0:
                    time_.append(np.arange(annotation[an - 4], annotation[an + 4]) - annotation[an])
                else:
                    if annotation[an] + time_[0][-1] + 1 < len(ecg_):
                        time_.append(time_[0])
                if len(time_) == 1:
                    data.append(ecg_[annotation[an - 4]:annotation[an + 4]])
                else:
                    if annotation[an] + time_[0][-1] + 1 < len(ecg_):
                        data.append(ecg_[np.arange(annotation[an] + time_[0][0], annotation[an] + time_[0][-1] + 1)])
    else:
        for an in qrs_ann:
            if len(annotation) > an + 4 and an-4 > 0:
                time_.append(np.arange(annotation[an-4], annotation[an+4]) - annotation[an])
                annotation = peaks
                break
        if len(time_) == 0:
            for an in peaks:
                if an - 100 > 0 and an + 150 < len(ecg_):
                    time_.append(np.arange(an - 100, an + 150) - an)
        for an in peaks:
            if len(time_) > 0:
                if an  + time_[0][0] > 0 and an  + time_[0][-1] +1< len(ecg_):
                    if len(time_) == 0:
                        time_.append(np.arange(an  + time_[0][0], an + time_[0][-1] +1))
                    else:
                        if an  + time_[0][0] > 0 and an  + time_[0][-1] +1< len(ecg_):
                            time_.append(time_[0])
                    if len(time_) == 1:
                        data.append(ecg_[np.arange(an + time_[0][0], an + time_[0][-1] +1)])
                    else:
                        if an  + time_[0][0] > 0 and an  + time_[0][-1] +1< len(ecg_):
                            data.append(
                                ecg_[np.arange(an  + time_[0][0], an  + time_[0][-1] +1)])
    train_data.append(data)
    times.append(time_)
print("Train data shape:" + str(len(train_data)))
for t in range(15438):
    print("----------Patient: " + str(t) + "-------", flush=True)
    data_ = train_data[t]
    stds = [d.std() for d in data_]
    data_ = [(d - d.mean())/d.std() for d in data_]
    time_ = times[t]
    if len(data_) > 0:
        if not np.isclose(np.abs(data_[0]).sum(), 0.0) and not np.any(np.isclose(stds, 0.0)):
            #d = (d - d.mean())/d.std()
            ini_lengthscale = 20.0
            bound_lengthscale = (0.1, 30.0)

            #bound_sigma_, sig_, outputscale_ = (d_var*0.01, d_var*0.02), d_var*0.1, 1.0  # v42
            #bound_sigma_, sig_, outputscale_ = (0.01, 0.02), 0.0001, 1.0  # v42
            bound_sigma_, sig_, outputscale_ = (0.001, 0.002), 0.01, 1.0  # v42

            #noise_warp = bound_sigma_[0] * 3.0
            #bound_noise_warp = (bound_sigma_[0] * 2.8, bound_sigma_[0] * 3.2)

            noise_warp = 100.0
            bound_noise_warp = (100.0, 200.0)

            M = 2
            sigma = [sig_] * M
            #gamma = [d_var*0.2] * M
            #gamma = [0.000001] * M
            gamma = [0.001] * M
            bound_gamma = (0.1 ** 2, 20.0 ** 2)

            # Print hyperparams
            #print("Bound Sigma: ", bound_sigma_)
            #print("Sigma: ", sigma)
            #print("Outputscale: ", outputscale_)
            #print("Gamma: ", gamma)

            #d = np.reshape(d, (d.shape[0],-1,1))
            l = 0
            L = len(data_[0])
            N_0 = 0
            N = len(data_)

            # x_basis = np.atleast_2d(np.arange(l, L, 1, dtype=np.float64)).T
            x_basis = np.atleast_2d(time_[0]).astype(np.float64).T
            # ini_lengthscale = 5.0
            # x_basis = np.atleast_2d([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0]).T
            x_basis_warp = np.atleast_2d(time_[0]).astype(np.float64).T


            if t > 0:
                if basis_gpmodel_kernel is None:
                    with open(homedir + "/data/takotsubo/models_filt/"+lead+"/sw_gp_0.plk",'rb') as inp:
                        sw_gp = plk.load(inp)
                    ind = []
                    for gp in sw_gp.gpmodels:
                        ind.append(len(gp.indexes))
                    gp_chosen = sw_gp.gpmodels[np.argmax(np.array(ind))]
                    basis_gpmodel_kernel = gp_chosen.gp.kernel

            sw_gp = hdpgp.GPI_HDP(x_basis, x_basis_warp=x_basis_warp, kernels=None, model_type='dynamic',
                                  ini_lengthscale=ini_lengthscale, bound_lengthscale=bound_lengthscale,
                                  ini_gamma=gamma, ini_sigma=sigma, ini_outputscale=outputscale_, noise_warp=noise_warp,
                                  bound_sigma=bound_sigma_, bound_gamma=bound_gamma, bound_noise_warp=bound_noise_warp,
                                  warp_updating=False, method_compute_warp='greedy', verbose=True,
                                  annealing=False, hmm_switch=True, max_models=100, batch=0, mode_warp='rough',
                                  check_var=False, bayesian_params=True, cuda=False, inducing_points=False)

            if not basis_gpmodel_kernel is None:
                sw_gp.gpmodels[0].gp.kernel = basis_gpmodel_kernel
                sw_gp.gpmodels[0].fitted = True

            for i in range(N_0, N, 1):
                x_train = np.atleast_2d(time_[i]).astype(np.float64).T
                data__ = np.atleast_2d(data_[i]).T
                #start_time = time.time()
                print("Sample:", i, "/", str(N-1))
                sw_gp.include_sample(x_train, data__, with_warp=warp)#, force_model=0)# force_model=0 if labels[i]==-1 else 1)#
                #print("Time --- %s seconds ---" % (time.time() - start_time))

            if save_models:
                with open(homedir + "/data/takotsubo/models_filt/"+lead+"/sw_gp_"+str(t)+".plk",'wb') as inp:
                    plk.dump(sw_gp, inp)

            ind = []
            for gp in sw_gp.gpmodels:
                ind.append(len(gp.indexes))

            gp_chosen = sw_gp.gpmodels[np.argmax(np.array(ind))]
            #
            #
            # dynamic_params.append([gp_chosen.f_star, gp_chosen.cov_f[-1],
            #                        gp_chosen.A[-1], gp_chosen.Gamma[-1],
            #                        gp_chosen.C[-1], gp_chosen.Sigma[-1], sw_gp.x_w])
            # lab_train.append(train_lab_tak[t])
            # if t%100 == 0 or t==len(train_data):
            #     with open(homedir + "/data/takotsubo/models/"+lead+"/dynamic_params_train_"+lead+".pkl",'wb') as inp:
            #         plk.dump(dynamic_params, inp)
            if basis_gpmodel_kernel is None:
                basis_gpmodel_kernel = gp_chosen.gp.kernel

print("Time --- %s mins ---" % str((time.time() - start_ini_time)/60.0))

