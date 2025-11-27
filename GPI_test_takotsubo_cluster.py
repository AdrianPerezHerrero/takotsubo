# -*- coding: utf-8 -*-

# %% Directory check cell

import os
import sys

from tqdm import trange

homedir = os.getenv('HOME')
if homedir is None:
    os.chdir("C:/Users/Adrian/Projects/Thesis/Takotsubo/takotsubo_data/HDP-SLDS-GP")
    homedir = "C:/Users/Adrian/Projects/Thesis/Takotsubo/takotsubo_data/"
else:
    try:
        os.chdir(homedir + "/Documents/takotsubo_data/HDP-SLDS-GP")
        homedir = homedir + "/Documents/takotsubo_data/"
    except FileNotFoundError:
        aux_dir = input("Please specify directory:\n")
        os.chdir(aux_dir)

cwd = os.getcwd()

# %% Import cell and config cell

import hdpgpc.GPI_HDP as hdpgp
from hdpgpc.util_plots import print_results, plot_models_plotly
import numpy as np
import torch
import pickle as plk
from sklearn.metrics import confusion_matrix
import pickle
import wfdb
#import tensorflow as tf

dtype = torch.float64
torch.set_default_dtype(dtype)

import matplotlib.pyplot as plt

import time
start_ini_time = time.time()
# %% Get data cell
#samples = [50, 130]
#with open("D:/Programs/Workspaces/spyder-workspace/just-experiments/data/takotsubo/data_ntk.pkl",'rb') as inp:
#    data = pickle.load(inp)
#ld = int(sys.argv[1])
#ld = 0
#ld = 3
#ecg = np.load(homedir + "/data/takotsubo/full_dataset_filtered/data_ecg_leads.npy")
#ecg = np.load(homedir + "/data/takotsubo/ecg_version_4_leads_filtered.npy")
#labels_tak = np.load(homedir + "/data/takotsubo/ecg_version_4_has_takotsubo.npy")
#labels_tak = np.load(homedir + "/data/takotsubo/ecg_version_4_has_takotsubo.npy")
basis_gpmodel_kernel = None
#ran = np.arange(2000)
maxs = []
warp = False
save_models = True
dynamic_params = []
lab_train = []
with open(homedir + "/data/takotsubo/full_dataset_filtered/window_data_xqrs.pickle",'rb') as inp:
    train_data = pickle.load(inp)
with open(homedir + "/data/takotsubo/full_dataset_filtered/xbas_window_data_xqrs.pickle",'rb') as inp:
    times = pickle.load(inp)
print("Train data shape:" + str(len(train_data)))
for t in range(len(train_data)):
    print(">>>>>>>>>>>>>>>>>>----------Patient: " + str(t) + "-------<<<<<<<<<<<<<<<<<<<", flush=True)
    data_ = train_data[t]
    # stds = [d.std() for d in data_]
    # data_ = [(d - d.mean())/d.std() for d in data_]
    time_ = times[t]
    if len(data_) > 1:
        #d = (d - d.mean())/d.std()
        ini_lengthscale = 5.0
        bound_lengthscale = (0.5, 50.0)

        #bound_sigma_, sig_, outputscale_ = (d_var*0.01, d_var*0.02), d_var*0.1, 1.0  # v42
        #bound_sigma_, sig_, outputscale_ = (0.01, 0.02), 0.0001, 1.0  # v42
        if t==0:
            bound_sigma_ = (200.0, 300.0)
        else:
            bound_sigma_ = (200.0, 300.0)
        sig_, outputscale_ = 0.1, 200.0
        #noise_warp = bound_sigma_[0] * 3.0
        #bound_noise_warp = (bound_sigma_[0] * 2.8, bound_sigma_[0] * 3.2)

        noise_warp = 100.0
        bound_noise_warp = (100.0, 200.0)

        M = 2
        sigma = [sig_] * M
        #gamma = [d_var*0.2] * M
        #gamma = [0.000001] * M
        gamma = [0.5] * M
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



        if basis_gpmodel_kernel is None:
            with open(homedir + "/data/takotsubo/full_dataset_filtered/saved_models/sw_gp_0.plk",'rb') as inp:
                sw_gp = plk.load(inp)
            ind = []
            for gp in sw_gp.gpmodels[0]:
                ind.append(len(gp.indexes))
            gp_chosen = sw_gp.gpmodels[0][np.argmax(np.array(ind))]
            basis_gpmodel_kernel = [gp_chosen.gp.kernel]
            basis_gpmodel_kernel[0].k1.k2.length_scale = 10.0

        # sw_gp = hdpgp.GPI_HDP(x_basis, x_basis_warp=x_basis_warp, kernels=None, model_type='dynamic',
        #                       ini_lengthscale=ini_lengthscale, bound_lengthscale=bound_lengthscale,
        #                       ini_gamma=gamma, ini_sigma=sigma, ini_outputscale=outputscale_, noise_warp=noise_warp,
        #                       bound_sigma=bound_sigma_, bound_gamma=bound_gamma, bound_noise_warp=bound_noise_warp,
        #                       warp_updating=False, method_compute_warp='greedy', verbose=True,
        #                       annealing=False, hmm_switch=True, max_models=100, batch=0, mode_warp='rough',
        #                       check_var=False, bayesian_params=True, cuda=False, inducing_points=False)
        #fitted = False if t==0 else True
        fitted = True
        sw_gp = hdpgp.GPI_HDP(x_basis, kernels=basis_gpmodel_kernel, x_basis_warp=x_basis_warp, n_outputs=8,
                              model_type='dynamic',
                              ini_lengthscale=ini_lengthscale, bound_lengthscale=bound_lengthscale,
                              ini_gamma=gamma, ini_sigma=sigma, ini_outputscale=outputscale_, noise_warp=noise_warp,
                              bound_sigma=bound_sigma_, bound_gamma=bound_gamma, bound_noise_warp=bound_noise_warp,
                              warp_updating=False, method_compute_warp='greedy', verbose=False,
                              hmm_switch=True, max_models=3, mode_warp='rough', annealing=False,
                              bayesian_params=True, inducing_points=False, reestimate_initial_params=False,
                              n_explore_steps=1, free_deg_MNIV=8, share_gp=True, fitted=fitted)

        # if not basis_gpmodel_kernel is None:
        #     for ld in range(8):
        #         sw_gp.gpmodels[ld][0].gp.kernel = basis_gpmodel_kernel
        #         sw_gp.gpmodels[ld][0].fitted = True

        x_train = np.atleast_2d(time_[0]).astype(np.float64).T
        x_trains = [x_train] * (N-1)
        sw_gp.include_batch(x_trains,data_[1:])
        if len(sw_gp.gpmodels[0]) > 1:
            print("Stop")
        # for i in range(N_0, N, 1):
        #     #x_train = np.atleast_2d(time_[i]).astype(np.float64).T
        #     x_train = np.atleast_2d(time_[0]).astype(np.float64).T
        #     data__ = np.atleast_2d(data_[i,:,0]).T
        #     #start_time = time.time()
        #     print("Sample:", i, "/", str(N-1))
        #     sw_gp.include_sample(x_train, data__, with_warp=warp)#, force_model=0)# force_model=0 if labels[i]==-1 else 1)#
        #     #print("Time --- %s seconds ---" % (time.time() - start_time))

        mean = np.array([gps[0].f_star[-1] for gps in sw_gp.gpmodels]).T
        pure_mean = np.mean(data_[1:][np.array(sw_gp.gpmodels[0][0].indexes)], axis=0)[np.newaxis, :,:]
        num_clust = np.array([len(sw_gp.gpmodels[0])])
        num_samp_included = np.array([len(sw_gp.gpmodels[0][0].indexes)])
        diag_sigma = np.array([np.diag(gps[0].Sigma[-1])[:,np.newaxis] for gps in sw_gp.gpmodels]).T
        if t == 0:
            np.save(homedir + "/data/takotsubo/full_dataset_filtered/saved_parameters/mean_6.npy", mean)
            np.save(homedir + "/data/takotsubo/full_dataset_filtered/saved_parameters/pure_mean_6.npy", pure_mean)
            np.save(homedir + "/data/takotsubo/full_dataset_filtered/saved_parameters/num_samp_6.npy", num_samp_included)
            np.save(homedir + "/data/takotsubo/full_dataset_filtered/saved_parameters/num_clust_6.npy", num_clust)
            np.save(homedir + "/data/takotsubo/full_dataset_filtered/saved_parameters/diag_sig_6.npy", diag_sigma)
        else:
            mean_ = np.load(homedir + "/data/takotsubo/full_dataset_filtered/saved_parameters/mean_6.npy")
            mean_ = np.vstack([mean_, mean])
            np.save(homedir + "/data/takotsubo/full_dataset_filtered/saved_parameters/mean_6.npy", mean_)
            pure_mean_ = np.load(homedir + "/data/takotsubo/full_dataset_filtered/saved_parameters/pure_mean_6.npy")
            pure_mean_ = np.vstack([pure_mean_, pure_mean])
            np.save(homedir + "/data/takotsubo/full_dataset_filtered/saved_parameters/pure_mean_6.npy", pure_mean_)
            num_samp_ = np.load(homedir + "/data/takotsubo/full_dataset_filtered/saved_parameters/num_samp_6.npy")
            num_samp_ = np.hstack([num_samp_, num_samp_included])
            np.save(homedir + "/data/takotsubo/full_dataset_filtered/saved_parameters/num_samp_6.npy", num_samp_)
            num_clust_ = np.load(homedir + "/data/takotsubo/full_dataset_filtered/saved_parameters/num_clust_6.npy")
            num_clust_ = np.hstack([num_clust_, num_clust])
            np.save(homedir + "/data/takotsubo/full_dataset_filtered/saved_parameters/num_clust_6.npy", num_clust_)
            diag_sigma_ = np.load(homedir + "/data/takotsubo/full_dataset_filtered/saved_parameters/diag_sig_6.npy")
            diag_sigma_ = np.vstack([diag_sigma_, diag_sigma])
            np.save(homedir + "/data/takotsubo/full_dataset_filtered/saved_parameters/diag_sig_6.npy", diag_sigma_)
        if save_models:
            if t==0:
                sw_gp.save_swgp(homedir + "/data/takotsubo/full_dataset_filtered/saved_models/sw_gp_"+str(t)+".plk")

        gp_chosen = sw_gp.gpmodels[0][0]
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
            basis_gpmodel_kernel = [gp_chosen.gp.kernel]
    else:
        mean = np.zeros((1, data_.shape[1], data_.shape[2]))
        pure_mean = np.zeros((1, data_.shape[1], data_.shape[2]))
        num_samp_included = np.array([0])
        num_clust = np.array([1])
        diag_sigma = np.zeros((1, data_.shape[1], data_.shape[2]))
        if t == 0:
            np.save(homedir + "/data/takotsubo/full_dataset_filtered/saved_parameters/mean_6.npy", mean)
            np.save(homedir + "/data/takotsubo/full_dataset_filtered/saved_parameters/pure_mean_6.npy", pure_mean)
            np.save(homedir + "/data/takotsubo/full_dataset_filtered/saved_parameters/num_samp_6.npy", num_samp_included)
            np.save(homedir + "/data/takotsubo/full_dataset_filtered/saved_parameters/num_clust_6.npy", num_clust)
            np.save(homedir + "/data/takotsubo/full_dataset_filtered/saved_parameters/diag_sig_6.npy", diag_sigma)
        else:
            mean_ = np.load(homedir + "/data/takotsubo/full_dataset_filtered/saved_parameters/mean_6.npy")
            mean_ = np.vstack([mean_, mean])
            np.save(homedir + "/data/takotsubo/full_dataset_filtered/saved_parameters/mean_6.npy", mean_)
            pure_mean_ = np.load(homedir + "/data/takotsubo/full_dataset_filtered/saved_parameters/pure_mean_6.npy")
            pure_mean_ = np.vstack([pure_mean_, pure_mean])
            np.save(homedir + "/data/takotsubo/full_dataset_filtered/saved_parameters/pure_mean_6.npy", pure_mean_)
            num_samp_ = np.load(homedir + "/data/takotsubo/full_dataset_filtered/saved_parameters/num_samp_6.npy")
            num_samp_ = np.hstack([num_samp_, num_samp_included])
            np.save(homedir + "/data/takotsubo/full_dataset_filtered/saved_parameters/num_samp_6.npy", num_samp_)
            num_clust_ = np.load(homedir + "/data/takotsubo/full_dataset_filtered/saved_parameters/num_clust_6.npy")
            num_clust_ = np.hstack([num_clust_, num_clust])
            np.save(homedir + "/data/takotsubo/full_dataset_filtered/saved_parameters/num_clust_6.npy", num_clust_)
            diag_sigma_ = np.load(homedir + "/data/takotsubo/full_dataset_filtered/saved_parameters/diag_sig_6.npy")
            diag_sigma_ = np.vstack([diag_sigma_, diag_sigma])
            np.save(homedir + "/data/takotsubo/full_dataset_filtered/saved_parameters/diag_sig_6.npy", diag_sigma_)

    print("Time --- %s mins ---" % str((time.time() - start_ini_time)/60.0))

