# -*- coding: utf-8 -*-

# %% Directory check cell

import os
import pickle

from tqdm import trange

homedir = os.getenv('HOME')
if homedir is None:
    os.chdir("D:/Programs/Workspaces/spyder-workspace/just-experiments/HDP-SLDS-GP")
else:
    try:
        os.chdir(homedir + "/Documents/just-experiments/HDP-SLDS-GP")
    except FileNotFoundError:
        aux_dir = input("Please specify directory:\n")
        os.chdir(aux_dir)

cwd = os.getcwd()

# %% Import cell and config cell

import numpy as np
import torch
import sys
dtype = torch.float64
torch.set_default_dtype(dtype)

import time
start_ini_time = time.time()
# %% Get data cell from SSH
ld = sys.argv[1]
lead = "lead"+str(ld)
#num_clust = np.load(homedir + "/Documents/just-experiments/data/takotsubo/separate_params/num_clust_lead1.npy")
num_clust = np.zeros(15437)
num_samples = np.zeros(15437)
#num_clust = np.zeros(12349)
dynamic_params_fstar_ = []
dynamic_params_xbasis_ = []
dynamic_params_A_ = []
dynamic_params_C_ = []
dynamic_params_Gamma_ = []
dynamic_params_Sigma_ = []

#for i in trange(12349):
for i in trange(15438):
    try:
        with open(homedir + '/Documents/just-experiments/data/takotsubo/models_filt/'+lead+'/sw_gp_'+str(i)+'.plk', 'rb') as inp:
            sw_gp = pickle.load(inp)
            ind = []
            for gp in sw_gp.gpmodels:
                ind.append(len(gp.indexes))
            gp_chosen = sw_gp.gpmodels[np.argmax(np.array(ind))]
            num_clust[i] = len(sw_gp.gpmodels)
            num_samples[i] = len(gp_chosen.indexes)
            dynamic_params_fstar_.append(gp_chosen.f_star[-1])
            dynamic_params_xbasis_.append(gp_chosen.x_basis)
            dynamic_params_A_.append(gp_chosen.A[-1])
            dynamic_params_C_.append(gp_chosen.C[-1])
            dynamic_params_Gamma_.append(gp_chosen.Gamma[-1])
            dynamic_params_Sigma_.append(gp_chosen.Sigma[-1])
    except FileNotFoundError:
        print("Missed: "+str(i))

dynamic_params_fstar_ = np.array(dynamic_params_fstar_)
dynamic_params_xbasis_ = np.array(dynamic_params_xbasis_)
dynamic_params_A_ = np.array(dynamic_params_A_)
dynamic_params_C_ = np.array(dynamic_params_C_)
dynamic_params_Gamma_ = np.array(dynamic_params_Gamma_)
dynamic_params_Sigma_ = np.array(dynamic_params_Sigma_)

#dynamic_params = np.load(homedir + "/Documents/just-experiments/data/takotsubo/separate_params/dynamic_params_fstar.npy")
#dynamic_params = np.concatenate([dynamic_params, dynamic_params_fstar_])
np.save(homedir + "/Documents/just-experiments/data/takotsubo/separate_params_filt/"+lead+"/dynamic_params_fstar.npy",
        dynamic_params_fstar_)

np.save(homedir + "/Documents/just-experiments/data/takotsubo/separate_params_filt/"+lead+"/dynamic_params_xbasis.npy",
        dynamic_params_xbasis_)

#dynamic_params = np.load(homedir + "/Documents/just-experiments/data/takotsubo/separate_params/dynamic_params_A.npy")
#dynamic_params = np.concatenate([dynamic_params, dynamic_params_A_])
np.save(homedir + "/Documents/just-experiments/data/takotsubo/separate_params_filt/"+lead+"/dynamic_params_A.npy",
        dynamic_params_A_)

#dynamic_params = np.load(homedir + "/Documents/just-experiments/data/takotsubo/separate_params/dynamic_params_C.npy")
#dynamic_params = np.concatenate([dynamic_params, dynamic_params_C_])
np.save(homedir + "/Documents/just-experiments/data/takotsubo/separate_params_filt/"+lead+"/dynamic_params_C.npy",
        dynamic_params_C_)

#dynamic_params = np.load(homedir + "/Documents/just-experiments/data/takotsubo/separate_params/dynamic_params_Gamma.npy")
#dynamic_params = np.concatenate([dynamic_params, dynamic_params_Gamma_])
np.save(homedir + "/Documents/just-experiments/data/takotsubo/separate_params_filt/"+lead+"/dynamic_params_Gamma.npy",
        dynamic_params_Gamma_)

#dynamic_params = np.load(homedir + "/Documents/just-experiments/data/takotsubo/separate_params/dynamic_params_Sigma.npy")
#dynamic_params = np.concatenate([dynamic_params, dynamic_params_Sigma_])
np.save(homedir + "/Documents/just-experiments/data/takotsubo/separate_params_filt/"+lead+"/dynamic_params_Sigma.npy",
        dynamic_params_Sigma_)

# dynamic_params_A =  dynamic_params_A_
# dynamic_params_C = dynamic_params_C_
# dynamic_params_Gamma = dynamic_params_Gamma_
# dynamic_params_Sigma = dynamic_params_Sigma_
# dynamic_params_fstar = dynamic_params_fstar_
#with open(homedir + "/Documents/just-experiments/data/takotsubo/dynamic_params_labels_train_lead1.pkl", 'rb') as inp:
#    dynamic_params_A_labels = pickle.load(inp)

np.save(homedir + "/Documents/just-experiments/data/takotsubo/separate_params_filt/"+lead+"/num_clust_"+lead+".npy", num_clust)
np.save(homedir + "/Documents/just-experiments/data/takotsubo/separate_params_filt/"+lead+"/num_samples_"+lead+".npy", num_samples)
#np.save(homedir + "/Documents/just-experiments/data/takotsubo/dynamic_params_A_labels.npy", dynamic_params_A_labels)

print("END")

