# -*- coding: utf-8 -*-

# %% Directory check cell

import os
import pickle

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import RocCurveDisplay, roc_curve, roc_auc_score, auc, precision_recall_curve
from sklearn.inspection import permutation_importance
import umap
import pandas as pd
from torch.special import logit

homedir = os.getenv('HOME')
if homedir is None:
    os.chdir("C:/Users/Adrian/Projects/Thesis/Takotsubo/takotsubo_data/HDP-SLDS-GP")
    homedir = "C:/Users/Adrian/Projects/Thesis/Takotsubo"
else:
    try:
        os.chdir(homedir + "/Documents/takotsubo_data/HDP-SLDS-GP")
        homedir = homedir + "/Documents"
    except FileNotFoundError:
        aux_dir = input("Please specify directory:\n")
        os.chdir(aux_dir)

cwd = os.getcwd()

# %% Import cell and config cell

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
dtype = torch.float64
torch.set_default_dtype(dtype)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap([[1,0,0], [0.2,0.2,1.0]])
import time
start_ini_time = time.time()
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from torch.utils.data import Dataset, DataLoader
from util_plots import plot_scatter
import statsmodels.api as sm
# %% Get data cell

leads = [1,2,3,4,5,6,7,8]
#leads = [1]
age = np.load(homedir + "/takotsubo_data/data/takotsubo/full_dataset_filtered/data_age.npy")
labels = np.load(homedir + "/takotsubo_data/data/takotsubo/full_dataset_filtered/labels_takotsubo.npy")
labels_female = np.load(homedir + "/takotsubo_data/data/takotsubo/full_dataset_filtered/labels_sex.npy")
filepaths = np.load(homedir + "/takotsubo_data/data/takotsubo/full_dataset_filtered/data_filepaths.npy")

data = None

params = ["fstar"]
data = np.load(homedir + "/takotsubo_data/data/takotsubo/full_dataset_filtered/saved_parameters/pure_mean_6.npy")
data_std = np.load(homedir + "/takotsubo_data/data/takotsubo/full_dataset_filtered/saved_parameters/diag_sig_6.npy")


max__ = 200
min__ = -100
# max__ = 180
# min__ = -50
leads = [0, 1, 2, 3, 4, 5, 6, 7]
data = data[:, :, np.array(leads)]
leads = [ld + 1 for ld in leads]

standarize = True
female_include = True
st_dev_include = False
include_clust = True
include_samples = True
include_age = False

if st_dev_include:
    st_dev = np.zeros((data.shape[0], len(leads)))
    for ld in leads:
        fstar_ind = params.index("fstar") + len(params) * leads.index(ld)
        #Compute ST-segment elevation
        isoelec_val = np.array([np.mean(data[i, 45:55, fstar_ind]) for i in range(data.shape[0])])
        fin_diff = np.diff(np.array([gaussian_filter1d(data[i, 100:150, fstar_ind], 1.0) for i in range(data.shape[0])]),
                           axis=1)
        fin_diff_2 = np.diff(fin_diff, axis=1)
        pos_J = np.zeros(data.shape[0], dtype=np.int32)
        for i, f in enumerate(fin_diff_2):
            influent_points = np.where(np.diff(np.sign(f)))[0]
            pos_J[i] = np.where(np.diff(np.sign(f)))[0][np.argmin(np.diff(influent_points)).astype(np.int32)] if np.diff(influent_points).shape[0] > 0 else 10
        pos_J = pos_J + 100
        st_dev[:, leads.index(ld)] = np.array([np.mean(data[i, pos_J[i] + 20:pos_J[i] + 30, fstar_ind]) for i in range(data.shape[0])])
        #st_dev[:, leads.index(ld)] = np.mean(data[:,120:140, fstar_ind], axis=1) - np.mean(data[:,40:50,fstar_ind], axis=1)
#Good one stemi vs takotsubo with 4-step
point_J_division = False
step = 1
if not point_J_division:
    #Same division for all
    ps = np.arange(0, max__-min__, step)
    #ps = np.arange(60, 230)
    #ps = np.arange(0, max__-min__, 20)
    #ps = np.arange(1)
    data = data[:,ps,:]
else:
#ST division for each
    ps = np.arange(0,70)
    data = np.array([data[i, pos_J[i]-10:pos_J[i]+80, :] for i in range(data.shape[0])])
data = np.reshape(np.transpose(data, (0, 2, 1)), (data.shape[0], -1)) / 100.0
data_std = np.reshape(np.transpose(data_std, (0, 2, 1)), (data.shape[0], -1)) / 100.0


## -----------------------FINAL ONES PURE MEAN -------------------- ##
# New puremean max 0.275
logit_importance_variables = np.array([  16, 29, 170,  329, 401,
        423,  438,  470,  517,  704,  950,
       1006, 1053, 1550, 1702, 1910, 2215])

### TS vs STEMI-NSTEMI ROC 0.917 PR 0.529
# logit_importance_variables = np.array([  35,   99,  164,  330,  365,  378,  382,  423,  457,
#         561,  568,  954, 1038, 1550, 1703, 1935, 1960, 2155,
#        2216])

# ### TS vs NSTEMI     ROC 0.908 PR 0.595
# logit_importance_variables = np.array([ 35,  99,  151,  165,  170,  182,  346,  362,
#         432,  457,  954, 1005, 1046, 1114, 1534,
#        1550, 1703, 1935, 2155])

# ### A TS vs A STEMI  ROC 0.896 PR 0.689 (fit)
# logit_importance_variables = np.array([135,  150,  164,  241,  283,
#         289,  319,  421,  446,  632,  717,  722,  757,  780,
#         929,  954, 1200, 1250, 1347, 1552, 1600, 1603, 1608, 1801])

# ### TS vs STEMI     ROC 0.940 PR 0.839 (fit)
# logit_importance_variables = np.array([  16,  163,  311,  328,  362,  442,
#         473,  654,  965, 1267, 1290,
#        1361, 1450, 1556, 1605, 1614, 1814, 1921, 1939])

# logit_importance_variables = np.array([  16,  163,  311,  328,  362,  442,
#         473,  654,  965, 1267, 1290,
#        1361, 1450, 1556, 1605, 1614, 1814, 1921, 1939, 1702])

#ALL but only male ROC 0.794 PR 0.082
# logit_importance_variables = np.array([  94,   95,   96,  159,  317,  332,  333,  368,  390,  421,  422,
#         423,  427,  428,  548,  572,  573,  574,  614,  622,  635,  757,
#         797,  928,  989, 1046, 1082, 1257, 1563, 1813, 1814, 1910, 1911,
#        2037, 2038, 2040, 2041, 2106, 2126, 2216, 2348, 2352])
#ALL but only male ROC 0.790 PR 0.068
# logit_importance_variables = np.array([ 95,  159,  317,  332,  368,  390,  421,
#         427,  548,  572,  614,  622,  635,  757,
#         797,  928,  989, 1046, 1082, 1257, 1563, 1813, 1911,
#        2038, 2106, 2126, 2216, 2348])
# TS vs STEMI male
# logit_importance_variables = np.array([ 95,  159,  317,  332,  368,  390,  421,
#         427,  572,  622,  635,  757,
#         797,  928,  989, 1046, 1082, 1257, 1563, 1813, 1911,
#        2038, 2106, 2216, 2348])

# STEMI vs NSTEMI
# logit_importance_variables = np.array([  20,   22,   25,   59,   97,  110,  127,  174,  176,  181,  224,
#         236,  272,  278,  342,  352,  483,  591,  597,  660,  757,  823,
#         927, 1007, 1039, 1044, 1113, 1300, 1305, 1407, 1500, 1604, 1800,
#        1857, 1949])
#------------------------------------------------------------------

#
d__ = np.mean(data[np.where(labels)[0][:30]], axis=0)
plt.figure(figsize=(8,5))
plt.plot(d__, 'r')
plt.vlines(logit_importance_variables, np.min(d__)*1.05, np.max(d__)*1.05, 'b', alpha=0.3)
plt.vlines(np.arange(0, ps.shape[0] * len(leads), ps.shape[0]), np.min(d__) * 0.5, np.max(d__) * 0.5, 'k',
           linestyles='dashed', alpha=0.3)
plt.xlabel("Location of important features for split All")
plt.show()

#linear_importance_variables = np.array([193,106,136,130,157,16,134,187,202,230,88,76,198,112,43,50])
#rbf_importance_variables = np.array([36,38,180,181,93,91,120,121,208,64,66,2,7,167,151])
#importance_variables = np.sort(np.concatenate([linear_importance_variables, rbf_importance_variables]))
data = data[:, logit_importance_variables]

#Modification TS vs STEMI:
# newvar = data[:, 6] - data[:, 5] - data[:, 4]
# newvar_2 = data[:, 18] - data[:, 16]# - data[:, 17]
# newvar_3 = data[:, 11] - data[:, 8]# - data[:, 17]
# newvar_4 = data[:, 0]  - data[:, 4]# - data[:, 17]
# data[:, 11] = newvar_3
# data[:, 5] = newvar
# data[:, 16] = newvar_2
# data[:, 0] = newvar_4
# delete = [4, 6, 8, 17, 18]
# delete = [4, 6, 8, 18]
# data = np.delete(data, delete, axis=1)

#Modification TS vs suspected:
newvar = data[:, 0] - data[:, 1]
newvar_2 = data[:, 6] + data[:, 7] + data[:, 8] - data[:, 5]

data[:, 0] = newvar
data[:, 5] = newvar_2
delete = [1, 6, 7, 8]
data = np.delete(data, delete, axis=1)

#Modification ATS vs ASTEMI:
# newvar = data[:, 2] - data[:, 1]
# newvar_ = data[:, 0] - data[:, 7]
# newvar_2 = data[:, 8] - data[:, 4]# - data[:, 3]
# newvar_3 = data[:, 14] - data[:, 13]# - data[:, 12]
# newvar_4 = data[:, 10] - data[:, 11]
# newvar_5 = data[:, 18] - data[:, 17]
# #
# data[:, 1] = newvar
# data[:, 0] = newvar_
# data[:, 8] = newvar_2
# data[:, 14] = newvar_3
# data[:, 10] = newvar_4
# data[:, 18] = newvar_5
# delete = [2, 4, 7,11, 13]
# data = np.delete(data, delete, axis=1)

#Modification TS vs Verified:
# newvar = data[:, 2] - data[:, 4]
# newvar_ = data[:, 8] - data[:, 7]
# newvar_2 = data[:, 16] - data[:, 17]# - data[:, 3]
# #
# data[:, 2] = newvar
# data[:, 8] = newvar_
# data[:, 16] = newvar_2
# delete = [4, 7, 17]
# data = np.delete(data, delete, axis=1)

#Modification TS vs NSTEMI:
# newvar = data[:, 3] - data[:, 2]
# newvar_ = data[:, 9] - data[:, 8]
# #newvar_2 = data[:, 4] - data[:, 7]# - data[:, 3]
# # #
# data[:, 3] = newvar
# data[:, 9] = newvar_
# #data[:, 4] = newvar_2
# delete = [2, 8]
# data = np.delete(data, delete, axis=1)

# data_std = data_std[:, logit_importance_variables]
#data = np.concatenate([data, data_std], axis=1)
#Add num clusters and num samples.
if include_clust:
    fold = homedir + "/takotsubo_data/data/takotsubo/full_dataset_filtered/saved_parameters"
    num_clust = np.load(fold + "/num_clust_6.npy")
    data = np.concatenate([data, num_clust[:,np.newaxis]], axis=1)
    if st_dev_include:
        data = np.concatenate([data, st_dev[:,leads.index(ld)][:,np.newaxis]], axis=1)
if include_samples:
    ld=1
    lead = "lead" + str(ld)
    fold = homedir + "/takotsubo_data/data/takotsubo/full_dataset_filtered"
    num_samples = np.load(fold + "/num_beats.npy")
    #num_samples = np.load(fold + "/saved_parameters/num_samp_6.npy")
    data = np.concatenate([data, num_samples[:,np.newaxis]], axis=1)

if female_include:
    data = np.concatenate([data, labels_female[:,np.newaxis]], axis=1)

if include_age:
    data = np.concatenate([data, age[:,np.newaxis]], axis=1)


with open(homedir + "/takotsubo_data/data/takotsubo/STEMI_scripts/labels_per_fold_takotsubo.pickle", "rb") as f:
    labels_per_fold = pickle.load(f)
with open(homedir + "/takotsubo_data/data/takotsubo/STEMI_scripts/filepaths_per_fold_takotsubo.pickle", "rb") as f:
    filepaths_per_fold = pickle.load(f)

def load_list_of_ids(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
        return lines


def id_from_filepath(filepath):
    file_parts = filepath.split('/')
    details = file_parts[1].split('_')
    return details[2]


for i, f in enumerate(filepaths):
    if f[:13] == 'Cases level 1':
        filepaths[i] = 'Cases level 3' + f[13:]
id_filepaths = [id_from_filepath(f) for f in filepaths]

stemi_id_set = set(load_list_of_ids(homedir + "/takotsubo_data/data/takotsubo/STEMI_scripts/stemi.txt"))
nstemi_id_set = set(load_list_of_ids(homedir + "/takotsubo_data/data/takotsubo/STEMI_scripts/nstemi.txt"))
ant_stemi_id_set = set(load_list_of_ids(homedir + "/takotsubo_data/data/takotsubo/AptikaAntstemiPlot/source/ant_stemi.txt"))
apical_ts_id_set = set(load_list_of_ids(homedir + "/takotsubo_data/data/takotsubo/AptikaAntstemiPlot/source/aptical.txt"))

labels_stemi = []
for i, id in enumerate(id_filepaths):
    if id in stemi_id_set:
        labels_stemi.append(1)
    else:
        labels_stemi.append(0)
labels_stemi = np.array(labels_stemi)

labels_nstemi = []
for i, id in enumerate(id_filepaths):
    if id in nstemi_id_set:
        labels_nstemi.append(1)
    else:
        labels_nstemi.append(0)
labels_nstemi = np.array(labels_nstemi)

labels_ant_stemi = []
for i, id in enumerate(id_filepaths):
    if id in ant_stemi_id_set:
        labels_ant_stemi.append(1)
    else:
        labels_ant_stemi.append(0)
labels_ant_stemi = np.array(labels_ant_stemi)

labels_apical_ts = []
for i, id in enumerate(id_filepaths):
    if id in apical_ts_id_set:
        labels_apical_ts.append(1)
    else:
        labels_apical_ts.append(0)
labels_apical_ts = np.array(labels_apical_ts)

#taken = np.sort(np.unique(np.concatenate([np.where(labels_apical_ts)[0], np.where(labels_ant_stemi)[0]])))
taken = np.sort(np.unique(np.concatenate([np.where(labels_stemi)[0], np.where(labels)[0]])))
#taken = np.sort(np.unique(np.concatenate([np.where(labels_stemi)[0], np.where(labels_nstemi)[0], np.where(labels)[0]])))
#taken = np.sort(np.unique(np.concatenate([np.where(labels_nstemi)[0], np.where(labels)[0]])))
#taken = np.sort(np.unique(np.concatenate([np.where((labels_female==0) & (labels==1))[0], np.where((labels_female==0) & (labels_stemi==1))[0]])))
#taken = np.sort(np.unique(np.concatenate([np.where((labels_female==1) & (labels==1))[0], np.where((labels_female==1) & (labels_stemi==1))[0], np.where((labels_female==1) & (labels_nstemi==1))[0]])))
#taken = np.sort(np.unique(np.concatenate([np.where(labels_nstemi)[0], np.where(labels_stemi)[0]])))
#taken = np.where(labels_female==0)[0]
#labels_joint = labels_stemi + labels_nstemi
#labels_joint[np.where(labels_joint > 1.0)] = labels_joint[np.where(labels_joint > 1.0)] - 1
#taken = np.arange(data.shape[0])
taken = taken[np.where(taken!=12279)] #Remove this index which is nonsense data
filepaths = filepaths[taken]
data = data[taken]
labels = labels[taken]
data_fold_ids = []
labels_fold_ids = []
removed_per_fold = []
for i, fold in enumerate(filepaths_per_fold):
    data_fold_ids_ = [np.where(f == filepaths)[0] for f in fold]
    num_id = np.array([len(d) for d in data_fold_ids_])
    data_fold_ids.append(np.array([data_fold_ids_[j] for j in np.where(num_id > 0)[0]]))
    labels_fold_ids.append(np.array(labels[data_fold_ids[-1]]))
    removed_per_fold.append(np.where(num_id == 0)[0])

all_filepaths = np.concatenate(filepaths_per_fold)


roc_curves = []
confusion_matrices_train = []
confusion_matrices_test = []
accuracy_score = []
predictions_per_fold = []
label_predictions_per_fold = []
importance_per_fold = []
shap_per_fold = []
coefs_per_fold = []
results_per_fold = []
model_per_fold = []
ci_lower_per_fold = []
ci_upper_per_fold = []
for i in range(len(data_fold_ids)):
    print("----- Fold " + str(i + 1) + " -----")
    data_test_ids = np.array(data_fold_ids[i]).squeeze()
    data_train_ids = np.concatenate([x for j,x in enumerate(data_fold_ids) if j!=i]).squeeze()
    #data_train = data
    #labels_train = labels
    data_train = data[data_train_ids]
    data_test = data[data_test_ids]
    if standarize:
        std_ = np.std(data_train, axis=0)[np.newaxis, :]
        mean_ = np.mean(data_train, axis=0)[np.newaxis, :]
        data_train = (data_train - mean_) / std_
        data_test = (data_test - mean_) / std_
    else:
        mean_ = np.zeros(data.shape[1])[np.newaxis, :]
        std_ = np.ones(data.shape[1])[np.newaxis, :]
    # Adding the intercept
    data_train = sm.add_constant(data_train)
    data_test = sm.add_constant(data_test)
    labels_train = labels[data_train_ids]
    labels_test = labels[data_test_ids]

    # Calculate the weight for the positive class
    num_negative = (labels_train == 0).sum()
    num_positive = (labels_train == 1).sum()
    pos_weight = num_negative / num_positive * 1.0
    neg_weight = num_positive / num_negative * 1.0
    #pos_weight = 10.0
    sample_weight = (labels_train.astype(np.float64) * (pos_weight-1)) + 1
    #sample_weight = (1-labels_train.astype(np.float64) * neg_weight) + (labels_train.astype(np.float64) * pos_weight)

    model = sm.GLM(labels_train, data_train, family=sm.families.Binomial())#, var_weights=sample_weight)
    #results = model.fit(maxiter=10000, method='newton', tol=1e-15, disp=True)
    results = model.fit()
    # suspected
    # results.params = np.array([-0.47711231,  0.0305289 , -0.03321203,  0.0630134 ,  0.04307179,
    #         0.01291432, -0.03441353,  0.02973406,  0.02512343,  0.03297355,
    #        -0.04401908,  0.02400392,  0.04346289,  0.0262804 , -0.03045148,
    #        -0.03116507, -0.02699887,  0.03464375, -0.00598701,  0.03773831,
    #         0.14536005]) # 289
    # stemi
    # results.params = np.array([-2.91568966,  0.4586356 ,  1.3588938 ,  0.79193195,  0.78245554,
    #    -1.15981332, -0.94722688,  0.75723304,  0.88761015, -1.20286768,
    #     1.25178442,  0.48819805,  1.12706593,  0.7728372 ,  1.47407101,
    #     0.78673791,  0.8227099 , -1.44773659, -1.12190486,  1.90593721,
    #     0.20786489,  0.77211289,  1.8024081 ])
    # verified
    # results.params = np.array([-4.12699773, -0.28645969, -0.29074117,  0.59356393,  0.45004603,
    #    -0.47271088,  0.44509459, -0.36314497, -0.56337732,  0.30790242,
    #     0.444825  , -0.58314362,  0.30797364,  0.45844127, -0.28287171,
    #    -0.41399542,  0.40903296,  0.34140134, -0.36237051,  0.26679061,
    #     0.00677187,  0.39861595,  1.56452598])
    # nstemi
    # results.params = np.array([-3.54427127, -0.31458346, -0.31740316, -1.11276505,  1.16687359,
    #     1.48659339, -1.1005657 ,  0.43823757, -0.64462804, -0.65791969,
    #     0.64622418,  0.46671768,  0.41860826,  0.58932036,  0.51970719,
    #     0.62542193, -0.84528378, -0.53188966,  0.77268257, -0.37056689,
    #    -0.03753229,  0.32179135,  1.5351598 ])
    #ats astemi
    # results.params = np.array([-2.64129366,  0.64932886, -2.13122699,  2.06037106, -0.22488591,
    #    -0.62385278,  0.64069901,  0.81017605, -0.48964689,  0.86645808,
    #     0.56098693, -0.78433708,  0.82903096, -1.39276678,  1.20029372,
    #    -1.25843639,  1.86669326,  0.96202379, -1.49669578, -0.64164207,
    #     1.08336974, -1.26475865,  1.77250654, -0.99268764, -1.01603546,
    #    -0.08423356,  0.19634641,  1.71744011])
    # results = model.fit_regularized(method='elastic_net', alpha=0.001, refit=True,
    #                                 L1_wt=1.0, maxiter=8000, zero_tol=1e-2)
    #results = model.fit_regularized(method='elastic_net', alpha=0.000, refit=False, L1_wt=1.0, maxiter=8000)
    #results = model.fit_regularized(method='elastic_net', alpha=0.001, refit=False, L1_wt=0.0, maxiter=8000)
    #model = LogisticRegression(max_iter=5000, tol=1e-4, solver='saga', penalty='l2')
    # model = LogisticRegression(max_iter=2000, tol=1e-6, class_weight='balanced')
    #results = model.fit(data_train, labels_train)  # , sample_weight=sample_weight)
    model_per_fold.append(model)
    results_per_fold.append(results)

    print("Confusion matrix TRAIN logit score: ")
    print(confusion_matrix(labels_train, (model.predict(results.params, data_train)>0.5).astype(np.int32)))
    #print(confusion_matrix(labels_train, (model.predict_proba(data_train)[:, 1] > 0.5).astype(np.int32)))

    print("Confusion matrix TEST logit score: ")
    print(confusion_matrix(labels_test, (model.predict(results.params, data_test)>0.5).astype(np.int32)))
    #print(confusion_matrix(labels_test, (model.predict_proba(data_test)[:, 1] > 0.5).astype(np.int32)))
    y_pred = model.predict(results.params, data_test)
    y_pred_train = model.predict(results.params, data_train)
    #y_pred = model.predict_proba(data_test)[:, 1]
    fpr_train, tpr_train, _ = roc_curve(labels_train, y_pred_train)
    precision_train, recall_train, _ = precision_recall_curve(labels_train, y_pred_train)
    auc_train = auc(fpr_train, tpr_train)
    auc_pr_train = auc(recall_train, precision_train)
    print("AUC ROC TRAIN logit score: ", auc_train)
    print("AUC PR TRAIN logit score: ", auc_pr_train)
    plt.plot(fpr_train, tpr_train, lw=2, label='Train ROC curve fold '+str(i) + ' AUC: '+str(np.round(auc_train, 2)))
    plt.plot(recall_train, precision_train, lw=2, label='Train PR curve fold ' + str(i) + ' AUC: '+str(np.round(auc_pr_train, 2)))
    fpr, tpr, thresholds = roc_curve(labels_test, y_pred)
    auc_ = roc_auc_score(labels_test, y_pred)
    roc_curves.append([fpr, tpr, thresholds, auc_])
    predictions_per_fold.append(y_pred)
    coefs_per_fold.append(results.params)

    #pred = results.get_prediction(data_test)
    #pred_summary = pred.summary_frame(alpha=0.05)  # 95% confidence intervals
    #
    # # Extract confidence intervals
    #ci_lower = pred_summary['mean_ci_lower']
    #ci_upper = pred_summary['mean_ci_upper']
    # ci_lower_per_fold.append(ci_lower)
    # ci_upper_per_fold.append(ci_upper)
    #coefs_per_fold.append(np.concatenate([results.intercept_,results.coef_[0]]))

    # importances = permutation_importance(model, data_test, labels_test, n_repeats=50)
    # importance_per_fold.append(importances.importances_mean)
plt.legend(prop={'size': 7})
plt.show()
def compute_and_plot_roc(labels, predictions, name, labels_per_fold=None, predictions_per_fold=None):
    # Compute ROC AUC
    fpr, tpr, _ = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    has_folds = labels_per_fold is not None and predictions_per_fold is not None
    if has_folds:
        for i, (labels, predictions) in enumerate(zip(labels_per_fold, predictions_per_fold)):
            fpr_fold, tpr_fold, _ = roc_curve(labels, predictions)
            roc_auc_fold = auc(fpr_fold, tpr_fold)
            plt.plot(fpr_fold, tpr_fold, lw=lw, label=f'Split {i + 1}. ROC AUC = %0.3f' % roc_auc_fold)
    plt.plot(fpr, tpr, color='black', lw=lw, label='Cross-validated. ROC (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title('Receiver Operating Characteristic '+name)
    plt.legend(loc="lower right")
    plt.show()

def compute_and_plot_pr(labels, predictions, name, labels_per_fold=None, predictions_per_fold=None):
    # Compute PR AUC
    precision, recall, _ = precision_recall_curve(labels, predictions)
    pr_auc = auc(recall, precision)
    plt.figure()
    lw = 2
    has_folds = labels_per_fold is not None and predictions_per_fold is not None
    if has_folds:
        for i, (labels, predictions) in enumerate(zip(labels_per_fold, predictions_per_fold)):
            precision_fold, recall_fold, _ = precision_recall_curve(labels, predictions)
            pr_auc_fold = auc(recall_fold, precision_fold)
            plt.plot(recall_fold, precision_fold, lw=lw, label=f'Split {i + 1}. PR AUC = %0.3f' % pr_auc_fold)
    plt.plot(recall, precision, color='black', lw=lw, label='Cross-validated. PR AUC (area = %0.3f)' % pr_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve '+name)
    plt.legend()
    plt.show()

all_predictions = np.concatenate(predictions_per_fold)
#Takotsubo
all_labels = np.concatenate(labels_fold_ids)

compute_and_plot_roc(all_labels, all_predictions, name="logit_"+str(step)+"step", labels_per_fold=labels_fold_ids,
    predictions_per_fold=predictions_per_fold)

compute_and_plot_pr(all_labels, all_predictions, name="logit_"+str(step)+"step", labels_per_fold=labels_fold_ids,
    predictions_per_fold=predictions_per_fold)

importance_per_split = False
if importance_per_split:
    for i, perm_importance in enumerate(importance_per_fold):
        features = np.arange(data_test.shape[1])
        sorted_idx = perm_importance.argsort()
        ordered_features = list(map(str, features[sorted_idx]))
        plt.barh(ordered_features[-30:], perm_importance[sorted_idx][-30:])
        plt.xlabel("Permutation Importance split: "+str(i+1))
        plt.show()

    for i, perm_importance in enumerate(importance_per_fold):
        features = np.arange(data_test.shape[1])
        sorted_idx = perm_importance.argsort()
        d__ = (data[np.where(labels)[0][0]] * std_ + mean_).T
        plt.plot(d__, 'r')
        plt.vlines(sorted_idx[-15:], np.min(d__), np.max(d__), 'b')
        plt.vlines(np.arange(0, ps.shape[0] * len(leads), ps.shape[0]), np.min(d__) * 0.5, np.max(d__) * 0.5, 'k',
                   linestyles='dashed', alpha=0.3)
        plt.xlabel("Location of important features for split: " + str(i + 1))
        plt.show()

perm_importance_all = np.zeros(data.shape[1])
perm_importance_all_abs = np.zeros(data.shape[1])
for perm_importance in coefs_per_fold:
    perm_importance_all_abs = perm_importance_all_abs + np.abs(perm_importance[1:])
    perm_importance_all = perm_importance_all + perm_importance[1:]
features = np.arange(data_test.shape[1])
sorted_idx = perm_importance_all_abs.argsort()
ordered_features = list(map(str, features[sorted_idx]))
plt.barh(ordered_features, perm_importance_all_abs[sorted_idx])
plt.xlabel("Permutation Importance All")
plt.show()

plt.imshow(np.array(coefs_per_fold).T, aspect='auto')
plt.colorbar()
plt.show()

#TS vs suspected MI
# name_vars = ['Intercept', '1-I', '2-I', '3-I', '1-II', '2-II', '3-II', '4-II', '5-II', '6-II', '1-V1', '1-V2',
#              '2-V2', '3-V2', '1-V4', '2-V4', '1-V5', '1-V6',
#              'Num of clusters', 'Num of samples', 'Sex']
#TS vs verified MI
# name_vars = ['Intercept', '1-I', '2-I', '3-I', '1-II', '2-II', '3-II', '4-II', '5-II', '6-II', '7-II', '8-II',
#              '1-V2', '2-V2', '1-V4', '2-V4', '1-V5', '2-V5', '1-V6', '2-V6',
#              'Num of clusters', 'Num of samples', 'Sex']
#TS vs NSTEMI
# name_vars = ['Intercept', '1-I', '2-I', '3-I', '4-I', '5-I', '6-I', '1-II', '2-II', '3-II', '4-II',
#              '1-V2', '2-V2', '3-V2', '4-V2', '1-V4', '2-V4', '3-V4', '1-V5', '1-V6',
#              'Num of clusters', 'Num of samples', 'Sex']
#TS vs STEMI
# name_vars = ['Intercept', '1-I', '2-I', '1-II', '2-II', '3-II', '4-II', '5-II', '1-V1', '1-V2',
#              '1-V3', '2-V3', '3-V3', '4-V3', '1-V4', '2-V4', '3-V4',
#              '1-V5', '2-V5', '3-V5', 'Num of clusters', 'Num of samples', 'Sex']
#ATS vs ASTEMI
# name_vars = ['Intercept', '1-I', '2-I', '3-I', '4-I', '5-I', '6-I', '1-II', '2-II', '3-II',
#              '1-V1', '2-V1', '3-V1', '4-V1', '5-V1', '1-V2', '2-V2', '1-V3', '2-V3', '3-V3',
#              '1-V4', '2-V4', '3-V4', '4-V4', '1-V5',
#              'Num of clusters', 'Num of samples', 'Sex']

#TS vs STEMI (male)
# name_vars = ['Intercept', '1-I', '2-I', '1-II', '2-II', '3-II', '4-II', '5-II', '6-II', '7-II',
#              '1-V1', '2-V1', '3-V1', '4-V1', '1-V2', '2-V2', '3-V2', '4-V2', '1-V3',
#              '1-V4', '1-V5', '2-V5', '3-V5', '1-V6', '2-V6', '3-V6',
#              'Num of clusters', 'Num of samples']
#TS vs STEMI (modified)
# name_vars = ['Intercept', '1-I - 3-II', '2-I', '1-II', '2-II', '(5-II - 4-II) - 3-II', '1-V1',
#              '1-V3', '2-V3', '3-V3 - 1-V2', '4-V3', '1-V4', '2-V4', '3-V4',
#              '3-V5 - 1-V5', '2-V5', 'Num of clusters', 'Num of samples', 'Sex']
#name_vars = ['Intercept', '(5-II - 4-II) - 3-II', '3-V5 - 1-V5']
#name_vars = ['Intercept', '1-I - 3-II', '1-II', '2-II']
#TS vs suspected MI (modified)
name_vars = ['Intercept', '1-I - 2-I', '3-I', '1-II', '2-II','4-II + 5-II + 6-II - 3-II', '1-V1', '1-V2',
             '2-V2', '3-V2', '1-V4', '2-V4', '1-V5', '1-V6',
             'Num of clusters', 'Num of samples', 'Sex']
#ATS vs ASTEMI (modified)
# name_vars = ['Intercept', '1-I - 2-II', '3-I - 2-I', '4-I', '6-I', '1-II', '3-II - 5-I',
#              '1-V1 - 2V1', '3-V1', '5-V1 - 4-V1', '1-V2', '2-V2', '1-V3', '2-V3', '3-V3',
#              '1-V4', '2-V4', '3-V4', '4-V4', '1-V5',
#              'Num of clusters', 'Num of samples', 'Sex']
#TS vs verified MI (modified)
# name_vars = ['Intercept', '1-I', '2-I', '3-I - 2-II', '1-II', '3-II', '4-II', '6-II - 5-II', '7-II', '8-II',
#              '1-V2', '2-V2', '1-V4', '2-V4', '1-V5', '2-V5 - 1-V6', '2-V6',
#              'Num of clusters', 'Num of samples', 'Sex']
#TS vs NSTEMI
# name_vars = ['Intercept', '1-I', '2-I', '4-I - 3-I', '5-I', '6-I', '1-II', '2-II', '4-II - 3-II',
#              '1-V2', '2-V2', '3-V2', '4-V2', '1-V4', '2-V4', '3-V4', '1-V5', '1-V6',
#              'Num of clusters', 'Num of samples', 'Sex']

df_odds = pd.DataFrame(columns=['Variable', 'Coefficients', 'Odds ratio', 'P-values'])
df_odds['Variable'] = name_vars
p_values = np.array([r.pvalues for r in results_per_fold])
mean_p_values = np.mean(p_values, axis=0)
mean_p_values = np.round(np.array([p if p > 0.001 else 0.001 for p in mean_p_values]), 3)
coefs = np.mean(np.array(coefs_per_fold),axis=0)
odr = np.round(np.exp(coefs), 3)
df_odds['Coefficients'] = coefs
df_odds['Odds ratio'] = odr
df_odds['P-values'] = mean_p_values

#This is everything focused on TS vs STEMI:
#P_wave analysis
data_reduced = data
P_wave_vars = np.dot(data_reduced[:, np.array([0,2,3])], coefs_per_fold[0][np.array([1,3,4])])
print(np.argsort(P_wave_vars)[::-1][:20])
print(labels[np.argsort(P_wave_vars)[::-1][:20]])
print(taken[np.argsort(P_wave_vars)[::-1][:20]])

#Wilcoxon test as the variables do not follow a Normal distribution
from scipy.stats import mannwhitneyu
group_t = data_reduced[np.where(labels==1)[0],np.array([0])]
group_c = data_reduced[np.where(labels==0)[0],np.array([0])]
stat, p = mannwhitneyu(group_t, group_c)
print('Mean group TS: ', np.mean(group_t), ' Mean group STEMI:', np.mean(group_c))
print('Mann-Whitney U: stat', stat, ' P-value', p)

#ST-elevation study
data_reduced = data
ST_elevation_vars = np.dot(data_reduced[:, np.array([4,8])], coefs_per_fold[0][np.array([5,9])])
print(np.argsort(ST_elevation_vars)[::-1][:20])
print(labels[np.argsort(ST_elevation_vars)[::-1][:20]])
print(taken[np.argsort(ST_elevation_vars)[::-1][:20]])

#ST-depression study
data_reduced = data
ST_depression_vars = np.dot(data_reduced[:, np.array([1])], coefs_per_fold[0][np.array([2])])
print(np.argsort(ST_depression_vars)[:20])
print(labels[np.argsort(ST_depression_vars)[:20]])
print(taken[np.argsort(ST_depression_vars)[:20]])

#QTc study
data_reduced = data
QTc_vars = np.dot(data_reduced[:, np.array([9])], coefs_per_fold[0][np.array([10])])
print(np.argsort(QTc_vars)[::-1][:20])
print(labels[np.argsort(QTc_vars)[::-1][:20]])
print(taken[np.argsort(QTc_vars)[::-1][:20]])

#Q wave study
data_reduced = data
Q_wave_vars = np.dot(data_reduced[:, np.array([6])], coefs_per_fold[0][np.array([7])])
print(np.argsort(Q_wave_vars)[::-1][:20])
print(labels[np.argsort(Q_wave_vars)[::-1][:20]])
print(taken[np.argsort(Q_wave_vars)[::-1][:20]])


#Influence study
infl = results_per_fold[0].get_influence(observed=False)
summ_df = infl.summary_frame()
print(summ_df.sort_values("cooks_d", ascending=False)[:10])
fig = infl.plot_influence()
fig.tight_layout(pad=1.0)
plt.show()

#Try to perform dimensional reduction using the coefficients of logit regression.
#data_reduced = data
all_ids = np.concatenate(data_fold_ids).squeeze()
#labels_ = labels + labels_stemi[taken] * 2 + labels_nstemi[taken] * 3
#labels_[np.where(labels_ > 3.0)] = np.ones(np.where(labels_ > 3.0)[0].shape) * 3.0
labels_ = labels
labels_ = labels_[all_ids]
labels_test_pred = (all_predictions > 0.5).astype(np.int32)
data_reduced = data[all_ids]
clf = LDA()
clf.fit(data_reduced, labels_)
data_lda = clf.transform(data_reduced)
data_lda = np.hstack([data_lda, np.dot(data_reduced, coefs_per_fold[0][1:])[:,np.newaxis]])
out = cwd + "/plots/Run_" + time.asctime().replace(" ", "_").replace(":", "-") + "_Rec_tak_"
plot_scatter(data_lda[:, 0], data_lda[:, 1], lab_train=labels_.astype(np.int32),
             save=out+"scatter_LDA.html", explore=True)

#plot UMAP
all_ids = np.concatenate(data_fold_ids).squeeze()
#labels_ = labels + labels_stemi[taken] * 2 + labels_nstemi[taken] * 3
#labels_[np.where(labels_ > 3.0)] = np.ones(np.where(labels_ > 3.0)[0].shape) * 3.0
labels_ = labels
labels_ = labels_[all_ids]
labels_test_pred = (all_predictions > 0.5).astype(np.int32)
data_reduced = ((data - mean_)/std_)[all_ids]
clusterable_embedding = umap.UMAP(
        n_neighbors=400,
        n_components=2,
        random_state=42,
    ).fit(data_reduced)
map_train = clusterable_embedding.transform(data_reduced)
cp_umap = [0,1]
out = cwd + "/plots/Run_" + time.asctime().replace(" ", "_").replace(":", "-") + "_Rec_tak_"
plot_scatter(map_train[:, cp_umap[0]], map_train[:, cp_umap[1]], map_train[:, cp_umap[0]], map_train[:, cp_umap[1]], lab_train=labels_.astype(np.int32), lab_test_pred=labels_test_pred,
                 save=out+"scatter_UMAP_train.html", explore=False)


print("END")
