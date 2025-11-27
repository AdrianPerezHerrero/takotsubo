#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logistic Regression Analysis for Takotsubo Syndrome Classification

This script performs logistic regression analysis to discriminate between
Takotsubo syndrome and different cardiac conditions (STEMI, NSTEMI, etc.).
Multiple clinical scenarios are supported through configuration.

Author: Adrian Perez-Herrero
Project: Takotsubo Syndrome Classification
Reference: Takotsubo_classification_NPJ paper
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import statsmodels.api as sm
import time

from scipy.ndimage import gaussian_filter1d
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve,
    roc_curve, roc_auc_score, auc
)
import umap

# ============================================================================
# Configuration
# ============================================================================

dtype = torch.float64
torch.set_default_dtype(dtype)

# ============================================================================
# SCENARIO SELECTION
# ============================================================================
# Choose one of the following scenarios:
# 1. 'TS_VS_SUSPECTED_MI'     - Takotsubo vs Suspected Myocardial Infarction
# 2. 'TS_VS_STEMI'            - Takotsubo vs ST-Elevation MI
# 3. 'TS_VS_NSTEMI'           - Takotsubo vs Non-ST-Elevation MI
# 4. 'TS_VS_VERIFIED_MI'      - Takotsubo vs Verified MI cases
# 5. 'APICAL_TS_VS_ANT_STEMI' - Apical Takotsubo vs Anterior STEMI
# 6. 'TS_VS_STEMI_MALE'       - Takotsubo vs STEMI (males only)
# 7. 'STEMI_VS_NSTEMI'        - STEMI vs NSTEMI classification

SCENARIO = 'APICAL_TS_VS_ANT_STEMI'
#Apply interactions between important features
APPLY_INTERACTIONS = True

# ============================================================================
# Scenario Configurations
# ============================================================================

SCENARIO_CONFIGS = {
    'TS_VS_SUSPECTED_MI': {
        'description': 'Takotsubo vs Suspected Myocardial Infarction',
        'important_features': np.array([
            16, 29, 170, 329, 401, 423, 438, 470, 517, 704, 950,
            1006, 1053, 1550, 1702, 1910, 2215
        ]),
        'interactions': [
            # newvar = data[:, 0] - data[:, 1]
            {'target_idx': 0, 'operation': 'subtract', 'operands': [0, 1]},
            # newvar_2 = data[:, 6] + data[:, 7] + data[:, 8] - data[:, 5]
            {'target_idx': 5, 'operation': 'add_then_subtract', 'add_operands': [6, 7, 8], 'subtract_operand': 5},
        ],
        'delete_after_interaction': [1, 6, 7, 8],
        'feature_names_original': [
            'Intercept', '1-I', '2-I', '3-I', '1-II', '2-II', '3-II', '4-II',
            '5-II', '6-II', '1-V1', '1-V2', '2-V2', '3-V2', '1-V4', '2-V4',
            '1-V5', '1-V6', 'Num of clusters', 'Num of samples', 'Sex'
        ],
        'feature_names': [
            'Intercept', '1-I - 2-I', '3-I', '1-II', '2-II',
            '(4-II + 5-II + 6-II) - 3-II', '1-V1', '1-V2',
            '2-V2', '3-V2', '1-V4', '2-V4', '1-V5', '1-V6',
            'Num of clusters', 'Num of samples', 'Sex'
        ],
        'cohort_selection': 'takotsubo_and_all_mi',
        'gender_filter': None,
        'expected_roc_auc': 0.89,
        'expected_pr_auc': 0.28
    },

    'TS_VS_STEMI': {
        'description': 'Takotsubo vs ST-Elevation Myocardial Infarction',
        'important_features': np.array([
            16, 163, 311, 328, 362, 442, 473, 654, 965, 1267, 1290,
            1361, 1450, 1556, 1605, 1614, 1814, 1921, 1939
        ]),
        'interactions': [
            # newvar_4 = data[:, 0] - data[:, 4]
            {'target_idx': 0, 'operation': 'subtract', 'operands': [0, 4]},
            # newvar = data[:, 6] - data[:, 5] - data[:, 4]
            {'target_idx': 5, 'operation': 'subtract_cascade', 'operands': [6, 5, 4]},
            # newvar_3 = data[:, 11] - data[:, 8]
            {'target_idx': 11, 'operation': 'subtract', 'operands': [11, 8]},
            # newvar_2 = data[:, 18] - data[:, 16]
            {'target_idx': 16, 'operation': 'subtract', 'operands': [18, 16]},
        ],
        'delete_after_interaction': [4, 6, 8, 18],
        'feature_names_original': [
            'Intercept', '1-I', '2-I', '1-II', '2-II', '3-II', '4-II', '5-II',
            '1-V1', '1-V2', '1-V3', '2-V3', '3-V3', '4-V3', '1-V4', '2-V4',
            '3-V4', '1-V5', '2-V5', '3-V5',
            'Num of clusters', 'Num of samples', 'Sex'
        ],
        'feature_names': [
            'Intercept', '1-I - 3-II', '2-I', '1-II', '2-II',
            '(5-II - 4-II) - 3-II', '1-V1', '1-V3', '2-V3',
            '3-V3 - 1-V2', '4-V3', '1-V4', '2-V4', '3-V4',
            '3-V5 - 1-V5', '2-V5',
            'Num of clusters', 'Num of samples', 'Sex'
        ],
        'cohort_selection': 'takotsubo_and_stemi',
        'gender_filter': None,
        'expected_roc_auc': 0.94,
        'expected_pr_auc': 0.84
    },

    'TS_VS_NSTEMI': {
        'description': 'Takotsubo vs Non-ST-Elevation Myocardial Infarction',
        'important_features': np.array([
            35, 99, 151, 165, 170, 182, 346, 362, 432, 457, 954,
            1005, 1046, 1114, 1534, 1550, 1703, 1935, 2155
        ]),
        'interactions': [
            # newvar = data[:, 3] - data[:, 2]
            {'target_idx': 3, 'operation': 'subtract', 'operands': [3, 2]},
            # newvar_ = data[:, 9] - data[:, 8]
            {'target_idx': 9, 'operation': 'subtract', 'operands': [9, 8]},
        ],
        'delete_after_interaction': [2, 8],
        'feature_names_original': [
            'Intercept', '1-I', '2-I', '3-I', '4-I', '5-I', '6-I',
            '1-II', '2-II', '3-II', '4-II',
            '1-V2', '2-V2', '3-V2', '4-V2', '1-V4', '2-V4', '3-V4',
            '1-V5', '1-V6',
            'Num of clusters', 'Num of samples', 'Sex'
        ],
        'feature_names': [
            'Intercept', '1-I', '2-I', '4-I - 3-I', '5-I', '6-I',
            '1-II', '2-II', '4-II - 3-II',
            '1-V2', '2-V2', '3-V2', '4-V2', '1-V4', '2-V4', '3-V4',
            '1-V5', '1-V6',
            'Num of clusters', 'Num of samples', 'Sex'
        ],
        'cohort_selection': 'takotsubo_and_nstemi',
        'gender_filter': None,
        'expected_roc_auc': 0.91,
        'expected_pr_auc': 0.61
    },


    'TS_VS_VERIFIED_MI': {
        'description': 'Takotsubo vs Verified Myocardial Infarction',
        'important_features': np.array([
            16, 29, 170, 329, 401, 423, 438, 470, 517, 704, 950,
            1006, 1053, 1550, 1702, 1910, 2215
        ]),
        'interactions': [
            # newvar = data[:, 2] - data[:, 4]
            {'target_idx': 2, 'operation': 'subtract', 'operands': [2, 4]},
            # newvar_ = data[:, 8] - data[:, 7]
            {'target_idx': 8, 'operation': 'subtract', 'operands': [8, 7]},
            # newvar_2 = data[:, 16] - data[:, 17]
            {'target_idx': 16, 'operation': 'subtract', 'operands': [16, 17]},
        ],
        'delete_after_interaction': [4, 7, 17],
        'feature_names_original': [
            'Intercept', '1-I', '2-I', '3-I', '1-II', '2-II', '3-II', '4-II',
            '5-II', '6-II', '7-II', '8-II',
            '1-V2', '2-V2', '1-V4', '2-V4', '1-V5', '2-V5', '1-V6', '2-V6',
            'Num of clusters', 'Num of samples', 'Sex'
        ],
        'feature_names': [
            'Intercept', '1-I', '2-I', '3-I - 2-II', '1-II', '3-II',
            '4-II', '6-II - 5-II', '7-II', '8-II',
            '1-V2', '2-V2', '1-V4', '2-V4', '1-V5', '2-V5 - 1-V6', '2-V6',
            'Num of clusters', 'Num of samples', 'Sex'
        ],
        'cohort_selection': 'takotsubo_and_verified',
        'gender_filter': None,
        'expected_roc_auc': 0.92,
        'expected_pr_auc': 0.53
    },

    'APICAL_TS_VS_ANT_STEMI': {
        'description': 'Apical Takotsubo vs Anterior STEMI',
        'important_features': np.array([
            135, 150, 164, 241, 283, 289, 319, 421, 446, 632, 717,
            722, 757, 780, 929, 954, 1200, 1250, 1347, 1552, 1600,
            1603, 1608, 1801
        ]),
        'interactions': [
            # newvar = data[:, 2] - data[:, 1]
            {'target_idx': 1, 'operation': 'subtract', 'operands': [2, 1]},
            # newvar_ = data[:, 0] - data[:, 7]
            {'target_idx': 0, 'operation': 'subtract', 'operands': [0, 7]},
            # newvar_2 = data[:, 8] - data[:, 4]
            {'target_idx': 8, 'operation': 'subtract', 'operands': [8, 4]},
            # newvar_3 = data[:, 14] - data[:, 13]
            {'target_idx': 14, 'operation': 'subtract', 'operands': [14, 13]},
            # newvar_4 = data[:, 10] - data[:, 11]
            {'target_idx': 10, 'operation': 'subtract', 'operands': [10, 11]},
            # newvar_5 = data[:, 18] - data[:, 17]
            {'target_idx': 18, 'operation': 'subtract', 'operands': [18, 17]},
        ],
        'delete_after_interaction': [2, 4, 7, 11, 13, 17],
        'feature_names_original': [
            'Intercept', '1-I', '2-I', '3-I', '4-I', '5-I', '6-I',
            '1-II', '2-II', '3-II',
            '1-V1', '2-V1', '3-V1', '4-V1', '5-V1', '1-V2', '2-V2',
            '1-V3', '2-V3', '3-V3',
            '1-V4', '2-V4', '3-V4', '4-V4', '1-V5',
            'Num of clusters', 'Num of samples', 'Sex'
        ],
        'feature_names': [
            'Intercept', '1-I - 2-II', '3-I - 2-I', '4-I', '6-I', '1-II',
            '3-II - 5-I', '1-V1 - 2-V1', '3-V1', '5-V1 - 4-V1',
            '1-V2', '2-V2', '1-V3', '2-V3', '3-V3',
            '1-V4', '2-V4', '3-V4', '4-V4', '1-V5',
            'Num of clusters', 'Num of samples', 'Sex'
        ],
        'cohort_selection': 'apical_ts_and_ant_stemi',
        'gender_filter': None,
        'expected_roc_auc': 0.90,
        'expected_pr_auc': 0.71
    },

    'TS_VS_STEMI_MALE': {
        'description': 'Takotsubo vs STEMI (Males Only)',
        'important_features': np.array([
            95, 159, 317, 332, 368, 390, 421, 427, 572, 622, 635,
            757, 797, 928, 989, 1046, 1082, 1257, 1563, 1813, 1911,
            2038, 2106, 2216, 2348
        ]),
        'interactions': [],
        'delete_after_interaction': [],
        'feature_names_original': [
            'Intercept', '1-I', '2-I', '1-II', '2-II', '3-II', '4-II',
            '5-II', '6-II', '7-II', '1-V1', '2-V1', '3-V1', '4-V1',
            '1-V2', '2-V2', '3-V2', '4-V2', '1-V3',
            '1-V4', '1-V5', '2-V5', '3-V5', '1-V6', '2-V6', '3-V6',
            'Num of clusters', 'Num of samples'
        ],
        'feature_names': [
            'Intercept', '1-I', '2-I', '1-II', '2-II', '3-II', '4-II',
            '5-II', '6-II', '7-II', '1-V1', '2-V1', '3-V1', '4-V1',
            '1-V2', '2-V2', '3-V2', '4-V2', '1-V3',
            '1-V4', '1-V5', '2-V5', '3-V5', '1-V6', '2-V6', '3-V6',
            'Num of clusters', 'Num of samples'
        ],
        'cohort_selection': 'takotsubo_and_stemi',
        'gender_filter': 'male',
        'expected_roc_auc': 0.790,
        'expected_pr_auc': 0.068
    },

    'STEMI_VS_NSTEMI': {
        'description': 'STEMI vs NSTEMI Classification',
        'important_features': np.array([
            20, 22, 25, 59, 97, 110, 127, 174, 176, 181, 224,
            236, 272, 278, 342, 352, 483, 591, 597, 660, 757, 823,
            927, 1007, 1039, 1044, 1113, 1300, 1305, 1407, 1500,
            1604, 1800, 1857, 1949
        ]),
        'interactions': [],
        'delete_after_interaction': [],
        'feature_names_original': [
            'Intercept', 'ECG-1', 'ECG-2', 'ECG-3', 'ECG-4', 'ECG-5',
            'ECG-6', 'ECG-7', 'ECG-8', 'ECG-9', 'ECG-10', 'ECG-11',
            'ECG-12', 'ECG-13', 'ECG-14', 'ECG-15', 'ECG-16', 'ECG-17',
            'ECG-18', 'ECG-19', 'ECG-20', 'ECG-21', 'ECG-22', 'ECG-23',
            'ECG-24', 'ECG-25', 'ECG-26', 'ECG-27', 'ECG-28', 'ECG-29',
            'ECG-30', 'ECG-31', 'ECG-32', 'ECG-33', 'ECG-34', 'ECG-35',
            'Num of clusters', 'Num of samples', 'Sex'
        ],
        'feature_names': [
            'Intercept', 'ECG-1', 'ECG-2', 'ECG-3', 'ECG-4', 'ECG-5',
            'ECG-6', 'ECG-7', 'ECG-8', 'ECG-9', 'ECG-10', 'ECG-11',
            'ECG-12', 'ECG-13', 'ECG-14', 'ECG-15', 'ECG-16', 'ECG-17',
            'ECG-18', 'ECG-19', 'ECG-20', 'ECG-21', 'ECG-22', 'ECG-23',
            'ECG-24', 'ECG-25', 'ECG-26', 'ECG-27', 'ECG-28', 'ECG-29',
            'ECG-30', 'ECG-31', 'ECG-32', 'ECG-33', 'ECG-34', 'ECG-35',
            'Num of clusters', 'Num of samples', 'Sex'
        ],
        'cohort_selection': 'stemi_and_nstemi',
        'gender_filter': None,
        'expected_roc_auc': 0.850,
        'expected_pr_auc': 0.750
    }
}

# Analysis parameters
STANDARDIZE = True
INCLUDE_ST_DEV = False
INCLUDE_CLUSTERS = True
INCLUDE_SAMPLES = True
INCLUDE_AGE = False
INCLUDE_SEX = True


# ECG parameters
MAX_SAMPLE = 200
MIN_SAMPLE = -100
STEP_SIZE = 1


# ============================================================================
# Directory Setup
# ============================================================================

def setup_directories():
    """
    Configure working directories based on operating system.

    Returns:
        tuple: (homedir, cwd) paths
    """
    homedir = os.getenv('HOME')

    if homedir is None:
        # Windows environment
        homedir = "C:/Users/Adrian/Projects/Thesis/Takotsubo"
        os.chdir(os.path.join(homedir, "takotsubo_data/HDP-SLDS-GP"))
    else:
        # Unix-like environment
        try:
            os.chdir(homedir + "/Documents/takotsubo_data/HDP-SLDS-GP")
            homedir = homedir + "/Documents"
        except FileNotFoundError:
            aux_dir = input("Please specify directory:\n")
            os.chdir(aux_dir)

    return homedir, os.getcwd()


# ============================================================================
# Data Loading
# ============================================================================

def load_clinical_data(homedir):
    """
    Load clinical and ECG data.

    Args:
        homedir: Home directory path

    Returns:
        dict: Dictionary containing all loaded data
    """
    data_dir = os.path.join(homedir, "takotsubo_data", "data","takotsubo","full_dataset_filtered")

    data = {
        'age': np.load(os.path.join(data_dir, "data_age.npy")),
        'labels_takotsubo': np.load(os.path.join(data_dir, "labels_takotsubo.npy")),
        'labels_sex': np.load(os.path.join(data_dir, "labels_sex.npy")),
        'filepaths': np.load(os.path.join(data_dir, "data_filepaths.npy")),
        'ecg_features': np.load(os.path.join(data_dir, "saved_parameters","pure_mean_6.npy")),
        'ecg_std': np.load(os.path.join(data_dir, "saved_parameters","diag_sig_6.npy")),
        'num_clusters': np.load(os.path.join(data_dir, "saved_parameters","num_clust_6.npy")),
        'num_beats': np.load(os.path.join(data_dir, "num_beats.npy"))
    }

    return data


def load_diagnosis_labels(homedir, filepaths):
    """
    Load diagnosis labels for different cardiac conditions.

    Args:
        homedir: Home directory path
        filepaths: Array of file paths for each patient

    Returns:
        tuple: (labels_dict, corrected_filepaths)
    """
    scripts_dir = os.path.join(homedir, "takotsubo_data","data","takotsubo","STEMI_scripts")
    aptika_dir = os.path.join(homedir, "takotsubo_data","data","takotsubo","AptikaAntstemiPlot","source")

    # Helper function to load ID lists
    def load_id_list(filepath):
        with open(filepath, 'r') as f:
            return set(line.strip() for line in f.readlines())

    # Helper function to extract ID from filepath
    def extract_id(filepath):
        file_parts = filepath.split('/')
        details = file_parts[1].split('_')
        return details[2]

    # Fix filepath format
    filepaths = np.array([
        f.replace('Cases level 1', 'Cases level 3') if f.startswith('Cases level 1')
        else f for f in filepaths
    ])

    # Extract IDs
    patient_ids = [extract_id(f) for f in filepaths]

    # Load condition ID sets
    id_sets = {
        'stemi': load_id_list(os.path.join(scripts_dir, "stemi.txt")),
        'nstemi': load_id_list(os.path.join(scripts_dir, "nstemi.txt")),
        'ant_stemi': load_id_list(os.path.join(aptika_dir, "ant_stemi.txt")),
        'apical_ts': load_id_list(os.path.join(aptika_dir, "aptical.txt"))
    }

    # Create label arrays
    labels = {}
    for condition, id_set in id_sets.items():
        labels[condition] = np.array([1 if pid in id_set else 0 for pid in patient_ids])

    # Create combined labels
    labels['stemi_nstemi'] = labels['stemi'] + labels['nstemi']
    labels['stemi_nstemi'][labels['stemi_nstemi'] > 1] = 1

    labels['verified_mi'] = labels['stemi_nstemi']  # Alias for verified cases

    return labels, filepaths


# ============================================================================
# Feature Engineering
# ============================================================================

def prepare_features(data, leads=[0, 1, 2, 3, 4, 5, 6, 7],
                     include_sex=True, include_clusters=True,
                     include_samples=True, include_age=False):
    """
    Prepare feature matrix from ECG and clinical data.

    Args:
        data: Dictionary of loaded data
        leads: List of lead indices to include
        include_sex: Whether to include sex as feature
        include_clusters: Whether to include number of clusters
        include_samples: Whether to include number of samples
        include_age: Whether to include age

    Returns:
        numpy array: Feature matrix
    """
    # Extract ECG features for selected leads
    ecg_features = data['ecg_features'][:, :, leads]

    # Select time samples
    samples = np.arange(0, MAX_SAMPLE - MIN_SAMPLE, STEP_SIZE)
    ecg_features = ecg_features[:, samples, :]

    # Reshape to 2D (patients x features)
    features = np.reshape(
        np.transpose(ecg_features, (0, 2, 1)),
        (ecg_features.shape[0], -1)
    ) / 100.0

    # Add additional features
    if include_clusters:
        features = np.concatenate([features, data['num_clusters'][:, np.newaxis]], axis=1)

    if include_samples:
        features = np.concatenate([features, data['num_beats'][:, np.newaxis]], axis=1)

    if include_sex:
        features = np.concatenate([features, data['labels_sex'][:, np.newaxis]], axis=1)

    if include_age:
        features = np.concatenate([features, data['age'][:, np.newaxis]], axis=1)

    return features


def select_important_features(features, feature_indices, num_ecg_features):
    """
    Select subset of features based on importance indices.

    Args:
        features: Full feature matrix
        feature_indices: Indices of important features to select

    Returns:
        numpy array: Selected features
    """

    ecg_only = features[:, :num_ecg_features]
    additional_features = features[:, num_ecg_features:]

    # Select important ECG features
    selected_ecg = ecg_only[:, feature_indices]

    # Concatenate selected ECG features with additional features
    selected_features = np.concatenate([selected_ecg, additional_features], axis=1)

    return selected_features


def apply_interactions(features, interactions, delete_indices):
    """
    Create interaction features based on specified combinations.

    Args:
        features: Feature matrix
        interactions: List of interaction dictionaries with target_idx, operation, and operands
        delete_indices: Indices to delete after creating interactions

    Returns:
        numpy array: Modified feature matrix
    """
    if len(interactions) == 0:
        return features

    modified_features = features.copy()

    for interaction in interactions:
        target_idx = interaction['target_idx']
        operation = interaction['operation']

        if operation == 'subtract':
            # Simple subtraction: operands[0] - operands[1]
            operands = interaction['operands']
            new_feature = modified_features[:, operands[0]] - modified_features[:, operands[1]]

        elif operation == 'subtract_cascade':
            # Cascade subtraction: operands[0] - operands[1] - operands[2]
            operands = interaction['operands']
            new_feature = modified_features[:, operands[0]]
            for idx in operands[1:]:
                new_feature -= modified_features[:, idx]

        elif operation == 'add_then_subtract':
            # Sum add_operands, then subtract subtract_operand
            # Example: (6 + 7 + 8) - 5
            add_operands = interaction['add_operands']
            subtract_operand = interaction['subtract_operand']
            new_feature = np.sum(modified_features[:, add_operands], axis=1)
            new_feature -= modified_features[:, subtract_operand]

        elif operation == 'add':
            # Simple addition
            operands = interaction['operands']
            new_feature = np.sum(modified_features[:, operands], axis=1)

        elif operation == 'multiply':
            # Product of operands
            operands = interaction['operands']
            new_feature = np.prod(modified_features[:, operands], axis=1)

        # Replace target index with new feature
        modified_features[:, target_idx] = new_feature

    # Remove redundant features
    if len(delete_indices) > 0:
        modified_features = np.delete(modified_features, delete_indices, axis=1)

    return modified_features


# ============================================================================
# Cohort Selection
# ============================================================================

def select_cohort(data, diagnosis_labels, config):
    """
    Select patient cohort based on scenario configuration.

    Args:
        data: Dictionary of clinical data
        diagnosis_labels: Dictionary of diagnosis labels
        config: Scenario configuration dictionary

    Returns:
        tuple: (selected_indices, labels)
    """
    cohort_type = config['cohort_selection']
    gender_filter = config.get('gender_filter', None)

    # Get base cohort indices
    if cohort_type == 'takotsubo_and_stemi':
        positive_indices = np.where(data['labels_takotsubo'])[0]
        negative_indices = np.where(diagnosis_labels['stemi'])[0]
        labels = data['labels_takotsubo']

    elif cohort_type == 'takotsubo_and_nstemi':
        positive_indices = np.where(data['labels_takotsubo'])[0]
        negative_indices = np.where(diagnosis_labels['nstemi'])[0]
        labels = data['labels_takotsubo']

    elif cohort_type == 'takotsubo_and_stemi_nstemi':
        positive_indices = np.where(data['labels_takotsubo'])[0]
        negative_indices = np.where(diagnosis_labels['stemi_nstemi'])[0]
        labels = data['labels_takotsubo']

    elif cohort_type == 'takotsubo_and_verified':
        positive_indices = np.where(data['labels_takotsubo'])[0]
        negative_indices = np.where(diagnosis_labels['verified_mi'])[0]
        labels = data['labels_takotsubo']

    elif cohort_type == 'takotsubo_and_all_mi':
        positive_indices = np.where(data['labels_takotsubo'])[0]
        # All rest cases
        negative_indices = np.where(data['labels_takotsubo']==0)[0]
        labels = data['labels_takotsubo']

    elif cohort_type == 'apical_ts_and_ant_stemi':
        positive_indices = np.where(diagnosis_labels['apical_ts'])[0]
        negative_indices = np.where(diagnosis_labels['ant_stemi'])[0]
        labels = diagnosis_labels['apical_ts']

    elif cohort_type == 'stemi_and_nstemi':
        positive_indices = np.where(diagnosis_labels['stemi'])[0]
        negative_indices = np.where(diagnosis_labels['nstemi'])[0]
        labels = diagnosis_labels['stemi']

    else:
        raise ValueError(f"Unknown cohort selection: {cohort_type}")

    # Combine indices
    selected_indices = np.sort(np.concatenate([positive_indices, negative_indices]))

    # Apply gender filter if specified
    if gender_filter == 'male':
        # Keep only males (assuming 0 = male in labels_sex)
        male_mask = data['labels_sex'][selected_indices] == 0
        selected_indices = selected_indices[male_mask]

    elif gender_filter == 'female':
        # Keep only females (assuming 1 = female in labels_sex)
        female_mask = data['labels_sex'][selected_indices] == 1
        selected_indices = selected_indices[female_mask]

    # Remove problematic index if present
    selected_indices = selected_indices[selected_indices != 12279]

    return selected_indices, labels[selected_indices]


# ============================================================================
# Cross-Validation Setup
# ============================================================================

def load_fold_splits(homedir):
    """
    Load pre-defined cross-validation fold splits.

    Args:
        homedir: Home directory path

    Returns:
        tuple: (labels_per_fold, filepaths_per_fold)
    """
    scripts_dir = os.path.join(homedir, "takotsubo_data/data/takotsubo/STEMI_scripts")

    with open(os.path.join(scripts_dir, "labels_per_fold_takotsubo.pickle"), "rb") as f:
        labels_per_fold = pickle.load(f)

    with open(os.path.join(scripts_dir, "filepaths_per_fold_takotsubo.pickle"), "rb") as f:
        filepaths_per_fold = pickle.load(f)


    return labels_per_fold, filepaths_per_fold


def create_fold_indices(filepaths, filepaths_per_fold):
    """
    Create indices for cross-validation folds.

    Args:
        filepaths: Array of all filepaths
        filepaths_per_fold: List of filepath arrays per fold

    Returns:
        list: List of arrays with fold indices
    """
    data_fold_indices = []

    for fold in filepaths_per_fold:
        # Find indices of fold filepaths in main array
        fold_indices = [np.where(f == filepaths)[0] for f in fold]
        num_found = np.array([len(idx) for idx in fold_indices])

        # Keep only found indices
        valid_indices = np.array([fold_indices[j] for j in np.where(num_found > 0)[0]])
        data_fold_indices.append(valid_indices)

    return data_fold_indices


# ============================================================================
# Logistic Regression Model
# ============================================================================

def train_logistic_regression(X_train, y_train, class_weight_factor=1.0):
    """
    Train logistic regression model with optional class weighting.

    Args:
        X_train: Training features (with intercept column)
        y_train: Training labels
        class_weight_factor: Factor to adjust positive class weight

    Returns:
        tuple: (model, results) - fitted GLM model and results
    """
    # Calculate class weights
    num_negative = (y_train == 0).sum()
    num_positive = (y_train == 1).sum()
    pos_weight = (num_negative / num_positive) * class_weight_factor

    # Fit logistic regression using GLM
    model = sm.GLM(y_train, X_train, family=sm.families.Binomial())
    results = model.fit()

    return model, results


def evaluate_model(model, results, X_test, y_test):
    """
    Evaluate logistic regression model on test set.

    Args:
        model: Fitted GLM model
        results: Model fit results
        X_test: Test features
        y_test: Test labels

    Returns:
        dict: Dictionary with evaluation metrics
    """
    # Make predictions
    y_pred_prob = model.predict(results.params, X_test)
    y_pred_class = (y_pred_prob > 0.5).astype(np.int32)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_class)

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(recall, precision)

    metrics = {
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'pr_auc': pr_auc,
        'predictions': y_pred_prob,
        'predictions_class': y_pred_class
    }

    return metrics


# ============================================================================
# Cross-Validation
# ============================================================================

def perform_cross_validation(features, labels, fold_indices, standardize=True):
    """
    Perform k-fold cross-validation for logistic regression.

    Args:
        features: Feature matrix
        labels: Label array
        fold_indices: List of fold indices
        standardize: Whether to standardize features

    Returns:
        dict: Dictionary with results from all folds
    """
    results_dict = {
        'models': [],
        'results': [],
        'coefficients': [],
        'metrics_train': [],
        'metrics_test': [],
        'predictions': [],
        'true_labels': []
    }

    num_folds = len(fold_indices)

    for fold_idx in range(num_folds):
        print(f"\n{'=' * 60}")
        print(f"Fold {fold_idx + 1}/{num_folds}")
        print(f"{'=' * 60}")

        # Split data into train/test
        test_indices = fold_indices[fold_idx].squeeze()
        train_indices = np.concatenate([
            idx for i, idx in enumerate(fold_indices) if i != fold_idx
        ]).squeeze()

        X_train = features[train_indices]
        X_test = features[test_indices]
        y_train = labels[train_indices]
        y_test = labels[test_indices]

        # Standardize features
        if standardize:
            mean = np.mean(X_train, axis=0, keepdims=True)
            std = np.std(X_train, axis=0, keepdims=True)
            X_train = (X_train - mean) / std
            X_test = (X_test - mean) / std

        # Add intercept
        X_train = sm.add_constant(X_train)
        X_test = sm.add_constant(X_test)

        # Train model
        model, results = train_logistic_regression(X_train, y_train)

        # Evaluate on train set
        metrics_train = evaluate_model(model, results, X_train, y_train)
        print(f"\nTrain Confusion Matrix:")
        print(metrics_train['confusion_matrix'])
        print(f"Train ROC AUC: {metrics_train['roc_auc']:.3f}")
        print(f"Train PR AUC: {metrics_train['pr_auc']:.3f}")

        # Evaluate on test set
        metrics_test = evaluate_model(model, results, X_test, y_test)
        print(f"\nTest Confusion Matrix:")
        print(metrics_test['confusion_matrix'])
        print(f"Test ROC AUC: {metrics_test['roc_auc']:.3f}")
        print(f"Test PR AUC: {metrics_test['pr_auc']:.3f}")

        # Store results
        results_dict['models'].append(model)
        results_dict['results'].append(results)
        results_dict['coefficients'].append(results.params)
        results_dict['metrics_train'].append(metrics_train)
        results_dict['metrics_test'].append(metrics_test)
        results_dict['predictions'].append(metrics_test['predictions'])
        results_dict['true_labels'].append(y_test)

    return results_dict


# ============================================================================
# Visualization
# ============================================================================

def plot_roc_curves(cv_results, title="ROC Curves"):
    """
    Plot ROC curves for all folds and aggregate.
    """
    plt.figure(figsize=(10, 8))

    # Plot individual folds
    for i, metrics in enumerate(cv_results['metrics_test']):
        plt.plot(
            metrics['fpr'],
            metrics['tpr'],
            lw=2,
            alpha=0.7,
            label=f"Fold {i + 1} (AUC = {metrics['roc_auc']:.3f})"
        )

    # Plot aggregate
    all_labels = np.concatenate(cv_results['true_labels'])
    all_predictions = np.concatenate(cv_results['predictions'])

    fpr_agg, tpr_agg, _ = roc_curve(all_labels, all_predictions)
    roc_auc_agg = auc(fpr_agg, tpr_agg)

    plt.plot(
        fpr_agg,
        tpr_agg,
        color='black',
        lw=3,
        label=f"Cross-validated (AUC = {roc_auc_agg:.3f})"
    )

    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity (False Positive Rate)', fontsize=12)
    plt.ylabel('Sensitivity (True Positive Rate)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_precision_recall_curves(cv_results, title="Precision-Recall Curves"):
    """
    Plot PR curves for all folds and aggregate.
    """
    plt.figure(figsize=(10, 8))

    # Plot individual folds
    for i, metrics in enumerate(cv_results['metrics_test']):
        plt.plot(
            metrics['recall'],
            metrics['precision'],
            lw=2,
            alpha=0.7,
            label=f"Fold {i + 1} (AUC = {metrics['pr_auc']:.3f})"
        )

    # Plot aggregate
    all_labels = np.concatenate(cv_results['true_labels'])
    all_predictions = np.concatenate(cv_results['predictions'])

    precision_agg, recall_agg, _ = precision_recall_curve(all_labels, all_predictions)
    pr_auc_agg = auc(recall_agg, precision_agg)

    plt.plot(
        recall_agg,
        precision_agg,
        color='black',
        lw=3,
        label=f"Cross-validated (AUC = {pr_auc_agg:.3f})"
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision (PPV)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower left", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_coefficient_importance(cv_results, feature_names):
    """
    Plot feature importance based on coefficient magnitudes.
    """
    # Average coefficients across folds (exclude intercept)
    coefficients = np.array([res[1:] for res in cv_results['coefficients']])
    mean_coefs = np.mean(coefficients, axis=0)
    abs_mean_coefs = np.abs(mean_coefs)

    # Sort by importance
    sorted_indices = np.argsort(abs_mean_coefs)

    plt.figure(figsize=(10, max(8, len(feature_names) * 0.3)))
    plt.barh(
        range(len(sorted_indices)),
        abs_mean_coefs[sorted_indices],
        color='steelblue'
    )
    plt.yticks(range(len(sorted_indices)), [feature_names[i] for i in sorted_indices], fontsize=9)
    plt.xlabel('|Coefficient|', fontsize=12)
    plt.title('Feature Importance (Absolute Coefficient Values)', fontsize=14)
    plt.tight_layout()
    plt.show()

    # Heatmap of coefficients across folds
    plt.figure(figsize=(12, max(8, len(feature_names) * 0.25)))
    plt.imshow(coefficients.T, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3)
    plt.colorbar(label='Coefficient Value')
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.yticks(range(len(feature_names)), feature_names, fontsize=8)
    plt.title('Coefficient Values Across Folds', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_feature_locations(features, labels, important_indices, num_leads=8):
    """
    Visualize important feature locations on average ECG.
    """
    features = features['ecg_features']
    features = np.reshape(features, (features.shape[0], -1))
    # Calculate average ECG for positive class
    avg_ecg = np.mean(features[labels == 1], axis=0)

    plt.figure(figsize=(14, 6))
    plt.plot(avg_ecg, 'r', linewidth=2, label='Average Positive Class ECG')

    # Mark important features
    plt.vlines(
        important_indices,
        np.min(avg_ecg) * 1.05,
        np.max(avg_ecg) * 1.05,
        colors='blue',
        alpha=0.3,
        linewidth=2,
        label='Important Features'
    )

    # Mark lead boundaries
    samples_per_lead = len(avg_ecg) // num_leads
    for i in range(1, num_leads):
        plt.axvline(
            i * samples_per_lead,
            color='black',
            linestyle='--',
            alpha=0.3
        )

    plt.xlabel('Feature Index', fontsize=12)
    plt.ylabel('Normalized Amplitude', fontsize=12)
    plt.title('Important Feature Locations on Average ECG', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# Statistical Analysis
# ============================================================================

def generate_odds_ratio_table(cv_results, feature_names):
    """
    Generate odds ratio table with confidence intervals.
    """
    # Average coefficients and p-values across folds
    coefficients = np.array(cv_results['coefficients'])
    mean_coefs = np.mean(coefficients, axis=0)

    # Get p-values from each fold
    p_values = np.array([res.pvalues for res in cv_results['results']])
    mean_p_values = np.mean(p_values, axis=0)
    mean_p_values = np.round(np.maximum(mean_p_values, 0.001), 3)

    # Calculate odds ratios
    odds_ratios = np.exp(mean_coefs)

    # Create DataFrame
    df = pd.DataFrame({
        'Variable': feature_names,
        'Coefficient': np.round(mean_coefs, 4),
        'Odds Ratio': np.round(odds_ratios, 3),
        'P-value': mean_p_values
    })

    return df


def perform_mann_whitney_test(features, labels, feature_idx, feature_name="Feature"):
    """
    Perform Mann-Whitney U test for a specific feature.
    """
    group_positive = features[labels == 1, feature_idx]
    group_negative = features[labels == 0, feature_idx]

    statistic, p_value = mannwhitneyu(group_positive, group_negative)

    print(f"\nMann-Whitney U Test for {feature_name}:")
    print(f"  Mean Positive Group: {np.mean(group_positive):.4f}")
    print(f"  Mean Negative Group: {np.mean(group_negative):.4f}")
    print(f"  U-statistic: {statistic:.2f}")
    print(f"  P-value: {p_value:.4f}")

    return statistic, p_value


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """
    Main execution function for logistic regression analysis.
    """
    start_time = time.time()

    # Get scenario configuration
    if SCENARIO not in SCENARIO_CONFIGS:
        raise ValueError(f"Invalid scenario: {SCENARIO}. Choose from {list(SCENARIO_CONFIGS.keys())}")

    config = SCENARIO_CONFIGS[SCENARIO]

    # Setup
    homedir, cwd = setup_directories()

    print("=" * 70)
    print(f"Scenario: {SCENARIO}")
    print(f"Description: {config['description']}")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    data = load_clinical_data(homedir)
    diagnosis_labels, filepaths = load_diagnosis_labels(homedir, data['filepaths'])

    # Prepare features
    print("Preparing features...")
    leads = [0, 1, 2, 3, 4, 5, 6, 7]

    # Adjust sex inclusion based on gender filter
    include_sex = INCLUDE_SEX and (config.get('gender_filter') is None)

    features = prepare_features(
        data,
        leads=leads,
        include_sex=include_sex,
        include_clusters=INCLUDE_CLUSTERS,
        include_samples=INCLUDE_SAMPLES,
        include_age=INCLUDE_AGE
    )

    print(f"Initial feature matrix shape: {features.shape}")

    # Feature selection
    print(f"Selecting {len(config['important_features'])} important features plus added...")
    num_ecg_features = (MAX_SAMPLE-MIN_SAMPLE)*len(leads)
    features = select_important_features(features, config['important_features'], num_ecg_features=num_ecg_features)

    # Apply interactions
    if len(config['interactions']) > 0 and APPLY_INTERACTIONS:
        print(f"Applying {len(config['interactions'])} feature interactions...")
        features = apply_interactions(
            features,
            config['interactions'],
            config['delete_after_interaction']
        )

    print(f"Final feature matrix shape: {features.shape}")

    # Select cohort
    print(f"\nSelecting cohort: {config['cohort_selection']}")
    if config.get('gender_filter'):
        print(f"Gender filter: {config['gender_filter']}")

    selected_indices, labels = select_cohort(data, diagnosis_labels, config)

    features = features[selected_indices]
    filepaths = filepaths[selected_indices]

    print(f"\nCohort size: {len(selected_indices)}")
    print(f"Positive cases: {np.sum(labels)}")
    print(f"Negative cases: {len(labels) - np.sum(labels)}")
    print(f"Positive rate: {np.mean(labels):.2%}")

    # Setup cross-validation folds
    print("\nSetting up cross-validation...")
    labels_per_fold, filepaths_per_fold = load_fold_splits(homedir)
    fold_indices = create_fold_indices(filepaths, filepaths_per_fold)

    print(f"Number of folds: {len(fold_indices)}")

    # Perform cross-validation
    print("\n" + "=" * 70)
    print("Performing Cross-Validation")
    print("=" * 70)

    cv_results = perform_cross_validation(
        features,
        labels,
        fold_indices,
        standardize=STANDARDIZE
    )

    # Generate results summary
    print("\n" + "=" * 70)
    print("Cross-Validation Results Summary")
    print("=" * 70)

    all_labels = np.concatenate(cv_results['true_labels'])
    all_predictions = np.concatenate(cv_results['predictions'])

    overall_roc_auc = roc_auc_score(all_labels, all_predictions)
    precision, recall, _ = precision_recall_curve(all_labels, all_predictions)
    overall_pr_auc = auc(recall, precision)

    print(f"\nOverall ROC AUC: {overall_roc_auc:.3f}")
    print(f"Overall PR AUC: {overall_pr_auc:.3f}")

    if 'expected_roc_auc' in config:
        print(f"Expected ROC AUC: {config['expected_roc_auc']:.3f}")
        print(f"Expected PR AUC: {config['expected_pr_auc']:.3f}")

    # Generate odds ratio table
    feature_names = config['feature_names']
    odds_table = generate_odds_ratio_table(cv_results, feature_names)

    print("\n" + "=" * 70)
    print("Odds Ratios and P-values")
    print("=" * 70)
    print(odds_table.to_string(index=False))

    # Save results
    output_dir = os.path.join(cwd, "results", SCENARIO)
    os.makedirs(output_dir, exist_ok=True)

    odds_table.to_csv(os.path.join(output_dir, "odds_ratios.csv"), index=False)
    print(f"\nOdds ratio table saved to {output_dir}/odds_ratios.csv")

    # Visualizations
    print("\n" + "=" * 70)
    print("Generating Visualizations")
    print("=" * 70)

    plot_roc_curves(cv_results, title=f"ROC Curves: {config['description']}")
    plot_precision_recall_curves(cv_results, title=f"PR Curves: {config['description']}")
    plot_coefficient_importance(cv_results, feature_names[1:])  # Exclude intercept
    plot_feature_locations(data, labels, config['important_features'])

    # Print timing
    elapsed_time = (time.time() - start_time) / 60
    print(f"\n{'=' * 70}")
    print(f"Total analysis time: {elapsed_time:.2f} minutes")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
