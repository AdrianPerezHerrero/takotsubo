#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Lead ECG Heartbeat Clustering using HDP-GPC

This script performs unsupervised clustering of multi-lead ECG heartbeat segments
using the Hierarchical Dirichlet Process Gaussian Process Clustering (HDP-GPC) method.
The clustering identifies distinct cardiac patterns across multiple patients and ECG leads.

Key features:
- Multi-output Gaussian Processes (8 ECG leads simultaneously)
- Batch processing for efficiency
- Shared GP model across leads
- Incremental model saving

Author: Adrian Perez-Herrero
Project: Takotsubo Syndrome Classification
Reference: Takotsubo_classification_NPJ paper
HDP-GPC: https://github.com/AdrianPerezHerrero/HDP-GPC
"""

import os
import sys
import time
import pickle as plk
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import trange

# Import HDP-GPC clustering module
import hdpgpc.GPI_HDP as hdpgp
from hdpgpc.util_plots import print_results, plot_models_plotly

# ============================================================================
# Configuration
# ============================================================================

# Set PyTorch data type
dtype = torch.float64
torch.set_default_dtype(dtype)

# Multi-output parameters
NUM_ECG_LEADS = 8  # Number of ECG leads to process simultaneously

# HDP-GPC hyperparameters
INI_LENGTHSCALE = 5.0
BOUND_LENGTHSCALE = (0.5, 50.0)
BOUND_SIGMA = (200.0, 300.0)
INITIAL_SIGMA = 0.1
OUTPUT_SCALE = 200.0
NOISE_WARP = 100.0
BOUND_NOISE_WARP = (100.0, 200.0)
NUM_LDS_COMPONENTS = 2  # Number of Linear Dynamical Systems components
GAMMA_VALUE = 0.5
BOUND_GAMMA = (0.1 ** 2, 20.0 ** 2)
MAX_MODELS = 3  # Maximum number of clusters

# Processing options
USE_WARPING = False
SAVE_MODELS = True
SHARE_GP_ACROSS_LEADS = True  # Share GP model across all leads
REESTIMATE_INITIAL_PARAMS = False
N_EXPLORE_STEPS = 1
FREE_DEG_MNIV = 8  # Free degrees for Matrix-Normal Inverse-Wishart


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
        os.chdir(os.path.join("C:", os.sep, "Users", "Adrian", "Projects",
                              "Thesis", "Takotsubo", "takotsubo_data", "HDP-SLDS-GP"))
        homedir = os.path.join("C:", os.sep, "Users", "Adrian", "Projects",
                               "Thesis", "Takotsubo", "takotsubo_data")
    else:
        # Unix-like environment
        try:
            os.chdir(os.path.join(homedir, "Documents", "takotsubo_data", "HDP-SLDS-GP"))
            homedir = os.path.join(homedir, "Documents", "takotsubo_data")
        except FileNotFoundError:
            aux_dir = input("Please specify directory:\n")
            os.chdir(aux_dir)

    return homedir, os.getcwd()


# ============================================================================
# Data Loading
# ============================================================================

def load_preprocessed_data(homedir):
    """
    Load preprocessed ECG data with extracted heartbeat windows.

    Args:
        homedir: Home directory path

    Returns:
        tuple: (train_data, times) - List of patient data and time vectors
    """
    data_dir = os.path.join(homedir, "data", "takotsubo", "full_dataset_filtered")

    print("Loading preprocessed windowed data...")

    # Load windowed heartbeat data (already segmented and extracted)
    with open(os.path.join(data_dir, "window_data_xqrs.pickle"), 'rb') as f:
        train_data = plk.load(f)

    # Load corresponding time basis vectors
    with open(os.path.join(data_dir, "xbas_window_data_xqrs.pickle"), 'rb') as f:
        times = plk.load(f)

    print(f"Loaded data for {len(train_data)} patients")

    return train_data, times


def load_base_kernel(homedir):
    """
    Load pre-trained kernel from first patient for transfer learning.

    Args:
        homedir: Home directory path

    Returns:
        list: List containing base kernel, or None if not found
    """
    model_path = os.path.join(homedir, "data", "takotsubo", "full_dataset_filtered",
                              "saved_models", "sw_gp_0.plk")

    try:
        with open(model_path, 'rb') as f:
            sw_gp = plk.load(f)

        # Get kernel from largest cluster in first lead
        cluster_sizes = [len(gp.indexes) for gp in sw_gp.gpmodels[0]]
        largest_cluster_idx = np.argmax(cluster_sizes)
        gp_chosen = sw_gp.gpmodels[0][largest_cluster_idx]

        # Extract and modify kernel
        base_kernel = gp_chosen.gp.kernel
        base_kernel.k1.k2.length_scale = 10.0  # Adjust length scale

        print("Loaded base kernel for transfer learning")
        return [base_kernel]

    except FileNotFoundError:
        print("No base kernel found, will train from scratch")
        return None


# ============================================================================
# HDP-GPC Multi-Output Clustering
# ============================================================================

def initialize_multioutput_hdp_gpc(x_basis, x_basis_warp, base_kernel=None,
                                   n_outputs=NUM_ECG_LEADS, fitted=False):
    """
    Initialize multi-output HDP-GPC clustering model for simultaneous
    processing of multiple ECG leads.

    Args:
        x_basis: Basis points for GP (time vector)
        x_basis_warp: Basis points for warping function
        base_kernel: Optional pre-trained kernel for transfer learning
        n_outputs: Number of output channels (ECG leads)
        fitted: Whether the base kernel is already fitted

    Returns:
        GPI_HDP object: Initialized multi-output clustering model
    """
    # Set up hyperparameters
    sigma = [INITIAL_SIGMA] * NUM_LDS_COMPONENTS
    gamma = [GAMMA_VALUE] * NUM_LDS_COMPONENTS

    # Initialize multi-output model
    model = hdpgp.GPI_HDP(
        x_basis,
        kernels=base_kernel,
        x_basis_warp=x_basis_warp,
        n_outputs=n_outputs,
        model_type='dynamic',
        ini_lengthscale=INI_LENGTHSCALE,
        bound_lengthscale=BOUND_LENGTHSCALE,
        ini_gamma=gamma,
        ini_sigma=sigma,
        ini_outputscale=OUTPUT_SCALE,
        noise_warp=NOISE_WARP,
        bound_sigma=BOUND_SIGMA,
        bound_gamma=BOUND_GAMMA,
        bound_noise_warp=BOUND_NOISE_WARP,
        warp_updating=False,
        method_compute_warp='greedy',
        verbose=False,
        hmm_switch=True,
        max_models=MAX_MODELS,
        mode_warp='rough',
        annealing=False,
        bayesian_params=True,
        inducing_points=False,
        reestimate_initial_params=REESTIMATE_INITIAL_PARAMS,
        n_explore_steps=N_EXPLORE_STEPS,
        free_deg_MNIV=FREE_DEG_MNIV,
        share_gp=SHARE_GP_ACROSS_LEADS,
    )

    if not base_kernel is None:
        for ld in range(n_outputs):
            model.gpmodels[ld][0].gp.kernel = base_kernel
            model.gpmodels[ld][0].fitted = True

    return model


def perform_batch_clustering(data_segments, time_vectors, base_kernel=None):
    """
    Perform HDP-GPC clustering on multi-lead heartbeat segments using batch processing.

    Args:
        data_segments: Array of shape (n_samples, n_timepoints, n_leads)
        time_vectors: List of time vectors for each segment
        base_kernel: Optional kernel for transfer learning

    Returns:
        GPI_HDP object: Fitted clustering model
    """
    # Check if we have enough data
    if len(data_segments) < 2:
        raise ValueError("Need at least 2 samples for batch clustering")

    # Initialize model with first segment's time vector
    x_basis = np.atleast_2d(time_vectors[0]).astype(np.float64).T
    x_basis_warp = np.atleast_2d(time_vectors[0]).astype(np.float64).T

    # Determine if kernel is fitted
    fitted = base_kernel is not None

    # Initialize model
    model = initialize_multioutput_hdp_gpc(
        x_basis,
        x_basis_warp,
        base_kernel=base_kernel,
        n_outputs=data_segments[0].shape[1],
        fitted=fitted
    )

    # Prepare batch data (exclude first sample used for initialization)
    x_batch = [x_basis] * (len(data_segments) - 1)
    y_batch = data_segments[1:]  # Shape: (n_samples-1, n_timepoints, n_leads)

    # Perform batch clustering
    print(f"Processing batch of {len(y_batch)} segments...")
    model.include_batch(x_batch, y_batch)

    return model


# ============================================================================
# Feature Extraction and Saving
# ============================================================================

def extract_patient_features(model, data_segments):
    """
    Extract learned features from the trained HDP-GPC model.

    Args:
        model: Trained GPI_HDP clustering model
        data_segments: Original data segments

    Returns:
        dict: Dictionary containing extracted features
    """
    # Extract features from first lead's largest cluster
    cluster_sizes = [len(gp.indexes) for gp in model.gpmodels[0]]
    largest_cluster_idx = np.argmax(cluster_sizes)

    # Get mean representations across all leads
    mean = np.array([
        gps[largest_cluster_idx].f_star[-1]
        for gps in model.gpmodels
    ]).T  # Shape: (n_timepoints, n_leads)

    # Get pure average of clustered samples
    clustered_indices = model.gpmodels[0][largest_cluster_idx].indexes
    pure_mean = np.mean(
        data_segments[1:][np.array(clustered_indices)],
        axis=0
    )[np.newaxis, :, :]  # Shape: (1, n_timepoints, n_leads)

    # Extract cluster statistics
    num_clusters = np.array([len(model.gpmodels[0])])
    num_samples = np.array([len(model.gpmodels[0][largest_cluster_idx].indexes)])

    # Extract diagonal of observation noise covariance for each lead
    diag_sigma = np.array([
        np.diag(gps[largest_cluster_idx].Sigma[-1])[:, np.newaxis]
        for gps in model.gpmodels
    ]).T  # Shape: (n_timepoints, n_leads)

    features = {
        'mean': mean,
        'pure_mean': pure_mean,
        'num_clusters': num_clusters,
        'num_samples': num_samples,
        'diag_sigma': diag_sigma
    }

    return features


def save_patient_features(features, patient_idx, homedir, append=True):
    """
    Save or append patient features to disk.

    Args:
        features: Dictionary of extracted features
        patient_idx: Patient index
        homedir: Home directory path
        append: Whether to append to existing files (True) or create new (False)
    """
    output_dir = os.path.join(homedir, "data", "takotsubo", "full_dataset_filtered",
                              "saved_parameters")
    os.makedirs(output_dir, exist_ok=True)

    feature_names = ['mean', 'pure_mean', 'num_samp', 'num_clust', 'diag_sig']
    feature_keys = ['mean', 'pure_mean', 'num_samples', 'num_clusters', 'diag_sigma']

    for fname, fkey in zip(feature_names, feature_keys):
        filepath = os.path.join(output_dir, f"{fname}_7.npy")

        if patient_idx == 0 or not append:
            # Create new file
            np.save(filepath, features[fkey])
        else:
            # Append to existing file
            existing_data = np.load(filepath)

            # Handle different concatenation axes
            if fname in ['num_samp', 'num_clust']:
                combined_data = np.hstack([existing_data, features[fkey]])
            else:
                combined_data = np.vstack([existing_data, features[fkey]])

            np.save(filepath, combined_data)


def handle_empty_patient(patient_idx, data_shape, homedir):
    """
    Handle patient with insufficient data by saving zero features.

    Args:
        patient_idx: Patient index
        data_shape: Shape tuple (n_samples, n_timepoints, n_leads)
        homedir: Home directory path
    """
    # Create zero features
    features = {
        'mean': np.zeros((1, data_shape[1], data_shape[2])),
        'pure_mean': np.zeros((1, data_shape[1], data_shape[2])),
        'num_clusters': np.array([1]),
        'num_samples': np.array([0]),
        'diag_sigma': np.zeros((1, data_shape[1], data_shape[2]))
    }

    # Save features
    save_patient_features(features, patient_idx, homedir, append=(patient_idx > 0))


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """
    Main execution function for multi-lead ECG clustering.
    """
    start_time = time.time()

    # Setup directories
    homedir, cwd = setup_directories()

    print("=" * 70)
    print("Multi-Lead ECG Heartbeat Clustering using HDP-GPC")
    print("=" * 70)

    # Load preprocessed data
    train_data, times = load_preprocessed_data(homedir)

    # Load base kernel for transfer learning
    base_kernel = load_base_kernel(homedir)

    # Create output directory for models
    output_dir = os.path.join(homedir, "data", "takotsubo", "full_dataset_filtered",
                              "saved_models")
    os.makedirs(output_dir, exist_ok=True)

    # Process each patient
    num_patients = len(train_data)

    for patient_idx in range(num_patients):
        print(f"\n{'>' * 30}")
        print(f"Patient: {patient_idx + 1}/{num_patients}")
        print(f"{'<' * 30}")

        patient_data = train_data[patient_idx]
        patient_times = times[patient_idx]

        # Check if patient has sufficient data
        if len(patient_data) < 2:
            print(f"Insufficient data for patient {patient_idx} (only {len(patient_data)} samples)")
            handle_empty_patient(patient_idx, patient_data.shape, homedir)
            continue

        try:
            # Perform clustering
            model = perform_batch_clustering(
                patient_data,
                patient_times,
                base_kernel=base_kernel
            )

            # Check for multiple clusters (warning sign)
            num_clusters = len(model.gpmodels[0])
            if num_clusters > 1:
                print(f"WARNING: {num_clusters} clusters detected (expected 1)")

            # Extract features
            features = extract_patient_features(model, patient_data)

            # Save features
            save_patient_features(features, patient_idx, homedir, append=(patient_idx > 0))

            # Save model (only first patient as reference)
            if SAVE_MODELS and patient_idx == 0:
                model_path = os.path.join(output_dir, f"sw_gp_{patient_idx}.plk")
                model.save_swgp(model_path)
                print(f"Model saved to {model_path}")

            # Update base kernel if not yet initialized
            if base_kernel is None:
                base_kernel = [model.gpmodels[0][0].gp.kernel]
                print("Base kernel initialized from first patient")

        except Exception as e:
            print(f"ERROR processing patient {patient_idx+1}: {e}")
            handle_empty_patient(patient_idx, patient_data.shape, homedir)
            continue

        # Print elapsed time
        elapsed_mins = (time.time() - start_time) / 60.0
        print(f"Elapsed time: {elapsed_mins:.2f} minutes")

    # Print final summary
    total_time = (time.time() - start_time) / 60.0
    print(f"\n{'=' * 70}")
    print(f"Clustering Complete!")
    print(f"Total patients processed: {num_patients}")
    print(f"Total processing time: {total_time:.2f} minutes")
    print(f"Average time per patient: {total_time / num_patients:.2f} minutes")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
