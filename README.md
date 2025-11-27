# Takotsubo Syndrome Classification

This repository contains the implementation scripts for the paper **"Generative Machine learning-electrocardiography to differenciate takotsubo syndrome from myocardial infarction"**, which presents a novel approach to Takotsubo syndrome classification using Bayesian nonparametric clustering methods.

## Overview

Takotsubo syndrome, also known as stress-induced cardiomyopathy, presents diagnostic challenges due to its similarity to acute myocardial infarction. This project applies advanced machine learning techniques to improve the classification and understanding of Takotsubo syndrome through electrocardiogram (ECG) analysis.

## Repository Contents

The repository includes three main Python scripts:

### 1. `GPI_test_takotsubo_cluster.py`
ECG data segmentation and clustering
- Processes ECG signals to extract heartbeat segments
- Performs clustering using the HDP-GPC (Hierarchical Dirichlet Process Gaussian Process Clustering) method
- Identifies distinct cardiac patterns in the data

### 2. `GPI_test_takotsubo_from_ssh.py`
Model feature extraction
- Extracts important features and parameters from the trained clustering model
- Processes the learned representations for subsequent analysis
- Prepares data for statistical modeling

### 3. `GPI_test_STEMI_logit_statsmodels_comparison_newdata_pure_mean.py`
Logistic regression analysis
- Implements logistic regression models for different clinical scenarios
- Performs statistical analysis and model evaluation

## Methodology

### Clustering Method: HDP-GPC

The clustering analysis is performed using the **Hierarchical Dirichlet Process Gaussian Process Clustering (HDP-GPC)** method, a Bayesian nonparametric approach that:

- Automatically determines the optimal number of clusters
- Models the evolution of time series patterns over time
- Captures morphological variations in ECG signals

For detailed information about the HDP-GPC methodology, please refer to:
- **Paper**: [Bayesian Nonparametric Dynamical Clustering of Time Series](https://arxiv.org/abs/2501.01234) (Pérez-Herrero et al.)
- **Implementation**: [HDP-GPC GitHub Repository](https://github.com/AdrianPerezHerrero/HDP-GPC)

##Data

The data that support the findings of this study are available at the Department of Clinical Science and Education Söderjukhuset, Karolinska Institutet, Stockholm, Sweden, but restrictions apply to the availability of these data, which were used under license for the current study, and so are not publicly available. Data are, however, available from the authors upon reasobale request and with permission of the Department of Clinical Science and Education Söderjukhuset, Karolinska Institutet, Stockholm, Sweden.

