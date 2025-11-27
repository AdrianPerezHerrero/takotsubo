# Takotsubo Syndrome Classification

This repository contains the implementation scripts for the paper **"Generative Machine learning-electrocardiography to differenciate takotsubo syndrome from myocardial infarction"**, which presents a novel approach to Takotsubo syndrome classification using Bayesian nonparametric clustering methods.

## Overview

Takotsubo syndrome, also known as stress-induced cardiomyopathy, presents diagnostic challenges due to its similarity to acute myocardial infarction. This project applies advanced machine learning techniques to improve the classification and understanding of Takotsubo syndrome through electrocardiogram (ECG) analysis. It should be noted that the pipeline could be extended to perform an exploratory study on any ECG or biosignal classification problem.

### Key Features

- **Multi-output Gaussian Process modeling** of 8-lead ECG signals simultaneously
- **Bayesian nonparametric clustering** with automatic cluster number determination
- **Batch processing** for efficient analysis of multiple heartbeats
- **Multiple clinical scenarios** including TTS vs STEMI, NSTEMI, and suspected MI
- **Comprehensive statistical analysis** with cross-validation and odds ratio reporting

## Repository Structure

### Main Scripts

#### 1. `clustering_ecg_data.py`
**Multi-Lead ECG Heartbeat Clustering using HDP-GPC**

Performs unsupervised clustering of multi-lead ECG heartbeat segments to identify distinct cardiac patterns.

#### 2. `logistic_regression_analysis.py`
**Multi-Scenario Logistic Regression for TTS Classification**

Performs comprehensive logistic regression analysis to discriminate between Takotsubo syndrome and different cardiac conditions.

## Methodology

### Clustering Method: HDP-GPC

The clustering analysis is performed using the **Hierarchical Dirichlet Process Gaussian Process Clustering (HDP-GPC)** method, a Bayesian nonparametric approach that:

- Automatically determines the optimal number of clusters
- Models the evolution of time series patterns over time
- Captures morphological variations in ECG signals

For detailed information about the HDP-GPC methodology, please refer to:
- **Paper**: [Bayesian Nonparametric Dynamical Clustering of Time Series](https://arxiv.org/abs/2501.01234) (Pérez-Herrero et al.)
- **Implementation**: [HDP-GPC GitHub Repository](https://github.com/AdrianPerezHerrero/HDP-GPC)

## Data

The data that support the findings of this study are available at the Department of Clinical Science and Education Söderjukhuset, Karolinska Institutet, Stockholm, Sweden, but restrictions apply to the availability of these data, which were used under license for the current study, and so are not publicly available. Data are, however, available from the authors upon reasobale request and with permission of the Department of Clinical Science and Education Söderjukhuset, Karolinska Institutet, Stockholm, Sweden.

