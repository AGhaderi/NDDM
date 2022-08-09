#!/home/a.ghaderi/.conda/envs/envjm/bin/python
import os
import numpy as np
from scipy import stats
from time import time
import matplotlib.pyplot as plt

from numba import njit
import tensorflow as tf

import sys
sys.path.append('../../')
from bayesflow.networks import InvertibleNetwork, InvariantNetwork
from bayesflow.amortizers import SingleModelAmortizer
from bayesflow.trainers import ParameterEstimationTrainer
from bayesflow.diagnostics import *
from bayesflow.models import GenerativeModel


def prior(batch_size):
    """
    Samples from the prior 'batch_size' times.
    ----------

    Arguments:
    batch_size : int -- the number of samples to draw from the prior
    ----------

    Output:
    theta : np.ndarray of shape (batch_size, theta_dim) -- the samples batch of parameters
    """

    # Prior ranges for the simulator
    # mu_drift ~ U(-3.0, 3.0)
    # boundary ~ U(0.5, 4.0)
    # beta ~ U(0.1, 0.9)  # relative start point
    # tau ~ U(0.1, 1.0)
    # sigma ~ ~ U(.5, 4) # is 1/lambda
    # Eta ~ U(0.0, 1.0)
    n_parameters = 6
    p_samples = np.random.uniform(low=(0.01,  0.5, 0.1, 0.1, 0.0, 0.00),
                                  high=(3.0, 2.0, 0.9, 1.0, 2.0,  2.0), size=(batch_size, n_parameters))
    return p_samples.astype(np.float32)

@njit
def diffusion_trial(mu_drift, boundary, beta, tau, sigma, eta, dc=1.0, dt=.005):
    """Simulates a trial from the diffusion model."""

    n_steps = 0.
    evidence = boundary * beta
    
    # trial-to-trial drift rate variability
    drift_trial = mu_drift + eta * np.random.normal()
    
    # Simulate a single DM path
    while (evidence > 0 and evidence < boundary):

        # DDM equation
        evidence += drift_trial*dt + np.sqrt(dt) * dc * np.random.normal()

        # Increment step
        n_steps += 1.0

    rt = n_steps * dt

    
    # CPP slope
    cpp = np.random.normal(drift_trial, sigma)

    
    if evidence >= boundary:
        choicert =  tau + rt
        
    else:
        choicert = -tau - rt
    return choicert, cpp

@njit
def diffusion_condition(params, n_trials):
    """Simulates a diffusion process over an entire condition."""

    mu_drift, boundary, beta, tau, sigma, eta = params
    choicert = np.empty(n_trials)
    z = np.empty(n_trials)
    for i in range(n_trials):
        choicert[i], z[i] = diffusion_trial(mu_drift, boundary, beta, tau, sigma, eta)
    return choicert, z

def batch_simulator(prior_samples, n_obs):
    """
    Simulate multiple diffusion_model_datasets.
    """

    n_sim = prior_samples.shape[0]
    sim_choicert = np.empty((n_sim, n_obs), dtype=np.float32)
    sim_z = np.empty((n_sim, n_obs), dtype=np.float32)

    # Simulate diffusion data
    for i in range(n_sim):
        sim_choicert[i], sim_z[i] = diffusion_condition(prior_samples[i], n_obs)

    # For some reason BayesFlow wants there to be at least two data dimensions
    sim_data = np.stack([sim_choicert, sim_z], axis=-1)
    return sim_data

# Connect the networks through a SingleModelAmortizer instance.
summary_net = InvariantNetwork()
inference_net = InvertibleNetwork({'n_params': 6})
amortizer = SingleModelAmortizer(inference_net, summary_net)

# Connect the prior and simulator through a GenerativeModel class which will take care of forward inference.
generative_model = GenerativeModel(prior, batch_simulator)

trainer = ParameterEstimationTrainer(
    network=amortizer,
    generative_model=generative_model,
    checkpoint_path="../checkpoint/CPP_single_trial_sigma_eta"
)

# Variable n_trials
def prior_N(n_min=60, n_max=300):
    """
    A prior or the number of observation (will be called internally at each backprop step).
    """

    return np.random.randint(n_min, n_max + 1)


# Experience-replay training
losses = trainer.train_experience_replay(epochs=500,
                                         batch_size=32,
                                         iterations_per_epoch=1000,
                                         capacity=100,
                                         n_obs=prior_N)

# Validate (quick and dirty)
n_param_sets = 1000
n_samples = 1000
n_trials = 1000

true_params = prior(n_param_sets)
x = batch_simulator(true_params, n_trials).astype(np.float32)
param_samples = amortizer.sample(x, n_samples=n_samples)
param_means = param_samples.mean(axis=0)
true_vs_estimated(true_params, param_means, ['mu_drift', 'boundary', 'beta', 'tau', 'sigma', 'eta'], filename="../true_vs_estimate/CPP_single_trial_sigma_eta")
