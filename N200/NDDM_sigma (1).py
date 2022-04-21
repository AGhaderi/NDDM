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
    # drift ~ U(-3.0, 3.0)
    # boundary ~ U(0.5, 4.0)
    # beta ~ U(0.1, 0.9)  # relative start point
    # t_e ~ U(0.05, 0.6)
    # t_m ~ U(0.06, 0.8)
    # sigma_e ~ U(0, 0.3)
    n_parameters = 6
    p_samples = np.random.uniform(low=(-3.0, 0.5, 0.1, 0.05, 0.06, 0.0 ),
                                  high=(3.0, 4.0, 0.9, 0.6,  0.8,  0.3), size=(batch_size, n_parameters))
    return p_samples.astype(np.float32)

@njit
def diffusion_trial(drift, boundary, beta, t_e, t_m, sigma_e, dc=1.0, dt=.005, max_steps=2e3):
    """Simulates a trial from the diffusion model."""

    n_steps = 0.
    evidence = boundary * beta

    # Simulate a single DM path
    while (evidence > 0 and evidence < boundary and n_steps < max_steps):

        # DDM equation
        evidence += drift*dt + np.sqrt(dt) * dc * np.random.normal()

        # Increment step
        n_steps += 1.0

    rt = n_steps * dt

    # N200 latency
    z = np.random.normal(t_e, sigma_e)
    
    if evidence >= boundary:
        choicert =  t_e + rt + t_m
        
    elif evidence <= 0:
        choicert = -t_e - rt - t_m
    else:
        choicert = np.sign(evidence - boundary*.5)*(t_e + rt + t_m)  # Choose closest boundary at max_steps
    return choicert, z

@njit
def diffusion_condition(params, n_trials):
    """Simulates a diffusion process over an entire condition."""

    drift, boundary, beta, t_e, t_m, sigma_e = params
    choicert = np.empty(n_trials)
    z = np.empty(n_trials)
    for i in range(n_trials):
        choicert[i], z[i] = diffusion_trial(drift, boundary, beta, t_e, t_m, sigma_e)
    return choicert, z

def batch_simulator(prior_samples, n_obs, dt=0.005, s=1.0, max_iter=2e4):
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
    checkpoint_path="checkpoint/NDDM_sigma"
)

# Variable n_trials
def prior_N(n_min=60, n_max=300):
    """
    A prior or the number of observation (will be called internally at each backprop step).
    """

    return np.random.randint(n_min, n_max + 1)


# Experience-replay training
#losses = trainer.train_experience_replay(epochs=1000,
#                                         batch_size=32,
#                                         iterations_per_epoch=1000,
#                                         capacity=100,
#                                         n_obs=prior_N)

# Validate (quick and dirty)
n_param_sets = 1000
n_samples = 1000
n_trials = 1000

true_params = prior(n_param_sets)
x = batch_simulator(true_params, n_trials).astype(np.float32)
param_samples = amortizer.sample(x, n_samples=n_samples)
param_means = param_samples.mean(axis=0)
true_vs_estimated(true_params, param_means, ['drift', 'boundary', 'beta', 't_e', 't_m', 'sigma_e'], filename="true_vs_estimate/NDDM_sigma_1")



# Michael Nunez's recovery function
def recovery(possamps, truevals):  # Parameter recovery plots
    """Plots true parameters versus 99% and 95% credible intervals of recovered
    parameters. Also plotted are the median (circles) and mean (stars) of the posterior
    distributions.

    Parameters
    ----------
    possamps : ndarray of posterior chains where the last dimension is the
    number of chains, the second to last dimension is the number of samples in
    each chain, all other dimensions must match the dimensions of truevals

    truevals : ndarray of true parameter values
    """

    # Number of chains
    nchains = possamps.shape[-1]

    # Number of samples per chain
    nsamps = possamps.shape[-2]

    # Number of variables to plot
    nvars = np.prod(possamps.shape[0:-2])

    # Reshape data
    alldata = np.reshape(possamps, (nvars, nchains, nsamps))
    alldata = np.reshape(alldata, (nvars, nchains * nsamps))
    truevals = np.reshape(truevals, (nvars))

    # Plot properties
    LineWidths = np.array([2, 5])
    teal = np.array([0, .7, .7])
    blue = np.array([0, 0, 1])
    orange = np.array([1, .3, 0])
    Colors = [teal, blue]

    for v in range(0, nvars):
        # Compute percentiles
        bounds = stats.scoreatpercentile(alldata[v, :], (.5, 2.5, 97.5, 99.5))
        for b in range(0, 2):
            # Plot credible intervals
            credint = np.ones(100) * truevals[v]
            y = np.linspace(bounds[b], bounds[-1 - b], 100)
            lines = plt.plot(credint, y)
            plt.setp(lines, color=Colors[b], linewidth=LineWidths[b])
            if b == 1:
                # Mark median
                mmedian = plt.plot(truevals[v], np.median(alldata[v, :]), 'o')
                plt.setp(mmedian, markersize=10, color=[0., 0., 0.])
                # Mark mean
                mmean = plt.plot(truevals[v], np.mean(alldata[v, :]), '*')
                plt.setp(mmean, markersize=10, color=teal)
    # Plot line y = x
    tempx = np.linspace(np.min(truevals), np.max(
        truevals), num=100)
    recoverline = plt.plot(tempx, tempx)
    plt.setp(recoverline, linewidth=3, color=orange)


param_names = ['drift', 'boundary', 'beta', 't_e', 't_m', 'sigma_e']
for i in range(len(param_names)):
# Plot the results
    f, axarr = plt.subplots()
    recovery(param_samples[:, :, i].T.reshape(n_param_sets, n_samples, 1), true_params[:, i].squeeze())
    plt.xlabel('True')
    plt.ylabel('Posterior')
    plt.title(param_names[i])
    plt.savefig('true_vs_estimate/NDDM_sigma/'+ param_names[i] +'.png')

    # Compute NRMSE
    param_means = np.mean(param_samples, axis=0)
    rmse = np.sqrt(np.mean( (param_means[:,i] - true_params[:,i])**2 ))
    nrmse = rmse / (param_means[:, i].max() - true_params[:, i].min())

    plt.text(0.1, 0.9, 'NRMSE={:.3f}'.format(nrmse),
                 horizontalalignment='left',
                 verticalalignment='center',
                 transform=axarr.transAxes,
                 size=10)

    # Compute R2
    r2 = r2_score(true_params[:, i], param_means[:, i])
    plt.text(0.1, 0.8, '$R^2$={:.3f}'.format(r2),
                 horizontalalignment='left',
                 verticalalignment='center',
                 transform=axarr.transAxes, 
                 size=10)

