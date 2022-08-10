# Neural drift-diffusion model (NDDM).
The current repository is related to a project entitled "A general integrative neurocognitive modeling framework to jointly describe EEG and decision-making on single trials" in collaboration with the Institute for Cognitive and Brain Sciences from the Shahid Beheshti University and Psychological Methods from University of Amsterdam.

**Authors: Amin Ghaderi-Kangavari, Jamal Amani Rad, & Michael D. Nunez**

### Citation
Ghaderi-Kangavari, A., Rad, J. A., & Nunez, M. D. (2022, August 8). A general integrative neurocognitive modeling framework to jointly describe EEG and decision-making on single trials. https://doi.org/10.31234/osf.io/pqv2c


## Prerequisites

[BayesFlow](https://github.com/stefanradev93/BayesFlow)

[BayesFlow requirements](https://github.com/stefanradev93/BayesFlow/blob/master/requirements.txt)


## Abstract 

Despite advances in techniques for exploring reciprocity in brain-behavior relations, few studies focus on building neurocognitive models that describe both human EEG and behavioral modalities at the single-trial level. Here, we introduce a new integrative joint modeling framework for the simultaneous description of single-trial EEG measures and cognitive modeling parameters of decision making. The new framework can be utilized for the evaluation of research questions as well as the prediction of both data types concurrently. In the introduced joint models, we formalized how single-trial N200 latencies and Centro-parietal positivities (CPP) are predicted by changing single-trial parameters of various drift-diffusion models (DDMs). These models do not have clear closed-form likelihoods and are not easy to fit using Markov chain Monte Carlo (MCMC) methods because nuisance parameters on single trials are shared in both behavior and neural activity. We trained deep neural networks to learn the Bayesian posterior distributions of unobserved neurocognitive parameters based on model simulations. We then used parameter recovery assessment and model misspecification to ascertain how robustly the models’ parameters can be estimated. Moreover, we fit the models to three different real datasets to test their applicability. Our results show that the single-trial integrative models can recover their latent parameters. Finally, we provide some pieces of evidence that single-trial integrative joint models are superior to traditional integrative models. The current single-trial paradigm and the likelihood-free approach for parameter recovery can inspire scientists and modelers to conveniently develop new neuro-cognitive models for other neural measures and to evaluate them appropriately.

