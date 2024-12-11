#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==============================================================================
# Copyright 2020-* Luca Bortolussi. All Rights Reserved.
# Copyright 2020-* Laura Nenzi.     All Rights Reserved.
# Copyright 2020-* AI-CPS Group @ University of Trieste. All Rights Reserved.
# ==============================================================================


import torch
import copy
import matplotlib.pyplot as plt

# TODO: reason about density of sampling etc (both here and in formulae generation)

class Measure:
    def sample(self, samples=100000, varn=2, points=100):
        #must be overridden
        pass


class BaseMeasure(Measure):
    def __init__(
        self, mu0=0.0, sigma0=1.0, mu1=0.0, sigma1=1.0, q=0.02, q0=0.5, device="cpu", density=1, period=False):
        """

        Parameters
        ----------
        mu0 : mean of normal distribution of initial state, optional
            The default is 0.0.
        sigma0 : standard deviation of normal distribution of initial state, optional
            The default is 1.0.
        mu1 : DOUBLE, optional
            mean of normal distribution of total variation. The default is 0.0.
        sigma1 : standard deviation of normal distribution of total variation, optional
            The default is 1.0.
        q : DOUBLE, optional
            probability of change of sign in derivative. The default is 0.02.
        q0 : DOUBLE, optional
            probability of initial sign of  derivative. The default is 0.5.
        device : 'cpu' or 'cuda', optional
            device on which to run the algorithm. The default is 'cpu'..

        Returns
        -------
        None.

        """
        self.mu0 = mu0
        self.sigma0 = sigma0
        self.mu1 = mu1
        self.sigma1 = sigma1
        self.q = q
        self.q0 = q0
        self.device = device
        self.density = density
        self.period = period


    def sample(self, samples=100000, varn=3, points=100):
        """
        Samples a set of trajectories from the basic measure space, with parameters
        passed to the sampler

        Parameters
        ----------
        points : INT, optional
            number of points per trajectory, including initial one. The default is 1000.
        samples : INT, optional
            number of trajectories. The default is 100000.
        varn : INT, optional
            number of variables per trajectory. The default is 2.


        Returns
        -------
        signal : samples x varn x points double pytorch tensor
            The sampled signals.

        """
        if self.period:
            period_length=self.period
            self.period = False
            trajectories=self.sample(samples, varn, points=period_length)
            #noise= 0.1 * torch.randn(samples, varn, points-period_length, device=self.device)
            repeated_trajectories = trajectories.repeat(1, 1, (points // period_length) + 1)[:, :, :points]
            extended_trajectories = repeated_trajectories.clone()
            #extended_trajectories[:, :, period_length:] += noise
            return extended_trajectories

        
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("GPU card or CUDA library not available!")

        # generate unif RN
        signal = torch.rand(samples, varn, points, device=self.device)
        # first point is special - set to zero for the moment, and set one point to 1
        signal[:, :, 0] = 0.0
        signal[:, :, -1] = 1.0
        # sorting each trajectory
        signal, _ = torch.sort(signal, 2)
        # computing increments and storing them in points 1 to end
        signal[:, :, 1:] = signal[:, :, 1:] - signal[:, :, :-1]
        # generate initial state, according to a normal distribution
        signal[:, :, 0] = self.mu0 + self.sigma0 * torch.randn(signal[:, :, 0].size())

        # sampling change signs from bernoulli in -1, 1
        derivs = (1 - self.q) * torch.ones(samples, varn, points, device=self.device)
        derivs = 2 * torch.bernoulli(derivs) - 1
        # sampling initial derivative
        derivs[:, :, 0] = self.q0
        derivs[:, :, 0] = 2 * torch.bernoulli(derivs[:, :, 0]) - 1
        # taking the cumulative product along axis 2
        derivs = torch.cumprod(derivs, 2)

        # sampling total variation
        totvar = torch.pow(
            self.mu1 + self.sigma1 * torch.randn(samples, varn, 1, device=self.device),2
         )
        # multiplying total variation and derivatives an making initial point non-invasive
        derivs = derivs * totvar
        derivs[:, :, 0] = 1.0

        # computing trajectories by multiplying and then doing a cumulative sum
        signal = signal * derivs
        signal = torch.cumsum(signal, 2)
        # add point density
        if self.density > 1:
            dense_signal = torch.zeros(samples, varn, (points - 1) * self.density + 1)
            dense_signal[:, :, ::self.density] = signal
            diff = dense_signal[:, :, self.density::self.density] - dense_signal[:, :, 0:-self.density:self.density]
            for i in range(self.density - 1):
                dense_signal[:, :, i + 1::self.density] = dense_signal[:, :, 0:-self.density:self.density] + \
                                                          (diff / self.density) * (i + 1)
            signal = copy.deepcopy(dense_signal)
        return signal
    
    def normalize_trajectories(self,trajectories):
        flattened_trajectories = trajectories.view(-1)
        q1 = torch.quantile(flattened_trajectories, 0.15)
        q3 = torch.quantile(flattened_trajectories, 0.85)
        iqr = q3 - q1       
        scale = 3 / iqr
        normalized_trajectories = -1.5 + (trajectories - q1) * scale
        return normalized_trajectories
    
    def plot_trajectory(self, traj, nvar):
        '''Plot a list of trajectory'''
        for j in range(0, nvar):
            plt.plot(range(0,100),traj.tolist()[j])
        plt.show()
    