# Augmentation

[![Build Status](https://github.com/achillet/Augmentation.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/achillet/Augmentation.jl/actions/workflows/CI.yml?query=branch%3Amain)


The Augmentation module defines some function to create a data augmentation.
The aim is to add some channel model to the signals. This module can be used with synthetic or real data. 

The parameter N is the augmentation factor. 
    If N = 1, the function returns a matrix that has the same size as the input matrix and only applies the channel model to the data.
    If N = 2, the function returns a matrix that is double the size of the input matrix and applies two different channels.


The **multipath** mode is the one that we use in our experiment. 

The channel model implemented in our database generator for "multipath" mode is a wireless flat-fading
transmission with random delay spread. The maximum of the delay spread is set at 36,
which corresponds to the CP of the OFDM considered here. The signal obtained after the
channel is modeled as:
y(t) = h(t) ∗ xPA(t) + n(t)
where ∗ is the convolution operator, h represents the impluse response of the propagation
channel and, n(t) is a AWGN. The power of the tap follow a rayleigh model centered
around 1 (e.g Rice model), to ensure that enough power is always received at the reception
stage.

if t= τ : h(t) = 1 + αγ, else : h(t) = 0

where α corresponds to the rayleigh model random variable and γ respresent a ponderating
factor with the value of 0.3. 
The doppler effect is not considered and the channel power does not change in function
of time in other word we considered that the devices are fixed during the transmission.
In order to model changes in environmental propagation, we consider that different flat
fading channels can been countered. To achieve this, we generate different channels applied
to a few consecutive sequences of 256 IQ samples, with each channel having a different
random power and delay spread.


ETU and EVA are two different models specified with different delay profiles: the Extended Vehicular A (EVA) model and the Extended Typical Urban (ETU) model [1]. The EVA model
represents a medium delay environment, while the ETU model represents a low delay environment. For both models, we define 8 taps and the maximum delay spread is set to 36, which corresponds to the CP of the OFDM considered here. Both EVA and ETU are not yet tested with RiFyFi.