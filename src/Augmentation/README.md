# Augmentation

[![Build Status](https://github.com/achillet/Augmentation.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/achillet/Augmentation.jl/actions/workflows/CI.yml?query=branch%3Amain)


The Module Augmentation define some function to create a data augmentation.
The objective is to add some channel model on the signals. This Modules can be used with synthetic data or with real. 

The parameter N represents the augmentation factor. 
    If N = 1 the function return a matrix wich has the same size as the input matrix and only applying channel model on data.
    If N = 2 the function return a matrix which is the double of the input matrix and apply two different channel.