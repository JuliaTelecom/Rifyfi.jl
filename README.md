# RiFyFi.jl 

[![Build Status](https://github.com/achilletIrisa/RiFyFi.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/achilletIrisa/RiFyFi.jl/actions/workflows/CI.yml?query=branch%3Amain)

[introduction.pdf](https://github.com/JuliaTelecom/Rifyfi.jl/files/13974266/introduction.pdf)

RiFyFi is a framework for Radio Frequency Fingerprint Identification. RFF is a unique signature created in the emitter transmission chain by the hardware impairments. These impairments may be used as a secure identifier as they cannot be easily replicated for spoofing purposes. In recent years, the RFF identification relies mainly on Deep Learning (DL), and large databases are consequently needed to improve identification in different environmental conditions. RiFyFi is introduced to propose a framework combine with a virtual database to explore the RFF identification. Different transmission scenarios are modeled such as the data type (being a preamble or a payload) and the data size. 

RiFyFi is composed of different subpackage 
- RiFyFi_VDG : The Virtual Database Generator
- RiFyFi_IdF : Package for training and testing network
- Augmentation : Package used for data augmentation
- Results : Package to create some confusion matrix or F1 score evolution in function of time.

## Protocol to use RiFyFi with Julia 

Insatall Julia (here develop with 1.8.5)
Create a folders
Download or clone the project with: git clone https://github.com/JuliaTelecom/Rifyfi.jl
Going in Rifyfi.jl folders
Open a Julia terminal
Activate the project ] activate .
Instantiate the project with ] instantiate 

Then you can you the script script_example.jl, in the Julia Terminal :
include("src/script_exemple.jl")


Then you can use RiFyFi in two manners: 
-   Creating random parameter values for impairments 
-   Define the value of the impairments parameters with scenario file 



