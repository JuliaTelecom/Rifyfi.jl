# Tutorial 

[![Build Status](https://github.com/achilletIrisa/RiFyFi.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/achilletIrisa/RiFyFi.jl/actions/workflows/CI.yml?query=branch%3Amain)

This README describes the different script results and how to modify them. 

Don't hesitate to use this README as a tutorial if you want to quickly understand the different RiFyFi modes. 

## RiFyFi with virtual database generator

Use the script script_RiFyFi.jl, if you have already installed julia and the RiFyFi package and instantiated it, you can run the script


```
julia> include("src/script_RiFyFi.jl")
```


Here the generated dataset consists of 6 transmitters transmitting 10000 signals, each signal consisting of 256 IQ samples. The RFF parameters of each transmitter are defined by the scenario file in Configurations --> No_channel_6_256 --> E3_all_impairments_5_pourcent (Configuration = "scernario").
All impairments are enabled (RFF = "all_impairments").

The variable S describes the type of frame (symbols) S1 for preamble, S2 for Mac address, S3 for payload. 

- Preamble: all transmitters send the same sequence
- Mac address: each transmitter sends its own sequence
- Payload: all sequences are different and are not repeated. 

Variable E described the fingerprint, leaving E at E3 to activate the RFF, otherwise E1 to deactivate the RFF.

Then C describes the channel, in particular the noise, several possibilities are proposed. 
- C1: no noise 
- C2: 30dB gaussian noise
- C2_20dB: 20dB gaussian noise 
- C2_15dB: 15dB gaussian noise 
- C2_10dB: 10dB gaussian noise 
- C2_0dB: 0dB gaussian noise 

Finally, no propagation channel model is added to the signals.

The chosen network is called AlexNet and the training parameters are defined, such as the learning rate and the dropout.

Running the file without any changes gives the result shown below.


In run/Synth/No_channel_6_256_AlexNet/E3_S1_C2_20dB_all_impairments_10000_5_pourcent/GPU 
you can find this type of figure with the F1 score evolution as a function of time.
<div align="center">
  <img src="../docs/F1-scoreV1.png" alt="Makie.jl" width="380">
</div>


In Results/No_channel_6_256_AlexNet
you can find the confusion matrix obtained in the test
<div align="center">
  <img src="../docs/CMV1.png" alt="Makie.jl" width="380">
</div>

Then in Results/augment_6_256_AlexNet
You can find the confusion matrix obtained in the test with the addition of the propagation channel model<div align="center">
  <img src="../docs/CMV1_augment.png" alt="Makie.jl" width="380">
</div>

The confusion matrix shows that the network is not able to correctly classify/identify the transmitters, which means that the channel resilience is poor.

You can now modify the contents of script_RiFyFi.jl to train a network with a different database (RFF scenario, more transmitters, transmission scenario).

To illustrate the possibilities, you can choose to add data augmentation to the training dataset and then evaluate the resilience of the network.

In the script change the following lines:
```
########### Augmentation struct ###########
augmentationType = "augment"
Channel = "etu"
Channel_Test = "etu"
nb_Augment = 100
#seed_channel = 12
#seed_channel_test = 12
#burstSize =64
```
You can train different networks with different nb_Augment values and observe the effect of propagation channel diversity on network resilience. You will get better results if you increase the channel diversity sufficiently.








