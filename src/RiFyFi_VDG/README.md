# RiFyFi_VDG

[![Build Status](https://github.com/achillet/RiFyFi_VDG.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/achillet/RiFyFi_VDG.jl/actions/workflows/CI.yml?query=branch%3Amain)


The Module RiFyFi_VDG defines function to create virtual database. 

Different file description: 

- DatBinaryFiles.jl allow to read or write a binary file. 
- dynamic_CFO.jl This file can be use to change the confguration of the CFO between the training and test conditions. (Not use here)
- pa_memory.jl contains the function used when the Power Amplifier is modelled by a Memory model and not Saleh.
- plotAMAM.jl contains function to plot the AM/AM Figure of the PA model (Saleh or memory)
- RiFyFi_VDG.jl contains all main function used to created the virtual databases
- rx.jl this file exist but is not complete, can be a future work
- setup_impairments.jl contains a function for each type of impairments to initalise it
- tx_LoRa.jl contains function to generate LoRa Frame 
- tx_singleCarrier.jl contains function to generate single carrier frame 
- tx.jl contains function to generate OFDM frame 
- utils_dict.jl diverse function

In RiFyFi_VDG.jl you will found different function to create the dataset based on virtual database. The most imporant one are described bellow.


**setSynthetiquecsv**: this function takes in input a structure wich describe the wanted dataset and create 4 CSV files : traing data, testing data, and the corresponding labels for training and testing data.
This fucntion called the function "create_virtual_Database"


**create_virtual_Database**: this function create or load the configuration and the scenario files and then called generateDatabase two times to obtain the training and test dataset.
The configuration file save information about the number of transmitter, the type of data, the quantity of data ... 
The scenario file save the impairments parameter for each transmitters. 
You can create your own scenario file, and use it to create the database, or you can use the automatic and random case. 
In the automatic and random case the impairments of two different transmitter can possibly have a very close value. In this case it could be difficult for the network to separate the both transmitters.
This function can called Augmentation package to add propagation channel model on data.
Then the dataset is shuffle (not necessary because the dataloader format shuffle the data during the network training.)


**generateDatabase**: this function create a tup with different information, in particular dict_out contains the scenario parameters. bigMat contains the data matrix 


**loadCSV_Synthetic** This function allow to load the CSV files corresponding to the database that the data structure described.
You can create a database with the previous functions, and then use it many times to train different network without the need to generate the data again.


**reloadScenario** This function is called when you want to load a particular scenario file. The function return the scenario impairments value for each transmitter. 