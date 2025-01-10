# RiFyFi_VDG

[![Build Status](https://github.com/achillet/RiFyFi_VDG.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/achillet/RiFyFi_VDG.jl/actions/workflows/CI.yml?query=branch%3Amain)

The RiFyFi_VDG module defines a function to create a virtual database. 

Different file description: 

- DatBinaryFiles.jl allows to read or write a binary file. 
- dynamic_CFO.jl This file can be used to change the configuration of the CFO between training and test conditions. (Not used here)
- pa_memory.jl contains the function used when the power amplifier is modelled by a memory model and not by Saleh.
- plotAMAM.jl contains function to plot the AM/AM figure of the PA model (Saleh or Memory)
- RiFyFi_VDG.jl contains all the main functions used to create the virtual databases.
- rx.jl this file exists but is not complete, may be a future work
- setup_impairments.jl contains a function for each type of impairment to initialise it.
- tx_LoRa.jl contains functions to generate LoRa frames. 
- tx_singleCarrier.jl contains function to generate single carrier frame 
- tx.jl contains function to generate OFDM frame 
- utils_dict.jl various functions
In RiFyFi_VDG.jl you will found different function to create the dataset based on virtual database. The most imporant one are described bellow.

**SetSyntheticCsv**: This function takes as input a structure describing the desired dataset and creates 4 CSV files: training data, test data and the corresponding labels for training and test data.
This function calls the "create_virtual_Database" function.


**This function creates or loads the configuration and scenario files and then calls generateDatabase twice to obtain the training and test data sets.
The configuration file stores information about the number of transmitters, the type of data, the amount of data ... 
The scenario file stores the interference parameters for each transmitter. 
You can create your own scenario file and use it to create the database, or you can use the automatic and random cases. 
In the automatic and random case, the impairments of two different stations may have a very close value. In this case it may be difficult for the network to separate the two transmitters.
This function can be used with the Augmentation package to add a propagation channel model to the data.
Then the dataset is shuffled (not necessary because the dataloader format shuffles the data during network training).


**generateDatabase**: this function creates a tup with various information, in particular dict_out contains the scenario parameters. bigMat contains the data matrix. 



**loadCSV_Synthetic** This function allows to load the CSV files corresponding to the database that the data structure described.
You can create a database with the previous functions and then use it many times to train different networks without having to generate the data again.


**reloadScenario** This function is called when you want to load a specific scenario file. The function returns the scenario interference value for each transmitter. 