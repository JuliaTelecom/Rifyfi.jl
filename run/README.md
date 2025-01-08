# run


The run folder contains the a save file of the trained network. 

The folder is organized as follow

First the type of Data : Synth for virtual, WiSig, Oracle or Exp
Then information about channel, number of transmitters, chunksize and network architecture
Then the folders depend to the type of data. 

At the end each folders contains a bson file with the network parameters, a csv file with the F1 evolution during the training and a latex document to create a tikzpicture of the F1 score evolution

The files here have been obtained by runing script_RiFyFi.jl with the scenario 5% defined in Configurations folders
