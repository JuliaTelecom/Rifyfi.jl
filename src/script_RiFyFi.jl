# --------------------------------------------------------------------------------
# This script allow to create a virtual database thanks to RiFyFi_VDG package 
# And use this database to train a network and then create confusion matrix and F1-score evolution
# --------------------------------------------------------------------------------

using RiFyFi
using Infiltrator
using RFImpairmentsModels 

include("Augmentation/src/Augmentation.jl")
using .Augmentation

include("RiFyFi_VDG/src/RiFyFi_VDG.jl")
using .RiFyFi_VDG

include("RiFyFi_IdF/src/RiFyFi_IdF.jl")
using .RiFyFi_IdF

include("Results/src/Results.jl")
using .Results



########### Synthetic Data struct ###########
name = "5_pourcent"
nameModel = name
nbTx = 6            # Number of transmitters
nbSignals = 10000   # number of signals per transmitters
Chunksize = 256     # number of IQ samples per signals
features= "IQsamples"
S = "S1"            # Use S1 for modelling a Preamble mode, S2 for MAC address and S3 for payload mode
E = "E3"            # Use E3 for adding fingerprint 
C = "C2_20dB"       # Use C1 for perfect SNR, C2_0dB - C2_30dB to add Gaussian noise
RFF = "all_impairments"     # Use all_impairments to modeled the complete chaine, or use PA to model only the Power Amplifier, PN for Phase Noise, imbalance for IQ imbalance or cfo for carrier frequency offset.
Normalisation = true        # Use true to normalize the database 
pourcentTrain =0.9          # 90 % for train and 10% for test 
configuration  = "scenario" # Use nothing to create random scenario, or use "scenario" to load a pre create scenario 
seed_data = 1234
seed_model = 2345
if E == "E1" || E == "E2"
    seed_modelTest = seed_model 
else 
    seed_modelTest = 15987654321 * 100000000
end 
if S == "S1" || S == "S2"
    seed_dataTest = seed_data 
else 
    seed_dataTest = 9999246912 * 100000000
end 



########### Augmentation struct ###########
augmentationType = "augment"
Channel = "etu"
Channel_Test = "etu"
nb_Augment = 100
#seed_channel = 12
#seed_channel_test = 12
#burstSize =64

Augmentation_Value = RiFyFi_VDG.Data_Augmented(;augmentationType,Channel,Channel_Test,nb_Augment)




########### Args Network struct ###########

η = 1e-4           # learning rate e-5
dr = 0.25
#λ = 0               # L2 regularizer param, implemented as weight decay
batchsize = 600     # batch size
epochs = 1000        # number of epochs
#seed = 12           # set seed > 0 for reproducibility
use_cuda = true     # if true use cuda (if available)
#infotime = 1 	    # report every `infotime` epochs
#checktime = 0       # Save the model every `checktime` epochs. Set to 0 for no checkpoints.
#tblogger = true     # log training with tensorboard
#tInit       = 0.0 
#timings    = zeros(epochs) # Store timings of train 



########### Network struct ###########
Networkname = "AlexNet"
NbClass = nbTx
#Chunksize = 256
NbSignals = nbSignals
Seed_Network = 12
#Train_args =  Args()
#model  = initAlexNet(256,4,Train_args.dr)[1]
#loss = initAlexNet(256,4,Train_args.dr)[2]
Train_args = RiFyFi_IdF.Args(η = η ,dr=dr, epochs= epochs,batchsize=batchsize,use_cuda=use_cuda)



# ---------------------------------------------------------------------------------------------
savepathbson=""

# Creation of the data structure with the information of the dataset
Param_Data = RiFyFi_VDG.Data_Synth(name,nameModel,nbTx, NbSignals, Chunksize,features,S,E,C,RFF,Normalisation,pourcentTrain,configuration,seed_data,seed_model,seed_dataTest,seed_modelTest,Augmentation_Value)
# Train and test Datasets are created and saved in CSV files
#RiFyFi_VDG.setSynthetiquecsv(Param_Data)


# Creation of the Network structure with the information of the network
Param_Network = RiFyFi_IdF.Network_struct(;Networkname,NbClass,Chunksize,NbSignals,Seed_Network,Train_args) 
# Train the network and save it 
RiFyFi.main(Param_Data,Param_Network)   

# Create a figure to show the evolution of the F1-score during the training 
Results.main(Param_Data,Param_Network,"F1_score",savepathbson,Param_Data,[Seed_Network])
# Create a confusion matrix with testing dataset
Results.main(Param_Data,Param_Network,"Confusion_Matrix",savepathbson,Param_Data,[Seed_Network])


# Define an other transmission scenario with the same RFF
NbSignals_test = 1000
C_test = "C2"
configuration  = "scenario" # use the previous RFF scenario to create new signals

# Use for example data augmentation to add different channel realisation 
Augmentation_Value_test = Augmentation.Data_Augmented_construct(augmentationType="augment",nb_Augment=2,Channel=Channel,Channel_Test=Channel_Test)
# Creation of the new data structure with the information of the dataset
Param_Data_test = RiFyFi_VDG.Data_Synth(name,nameModel,nbTx, NbSignals_test, Chunksize,features,S,E,C_test,RFF,Normalisation,pourcentTrain,configuration,seed_data,seed_model,seed_dataTest,seed_modelTest,Augmentation_Value_test)
# Train and test Datasets are created and saved in CSV files


RiFyFi_VDG.setSynthetiquecsv(Param_Data_test)

# Use the Param_Data_test dataset to evaluated the resilience of the trained network in the new transmission scenario
Results.main(Param_Data,Param_Network,"Confusion_Matrix",savepathbson,Param_Data_test,[Seed_Network])


 








