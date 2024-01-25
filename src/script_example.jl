
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

# Parameters DataBase Synth
name= "Example_Database" 
nameModel= name
nbRadioTx=6 # Number of transmitters
NbSignals=1000 # number of signals per transmitters
Chunksize = 256 # number of IQ samples per signals
features="IQsamples" 
E="E3" # Use E3 for adding fingerprint 
S="S1" # Use S1 for modelling a Preamble mode, S2 for MAC address and S3 for payload mode
C="C2" # Use C1 for perfect SNR, C2_0dB - C2_30dB to add Gaussian noise
RFF="all_impairments"  # Use all_impairments to modeled the complete chaine, or use PA to model only the Power Amplifier, PN for Phase Noise, imbalance for IQ imbalance or cfo for carrier frequency offset.
Normalisation=true # Use true to normalize the database 
pourcentTrain = 0.9 # 90 % for train and 10% for test 
configuration ="nothing" # Use nothing to create random scenario, or use "scenario" to load a pre create scenario 
# Define the different seed for model and data 
seed_model =  234567   
seed_data = 123456
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

# Parameters for Data Augmentation
nb_Augment= 1  # Define the number of channel for each transmitter
augmentationType="No_channel" # use No_channel, augment or same_channel
Channel="etu" 
Channel_Test="eva"


# Network Parameters
Networkname="AlexNet" # Name of the Network
NbClass =nbRadioTx 
#Chunksize 
#NbSignals
Seed_Network= 11 

# Train Args
η = 1e-5   # learning rate 
dr = 0.25  # dropout 
batchsize = 64
epochs = 2000
use_cuda = true  

savepathbson=""


Augmentation_Value = Augmentation.Data_Augmented_construct(augmentationType=augmentationType,nb_Augment=nb_Augment,Channel=Channel,Channel_Test=Channel_Test)
Param_Data = RiFyFi_VDG.Data_Synth(name,nameModel,nbRadioTx, NbSignals, Chunksize,features,S,E,C,RFF,Normalisation,pourcentTrain,configuration,seed_data,seed_model,seed_dataTest,seed_modelTest,Augmentation_Value)
setSynthetiquecsv(Param_Data)

Train_args = RiFyFi_IdF.Args(η = η, dr=dr, epochs= epochs, batchsize=batchsize)
Param_Network = RiFyFi_IdF.Network_struct(;Networkname,NbClass,Chunksize,NbSignals,Seed_Network,Train_args) 
RiFyFi.main(Param_Data,Param_Network)     
 
    
nameTable = [name]
Seed_Network= Table_Seed_Network[1]
Train_args = RiFyFi_IdF.Args(η = η ,dr=dr, epochs= epochs,batchsize=batchsize)
Param_Network = RiFyFi_IdF.Network_struct(;Networkname,NbClass,Chunksize,NbSignals,Seed_Network,Train_args) 
Results.main(Param_Data,Param_Network,"F1_score",Table_Seed_Network,savepathbson)
Results.main(Param_Data,Param_Network,"Confusion_Matrix",Table_Seed_Network,savepathbson)





