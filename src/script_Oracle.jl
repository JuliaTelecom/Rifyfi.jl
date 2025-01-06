using RiFyFi
using Infiltrator
using RFImpairmentsModels 

include("Augmentation/src/Augmentation.jl")
using .Augmentation

include("RiFyFi_IdF/src/RiFyFi_IdF.jl")
using .RiFyFi_IdF

include("Results/src/Results.jl")
using .Results

include("Oracle_Database/src/Oracle_Database.jl")
using .Oracle_Database


########### Oracle Data struct ###########
distance =["62ft"]      # ["2ft","8ft","14ft","20ft","26ft","32ft","38ft","44ft","50ft","56ft","62ft"] or ["all_ft"]
run="2"                 # two records are available 1 and 2 
#File_name::String = "/media/HDD/achillet/RF_Fingerprint/Database/KRI-16Devices-RawData/"
name = "Oracle"        
nbTx= 6                 # Number of transmitters 16 are available
nbSignals = 1000        # Number of sequences of 256 IQ samples
Chunksize = 256         # size of the sequence 128, or 256
#features= "IQsamples"  # Can be used to preprocess data and test different pre-processing 
#Normalisation = true
#pourcentTrain =0.9     # pourcentage of data used for training 



########### Augmentation struct ###########
augmentationType = "No_channel"     # "No_channel": no augmentation is applied, "augment" : an augmentation is applied and depend nb_Augment
Channel = "etu"                     # etu, eva or multipath
Channel_Test = "etu"                # etu, eva or multipath
nb_Augment = 1                      # augmentation ratio 1 just to obtains nbSignals, 100 gives 100*nbSignals
#seed_channel = 12
#seed_channel_test = 12
#burstSize =64

Augmentation_Value = Oracle_Database.Data_Augmented(;augmentationType,Channel,Channel_Test,nb_Augment)




########### Args Network struct ###########

η = 1e-4             # learning rate e-5
dr = 0.25            # Dropout
#λ = 0               # L2 regularizer param, implemented as weight decay
batchsize = 64       # batch size
epochs = 10          # number of epochs
#seed = 12           # set seed > 0 for reproducibility
use_cuda = true      # if true use cuda (if available)
#infotime = 1 	     # report every `infotime` epochs
#checktime = 0       # Save the model every `checktime` epochs. Set to 0 for no checkpoints.
#tblogger = true     # log training with tensorboard
#tInit       = 0.0 
#timings    = zeros(epochs) # Store timings of train 



########### Network struct ###########
Networkname = "AlexNet"         # Define the network structure refering to their name
NbClass = nbTx                  # The number of class here correspond to the number of transmitters
#Chunksize = 256 
NbSignals = nbSignals
Seed_Network = 12
#Train_args =  Args()
#model  = initAlexNet(256,4,Train_args.dr)[1]
#loss = initAlexNet(256,4,Train_args.dr)[2]
Train_args = RiFyFi_IdF.Args(η = η ,dr=dr, epochs= epochs,batchsize=batchsize,use_cuda=use_cuda)





savepathbson=""
# Database creation
Param_Data=Oracle_Database.Data_Oracle(;distance=distance,run="2",nbTx=6,nbSignals=NbSignals,Chunksize=Chunksize)
Oracle_Database.setOraclecsv(Param_Data)


# Define Network structure
Param_Network = RiFyFi_IdF.Network_struct(;Networkname,NbClass,Chunksize,NbSignals,Seed_Network,Train_args) 
# Train network with the database
RiFyFi.main(Param_Data,Param_Network)  


distanceTest = ["62ft"]
# Database creation for test 
Param_Data_test = Oracle_Database.Data_Oracle(;distance=distanceTest,run="2",nbTx=6,nbSignals=NbSignals,Chunksize=Chunksize)
Oracle_Database.setOraclecsv(Param_Data_test)

Results.main(Param_Data,Param_Network,"Confusion_Matrix",savepathbson,Param_Data_test,Seed_Network)
 


