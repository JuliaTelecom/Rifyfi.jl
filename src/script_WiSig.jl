using RiFyFi
using Infiltrator
using RFImpairmentsModels 
using DelimitedFiles

include("Augmentation/src/Augmentation.jl")
using .Augmentation

include("RiFyFi_IdF/src/RiFyFi_IdF.jl")
using .RiFyFi_IdF

include("Results/src/Results.jl")
using .Results

include("WiSig_Database/src/WiSig_Database.jl")
using .WiSig_Database

include("Results/src/Create_palette.jl")
using .Palette

########### WiSig Data struct ###########
#File_name= "DataBases/WiSig/ManySig/pkl_wifi_ManySig/ManySig.pkl"
#name= "WiSig"
nbTx = 6                    # depends of the dataset : here only 6 transmitters in ManySig
nbSignals = 1000            # corresponds to the number of sequences of 256 IQ samples
#Chunksize = 256            # The sequences size considering here 
#features= "IQsamples"      # Can be used to preprocess data and test different pre-processing 
#txs = 1:6                  # Define the transmitters used here the transmitters 1 to 6
#rxs = 1                    # define the receiver(s) used, 12 receivers are available in ManySig, possible to use severals with the notation 1:3
days = 1                    # define the recording day(s) used, 4 days are available in ManySig, possible to use severals with the notation 1:3
equalized= 2                # Use equalized data 2, or no-equalized data 1.
#Normalisation = true
#pourcentTrain::Float64 =0.9  # pourcentage of data used for training 

########### Augmentation struct ###########
# Augmentation allow to applied a propagation channel model on data 
augmentationType = "No_channel"     # "No_channel": no augmentation is applied, "augment" : an augmentation is applied and depend nb_Augment
Channel = "etu"                     # etu, eva or multipath
Channel_Test = "etu"                # etu, eva or multipath
nb_Augment = 1                      # augmentation ratio 1 just to obtains nbSignals, 100 gives 100*nbSignals
#seed_channel = 12
#seed_channel_test = 12
#burstSize =64

Augmentation_Value = WiSig_Database.Data_Augmented(;augmentationType,Channel,Channel_Test,nb_Augment)



########### Args Network struct ###########

η = 1e-4             # learning rate 
dr = 0.25            # Dropout
#λ = 0               # L2 regularizer param, implemented as weight decay
batchsize = 600      # batch size
epochs = 10          # number of epochs
#seed = 12           # set seed > 0 for reproducibility
use_cuda = true      # if true use cuda (if available)
#infotime = 1 	     # report every `infotime` epochs
#checktime = 0       # Save the model every `checktime` epochs. Set to 0 for no checkpoints.
#tblogger = true     # log training with tensorboard
#tInit = 0.0 
#timings = zeros(epochs) # Store timings of train 



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


# ---------------------------------------------------------------------------------------------
savepathbson=""

dayTrain=1
# Database creation
Param_Data=WiSig_Database.Data_WiSig(equalized=equalized,days=dayTrain,Chunksize=Chunksize,Augmentation_Value=Augmentation_Value)
WiSig_Database.setWiSigcsv(Param_Data)
# Define Network structure
Param_Network = RiFyFi_IdF.Network_struct(;Networkname,NbClass,Chunksize,NbSignals,Seed_Network,Train_args) 
# Train network with the database
RiFyFi.main(Param_Data,Param_Network)  


dayTest=1
# Database creation for test 
Param_Data_test = WiSig_Database.Data_WiSig(equalized=equalized,days=dayTest,Chunksize=Chunksize,Augmentation_Value=Augmentation_Value)
WiSig_Database.setWiSigcsv(Param_Data_test)

Results.main(Param_Data,Param_Network,"Confusion_Matrix",savepathbson,Param_Data_test,Seed_Network)



