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
name = "Test"
nameModel = name
nbTx = 6
nbSignals = 1000
Chunksize = 256
features= "IQsamples"
S = "S1"
E = "E3"
C = "C2_20dB"
RFF = "all_impairments"
Normalisation = true
pourcentTrain =0.9
configuration  = "nothing"
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
augmentationType = "No_channel"
Channel = "etu"
Channel_Test = "etu"
nb_Augment = 1
#seed_channel = 12
#seed_channel_test = 12
#burstSize =64

Augmentation_Value = RiFyFi_VDG.Data_Augmented(;augmentationType,Channel,Channel_Test,nb_Augment)




########### Args Network struct ###########

η = 1e-4            # learning rate e-5
dr = 0.25
#λ = 0               # L2 regularizer param, implemented as weight decay
batchsize = 600     # batch size
epochs = 10        # number of epochs
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


Param_Data = RiFyFi_VDG.Data_Synth(name,nameModel,nbTx, NbSignals, Chunksize,features,S,E,C,RFF,Normalisation,pourcentTrain,configuration,seed_data,seed_model,seed_dataTest,seed_modelTest,Augmentation_Value)
RiFyFi_VDG.setSynthetiquecsv(Param_Data)



Param_Network = RiFyFi_IdF.Network_struct(;Networkname,NbClass,Chunksize,NbSignals,Seed_Network,Train_args) 
RiFyFi.main(Param_Data,Param_Network)   #filename is the .pkl file    


NbSignals_test =10000
C_test="C2"
configuration  = "scenario" # use the previous RFF scenario to create new signals

Augmentation_Value_test = Augmentation.Data_Augmented_construct(augmentationType="augment",nb_Augment=1,Channel=Channel,Channel_Test=Channel_Test)
Param_Data_test = RiFyFi_VDG.Data_Synth(name,nameModel,nbTx, NbSignals_test, Chunksize,features,S,E,C_test,RFF,Normalisation,pourcentTrain,configuration,seed_data,seed_model,seed_dataTest,seed_modelTest,Augmentation_Value_test)
RiFyFi_VDG.setSynthetiquecsv(Param_Data_test)


Results.main(Param_Data,Param_Network,"F1_score",savepathbson,Param_Data,[Seed_Network])
Results.main(Param_Data,Param_Network,"Confusion_Matrix",savepathbson,Param_Data_test,[Seed_Network])

 








