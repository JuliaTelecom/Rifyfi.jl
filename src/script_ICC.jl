
using RiFyFi
using Infiltrator
using RFImpairmentsModels 

include("Augmentation.jl/src/Augmentation.jl")
using .Augmentation

include("RiFyFi_VDG.jl/src/RiFyFi_VDG.jl")
using .RiFyFi_VDG

include("RiFyFi_IdF.jl/src/RiFyFi_IdF.jl")
using .RiFyFi_IdF

include("Results.jl/src/Results.jl")
using .Results

#=
# Parameters DataBase Synth
name= "control_1_fixe"
nameModel= name
nbRadioTx=6
NbSignals= 1000
Chunksize = 256
features="IQsamples"
E="E3"
S="S1"
C="C2"#_20dB"
RFF="all_impairments" 
Normalisation=true
pourcentTrain = 0.9
configuration ="scenario"
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

# Augmentation
#nb_Augment=1
Table_nb_Augment=[1]
augmentationType="sans"
Channel="etu"
Channel_Test="etu"


# Network Parameters
Networkname="AlexNet"
NbClass =nbRadioTx
#Chunksize 
#NbSignals
Table_Seed_Network= [12,13,14]#,44,55]

# Train Args
η = 1e-5   # learning rate 
dr = 0.25      # dropout 
batchsize = 64
epochs = 2000
use_cuda=true  

savepathbson=""
for i =1: 1 :size(Table_nb_Augment,1)
    nb_Augment = Table_nb_Augment[i]
    Augmentation_Value = Augmentation.Data_Augmented_construct(augmentationType=augmentationType,nb_Augment=nb_Augment,Channel=Channel,Channel_Test=Channel_Test)
    Param_Data = RiFyFi_VDG.Data_Synth(name,nameModel,nbRadioTx, NbSignals, Chunksize,features,S,E,C,RFF,Normalisation,pourcentTrain,configuration,seed_data,seed_model,seed_dataTest,seed_modelTest,Augmentation_Value)
   
     setSynthetiquecsv(Param_Data)
    
    for k = 1 :1: size(Table_Seed_Network,1)
    Seed_Network= Table_Seed_Network[k]
    Train_args = RiFyFi_IdF.Args(η = η ,dr=dr, epochs= epochs,batchsize=batchsize)
    Param_Network = RiFyFi_IdF.Network_struct(;Networkname,NbClass,Chunksize,NbSignals,Seed_Network,Train_args) 
    savepathbson = RiFyFi.main(Param_Data,Param_Network)   #filename is the .pkl file    
    end 
    
    nameTable = [name]
    Seed_Network= Table_Seed_Network[1]
    Train_args = RiFyFi_IdF.Args(η = η ,dr=dr, epochs= epochs,batchsize=batchsize)
    Param_Network = RiFyFi_IdF.Network_struct(;Networkname,NbClass,Chunksize,NbSignals,Seed_Network,Train_args) 
    Results.main(Param_Data,Param_Network,"F1_score",Table_Seed_Network,savepathbson)
   # Results.main(Param_Data,Param_Network,"Confusion_Matrix",Table_Seed_Network,savepathbson)

end 




name= "control_2_fixe"
nameModel= name

for i =1: 1 :size(Table_nb_Augment,1)
    nb_Augment = Table_nb_Augment[i]
    Augmentation_Value = Augmentation.Data_Augmented_construct(augmentationType=augmentationType,nb_Augment=nb_Augment,Channel=Channel,Channel_Test=Channel_Test)
    Param_Data = RiFyFi_VDG.Data_Synth(name,nameModel,nbRadioTx, NbSignals, Chunksize,features,S,E,C,RFF,Normalisation,pourcentTrain,configuration,seed_data,seed_model,seed_dataTest,seed_modelTest,Augmentation_Value)
   
     setSynthetiquecsv(Param_Data)
    
    for k = 1 :1: size(Table_Seed_Network,1)
    Seed_Network= Table_Seed_Network[k]
    Train_args = RiFyFi_IdF.Args(η = η ,dr=dr, epochs= epochs,batchsize=batchsize)
    Param_Network = RiFyFi_IdF.Network_struct(;Networkname,NbClass,Chunksize,NbSignals,Seed_Network,Train_args) 
    savepathbson = RiFyFi.main(Param_Data,Param_Network)   #filename is the .pkl file    
    end 
    
    nameTable = [name]
    Seed_Network= Table_Seed_Network[1]
    Train_args = RiFyFi_IdF.Args(η = η ,dr=dr, epochs= epochs,batchsize=batchsize)
    Param_Network = RiFyFi_IdF.Network_struct(;Networkname,NbClass,Chunksize,NbSignals,Seed_Network,Train_args) 
    Results.main(Param_Data,Param_Network,"F1_score",Table_Seed_Network,savepathbson)
   # Results.main(Param_Data,Param_Network,"Confusion_Matrix",Table_Seed_Network,savepathbson)

end 


name= "control_3_fixe"
nameModel= name

for i =1: 1 :size(Table_nb_Augment,1)
    nb_Augment = Table_nb_Augment[i]
    Augmentation_Value = Augmentation.Data_Augmented_construct(augmentationType=augmentationType,nb_Augment=nb_Augment,Channel=Channel,Channel_Test=Channel_Test)
    Param_Data = RiFyFi_VDG.Data_Synth(name,nameModel,nbRadioTx, NbSignals, Chunksize,features,S,E,C,RFF,Normalisation,pourcentTrain,configuration,seed_data,seed_model,seed_dataTest,seed_modelTest,Augmentation_Value)
   
     setSynthetiquecsv(Param_Data)
    
    for k = 1 :1: size(Table_Seed_Network,1)
    Seed_Network= Table_Seed_Network[k]
    Train_args = RiFyFi_IdF.Args(η = η ,dr=dr, epochs= epochs,batchsize=batchsize)
    Param_Network = RiFyFi_IdF.Network_struct(;Networkname,NbClass,Chunksize,NbSignals,Seed_Network,Train_args) 
    savepathbson = RiFyFi.main(Param_Data,Param_Network)   #filename is the .pkl file    
    end 
    
    nameTable = [name]
    Seed_Network= Table_Seed_Network[1]
    Train_args = RiFyFi_IdF.Args(η = η ,dr=dr, epochs= epochs,batchsize=batchsize)
    Param_Network = RiFyFi_IdF.Network_struct(;Networkname,NbClass,Chunksize,NbSignals,Seed_Network,Train_args) 
    Results.main(Param_Data,Param_Network,"F1_score",Table_Seed_Network,savepathbson)
   # Results.main(Param_Data,Param_Network,"Confusion_Matrix",Table_Seed_Network,savepathbson)

end 


name= "control_5_fixe"
nameModel= name

for i =1: 1 :size(Table_nb_Augment,1)
    nb_Augment = Table_nb_Augment[i]
    Augmentation_Value = Augmentation.Data_Augmented_construct(augmentationType=augmentationType,nb_Augment=nb_Augment,Channel=Channel,Channel_Test=Channel_Test)
    Param_Data = RiFyFi_VDG.Data_Synth(name,nameModel,nbRadioTx, NbSignals, Chunksize,features,S,E,C,RFF,Normalisation,pourcentTrain,configuration,seed_data,seed_model,seed_dataTest,seed_modelTest,Augmentation_Value)
   
     setSynthetiquecsv(Param_Data)
    
    for k = 1 :1: size(Table_Seed_Network,1)
    Seed_Network= Table_Seed_Network[k]
    Train_args = RiFyFi_IdF.Args(η = η ,dr=dr, epochs= epochs,batchsize=batchsize)
    Param_Network = RiFyFi_IdF.Network_struct(;Networkname,NbClass,Chunksize,NbSignals,Seed_Network,Train_args) 
    savepathbson = RiFyFi.main(Param_Data,Param_Network)   #filename is the .pkl file    
    end 
    
    nameTable = [name]
    Seed_Network= Table_Seed_Network[1]
    Train_args = RiFyFi_IdF.Args(η = η ,dr=dr, epochs= epochs,batchsize=batchsize)
    Param_Network = RiFyFi_IdF.Network_struct(;Networkname,NbClass,Chunksize,NbSignals,Seed_Network,Train_args) 
    Results.main(Param_Data,Param_Network,"F1_score",Table_Seed_Network,savepathbson)
   # Results.main(Param_Data,Param_Network,"Confusion_Matrix",Table_Seed_Network,savepathbson)

end 

=#


###############################


# Parameters DataBase Synth
name= "control_3_fixe"
nameModel= name
nbRadioTx=6
NbSignals= 1000
Chunksize = 256
features="IQsamples"
E="E3"
S="S1"
C="C2"#_20dB"
RFF="all_impairments" 
Normalisation=true
pourcentTrain = 0.9
configuration ="scenario"
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

# Augmentation
#nb_Augment=1
Table_nb_Augment=[2,3,4,6,8,10]
augmentationType="augment"
Channel="etu"
Channel_Test="eva"


# Network Parameters
Networkname="AlexNet"
NbClass =nbRadioTx
#Chunksize 
#NbSignals
Table_Seed_Network= [12,13,14]#,44,55]

# Train Args
η = 1e-5   # learning rate 
dr = 0.25      # dropout 
batchsize = 64
epochs = 2000
use_cuda=true  

savepathbson=""

for i =1: 1 :size(Table_nb_Augment,1)
    nb_Augment = Table_nb_Augment[i]
    Augmentation_Value = Augmentation.Data_Augmented_construct(augmentationType=augmentationType,nb_Augment=nb_Augment,Channel=Channel,Channel_Test=Channel_Test)
    Param_Data = RiFyFi_VDG.Data_Synth(name,nameModel,nbRadioTx, NbSignals, Chunksize,features,S,E,C,RFF,Normalisation,pourcentTrain,configuration,seed_data,seed_model,seed_dataTest,seed_modelTest,Augmentation_Value)
   
     setSynthetiquecsv(Param_Data)
    
    for k = 1 :1: size(Table_Seed_Network,1)
    Seed_Network= Table_Seed_Network[k]
    Train_args = RiFyFi_IdF.Args(η = η ,dr=dr, epochs= epochs,batchsize=batchsize)
    Param_Network = RiFyFi_IdF.Network_struct(;Networkname,NbClass,Chunksize,NbSignals,Seed_Network,Train_args) 
    savepathbson = RiFyFi.main(Param_Data,Param_Network)   #filename is the .pkl file    
    end 
    
    nameTable = [name]
    Seed_Network= Table_Seed_Network[1]
    Train_args = RiFyFi_IdF.Args(η = η ,dr=dr, epochs= epochs,batchsize=batchsize)
    Param_Network = RiFyFi_IdF.Network_struct(;Networkname,NbClass,Chunksize,NbSignals,Seed_Network,Train_args) 
    Results.main(Param_Data,Param_Network,"F1_score",Table_Seed_Network,savepathbson)
   # Results.main(Param_Data,Param_Network,"Confusion_Matrix",Table_Seed_Network,savepathbson)

end 



name= "control_5_fixe"
nameModel= name
nbRadioTx=6
NbSignals= 1000
Chunksize = 256
features="IQsamples"
E="E3"
S="S1"
C="C2"#_20dB"
RFF="all_impairments" 
Normalisation=true
pourcentTrain = 0.9
configuration ="scenario"
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

# Augmentation
#nb_Augment=1
Table_nb_Augment=[2,3,4,6,8,10]
augmentationType="augment"
Channel="etu"
Channel_Test="eva"


# Network Parameters
Networkname="AlexNet"
NbClass =nbRadioTx
#Chunksize 
#NbSignals
Table_Seed_Network= [12,13,14]#,44,55]

# Train Args
η = 1e-5   # learning rate 
dr = 0.25      # dropout 
batchsize = 64
epochs = 2000
use_cuda=true  

savepathbson=""

for i =1: 1 :size(Table_nb_Augment,1)
    nb_Augment = Table_nb_Augment[i]
    Augmentation_Value = Augmentation.Data_Augmented_construct(augmentationType=augmentationType,nb_Augment=nb_Augment,Channel=Channel,Channel_Test=Channel_Test)
    Param_Data = RiFyFi_VDG.Data_Synth(name,nameModel,nbRadioTx, NbSignals, Chunksize,features,S,E,C,RFF,Normalisation,pourcentTrain,configuration,seed_data,seed_model,seed_dataTest,seed_modelTest,Augmentation_Value)
   
     setSynthetiquecsv(Param_Data)
    
    for k = 1 :1: size(Table_Seed_Network,1)
    Seed_Network= Table_Seed_Network[k]
    Train_args = RiFyFi_IdF.Args(η = η ,dr=dr, epochs= epochs,batchsize=batchsize)
    Param_Network = RiFyFi_IdF.Network_struct(;Networkname,NbClass,Chunksize,NbSignals,Seed_Network,Train_args) 
    savepathbson = RiFyFi.main(Param_Data,Param_Network)   #filename is the .pkl file    
    end 
    
    nameTable = [name]
    Seed_Network= Table_Seed_Network[1]
    Train_args = RiFyFi_IdF.Args(η = η ,dr=dr, epochs= epochs,batchsize=batchsize)
    Param_Network = RiFyFi_IdF.Network_struct(;Networkname,NbClass,Chunksize,NbSignals,Seed_Network,Train_args) 
    Results.main(Param_Data,Param_Network,"F1_score",Table_Seed_Network,savepathbson)
   # Results.main(Param_Data,Param_Network,"Confusion_Matrix",Table_Seed_Network,savepathbson)

end 
