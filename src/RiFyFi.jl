module RiFyFi
# ----------------------------------------------------
# --- External Dependencies 
# ---------------------------------------------------- 
using BSON 
using CSV 
using CUDA 
using ColorSchemes 
using DSP 
using DataFrames 
using DelimitedFiles 
using DigitalComm 
using Distributed 
using FFTW 
using Flux 
using Infiltrator 
using JSON
using LinearAlgebra 
using Logging 
using MAT
using OrderedCollections
using PGFPlotsX 
using Pickle 
using Plots 
using ProgressMeter 
using Random 
using Statistics 
include("RiFyFi_VDG/src/RiFyFi_VDG.jl")
using .RiFyFi_VDG
include("RiFyFi_IdF/src/RiFyFi_IdF.jl")
using .RiFyFi_IdF

include("Oracle_Database/src/Oracle_Database.jl")
using .Oracle_Database
include("WiSig_Database/src/WiSig_Database.jl")
using .WiSig_Database

include("Experiment_Database/src/Experiment_Database.jl")
using .Experiment_Database
# ----------------------------------------------------
# --- Loading utility functions
# ---------------------------------------------------- 
include("utils.jl")
# ----------------------------------------------------
# --- Exportation 
# ---------------------------------------------------- 
export init
export main
export convertDataLabelToVect
export confusionMatrix
export getAccuracy



""" Function that drives the cnn and saves it in .bson\n
    Parameters : 
    - Param_Data type Data_Synth ou Data_WiSig 
    - Param_Network type of Network_struct 
"""
function main(Param_Data,Param_Network)  
    if Param_Network.Train_args.use_cuda 
        hardware= "GPU"
    else 
        hardware= "CPU"
    end
    # Load the datasets and init network
    (dataTrain,dataTest) = init(Param_Data,Param_Network)
    if Param_Data.name== "WiSig"
        if Param_Data.Augmentation_Value.augmentationType == "No_channel"
            savepath = "run/WiSig/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)_$(Param_Network.Networkname)/$(Param_Data.txs)_$(Param_Data.rxs)/$(Param_Data.txs)_$(Param_Data.rxs)_$(Param_Data.days)_$(Param_Data.equalized)_$(Param_Data.nbSignals)/$(hardware)"
        else
            savepath = "run/WiSig/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)_$(Param_Network.Networkname)/$(Param_Data.txs)_$(Param_Data.rxs)/$(Param_Data.txs)_$(Param_Data.rxs)_$(Param_Data.days)_$(Param_Data.equalized)_$(Param_Data.nbSignals)_$(Param_Data.Augmentation_Value.Channel)_$(Param_Data.Augmentation_Value.Channel_Test)_nbAugment_$(Param_Data.Augmentation_Value.nb_Augment)/$(hardware)"
        end 
    elseif Param_Data.name== "Oracle"
        savepath = "run/Oracle/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)_$(Param_Network.Networkname)/$(Param_Data.nbSignals)_$(Param_Data.distance)/$(hardware)"
    elseif Param_Data.name== "Exp"
        savepath = "run/Experiment/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)_$(Param_Network.Networkname)_$(Param_Data.Type_of_sig)/Run$(Param_Data.run)_Test$(Param_Data.Test)_$(Param_Data.nbTx)_$(Param_Data.nbSignals)_$(Param_Data.noise)/$(hardware)"
    elseif Param_Data.Augmentation_Value.augmentationType == "No_channel" # Synthetic data 
        savepath = "run/Synth/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)_$(Param_Network.Networkname)/$(Param_Data.E)_$(Param_Data.S)/$(Param_Data.E)_$(Param_Data.S)_$(Param_Data.C)_$(Param_Data.RFF)_$(Param_Data.nbSignals)_$(Param_Data.nameModel)/$(hardware)"
    else # Synthetic data  with data augmentation
        savepath = "run/Synth/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)_$(Param_Network.Networkname)/$(Param_Data.E)_$(Param_Data.S)/$(Param_Data.E)_$(Param_Data.S)_$(Param_Data.C)_$(Param_Data.RFF)_$(Param_Data.nbSignals)_$(Param_Data.nameModel)_$(Param_Data.Augmentation_Value.Channel)_$(Param_Data.Augmentation_Value.Channel_Test)_nbAugment_$(Param_Data.Augmentation_Value.nb_Augment)/$(hardware)"
    end 
    !ispath(savepath) && mkpath(savepath)
    
    (model,trainLoss,trainAcc,testLoss,testAcc,args) = customTrain!(dataTrain,dataTest,savepath,Param_Network)
    # ----------------------------------------------------
    # --- Saving model 
    # ---------------------------------------------------- 
    modelpath = joinpath(savepath, "model_seed_$(Param_Network.Seed_Network)_dr$(Param_Network.Train_args.dr).bson") 
    nbEpochs = args.epochs
    BSON.@save modelpath model nbEpochs trainLoss trainAcc testLoss testAcc args 
    testAugmented_acc =0
    testAugmented_loss=0    
    return (savepath,model,trainLoss,trainAcc,testLoss,testAcc,testAugmented_loss,testAugmented_acc) 
end




""" Function that load Data in Matrix format and initialize the Network
    Parameters : 
    - Param_Data type Data_Synth ou Data_WiSig 
    - Param_Network type of Network_struct
"""
function init(Param_Data,Param_Network)
    # ----------------------------------------------------
    # --- Loading and pre-processing data 
    # ---------------------------------------------------- 
    @info "init"
    Random.seed!(Param_Network.Seed_Network)
    if Param_Data.name== "WiSig"
        (X_train,Y_train,X_test,Y_test)=WiSig_Database.loadCSV_WiSig(Param_Data)
    elseif Param_Data.name== "Oracle"
        (X_train,Y_train,X_test,Y_test)=Oracle_Database.loadCSV_Oracle(Param_Data)
    elseif Param_Data.name== "Exp"
        (X_train,Y_train,X_test,Y_test)=Experiment_Database.loadCSV_Exp(Param_Data)
    else # Synthetic data 
        (X_train,Y_train,X_test,Y_test)=loadCSV_Synthetic(Param_Data)
    end 
    # ----------------------------------------------------
    # --- Create datasets
    # ---------------------------------------------------- 
    rng = MersenneTwister(Param_Network.Seed_Network)
    dataTrain = Flux.Data.DataLoader((X_train,Y_train), batchsize = Param_Network.Train_args.batchsize,rng=rng, shuffle = true)
    dataTest  = Flux.Data.DataLoader((X_test, Y_test), batchsize = Param_Network.Train_args.batchsize,rng=rng, shuffle = true)
    # ----------------------------------------------------
    # --- Init Network
    # ---------------------------------------------------- 
    @info "Init Network "
    if Param_Network.Networkname == "AlexNet"
        (nn,loss)= initAlexNet(Param_Data.Chunksize,Param_Data.nbTx,Param_Network.Train_args.dr)
    elseif Param_Network.Networkname=="NewCNN"
        (nn,loss)= initNewCNN(Param_Data.Chunksize,Param_Data.nbTx,Param_Network.Train_args.dr)
    elseif Param_Network.Networkname == "GDA"
        (nn,loss)= initGDA(Param_Data.Chunksize,Param_Data.nbTx,Param_Network.Train_args.dr)
    elseif Param_Network.Networkname=="WiSig"
        (nn,loss)= initWiSig(Param_Data.Chunksize,Param_Data.nbTx,Param_Network.Train_args.dr)
    end 
    Param_Network.loss = loss
    Param_Network.model = nn
    return (dataTrain,dataTest)
end





end