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
using Infiltrator
include("RiFyFi_VDG/src/RiFyFi_VDG.jl")
using .RiFyFi_VDG
include("RiFyFi_IdF/src/RiFyFi_IdF.jl")
using .RiFyFi_IdF

include("Augmentation/src/Augmentation.jl")
using .Augmentation

include("Results/src/Results.jl")
using .Results


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
    - Param_Data type Data_Synth  
    - Param_Network type of Network_struct 
"""
function main(Param_Data,Param_Network)  
     
    if Param_Network.Train_args.use_cuda 
        hardware= "GPU"
    else 
        hardware= "CPU"
    end

    (dataTrain,dataTest,dataTrain_dyn,dataTest_dyn) = init(Param_Data,Param_Network)
    

    if Param_Data.Augmentation_Value.augmentationType == "No_channel"
        savepath_model = "run/Synth/$(Param_Data.Modulation)/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)_$(Param_Network.Networkname)/$(Param_Data.E)_$(Param_Data.S)/$(Param_Data.E)_$(Param_Data.S)_$(Param_Data.C)_$(Param_Data.RFF)_$(Param_Data.nbSignals)_$(Param_Data.nameModel)/$(hardware)"
    else 
        savepath_model = "run/Synth/$(Param_Data.Modulation)/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)_$(Param_Network.Networkname)/$(Param_Data.E)_$(Param_Data.S)/$(Param_Data.E)_$(Param_Data.S)_$(Param_Data.C)_$(Param_Data.RFF)_$(Param_Data.nbSignals)_$(Param_Data.nameModel)_$(Param_Data.Augmentation_Value.Channel)_$(Param_Data.Augmentation_Value.Channel_Test)_nbAugment_$(Param_Data.Augmentation_Value.nb_Augment)/$(hardware)"
    end 
    !ispath(savepath_model) && mkpath(savepath_model)
    
    if Param_Data.RFF == "all_impairments_dynamic_cfo"
        (model,trainLoss,trainAcc,testLoss,testAcc,args) = customTrain!(dataTrain,dataTest,savepath_model,Param_Network,Param_Data.Modulation,dataTrain_dyn,dataTest_dyn)
    else   
        (model,trainLoss,trainAcc,testLoss,testAcc,args) = customTrain!(dataTrain,dataTest,savepath_model,Param_Network,Param_Data.Modulation)
    end 
    # ----------------------------------------------------
    # --- Saving model 
    # ---------------------------------------------------- 
    modelpath = joinpath(savepath_model, "model_seed_$(Param_Network.Seed_Network)_dr$(Param_Network.Train_args.dr)_$(Param_Data.Modulation).bson") 
    nbEpochs = args.epochs
    BSON.@save modelpath model nbEpochs trainLoss trainAcc testLoss testAcc args 
    testAugmented_acc =0
    testAugmented_loss=0    
    return (savepath_model,model,trainLoss,trainAcc,testLoss,testAcc,testAugmented_loss,testAugmented_acc) 
end




""" Function that load Data in Matrix format and initialize the Network
    Parameters : 
    - Param_Data type Data_Synth 
    - Param_Network type of Network_struct
"""
function init(Param_Data,Param_Network)
    # ----------------------------------------------------
    # --- Loading and pre-processing data 
    # ---------------------------------------------------- 
    @info "init"
    Random.seed!(Param_Network.Seed_Network)
    
    (X_train,Y_train,X_test,Y_test)=loadCSV_Synthetic(Param_Data)
    if Param_Data.RFF=="all_impairments_dynamic_cfo"
        (X_train_dyn,Y_train_dyn,X_test_dyn,Y_test_dyn)=loadCSV_Synthetic_dynCFO(Param_Data)
        rng = MersenneTwister(Param_Network.Seed_Network)
        dataTrain_dyn = Flux.Data.DataLoader((X_train_dyn,Y_train_dyn), batchsize = Param_Network.Train_args.batchsize,rng=rng, shuffle = true)
        dataTest_dyn  = Flux.Data.DataLoader((X_test_dyn, Y_test_dyn), batchsize = Param_Network.Train_args.batchsize,rng=rng, shuffle = true)
    else 
        dataTrain_dyn = 0
        dataTest_dyn = 0
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
    elseif Param_Network.Networkname=="DenseEmma"
        (nn,loss)= DenseEmma(Param_Data.Chunksize,Param_Data.nbTx,Param_Network.Train_args.dr)
    elseif Param_Network.Networkname == "GDA"
        (nn,loss)= initGDA(Param_Data.Chunksize,Param_Data.nbTx,Param_Network.Train_args.dr)
    elseif Param_Network.Networkname == "Merchant"
        (nn,loss)= initMerchant2021A(Param_Data.Chunksize,Param_Data.nbTx,Param_Network.Train_args.dr)
    elseif Param_Network.Networkname == "Shen"
        (nn,loss)= initShen(Param_Data.Chunksize,Param_Data.nbTx,Param_Network.Train_args.dr)
    elseif Param_Network.Networkname =="ResNet"
        (nn,loss)= initResNet(Param_Data.Chunksize,Param_Data.nbTx,Param_Network.Train_args.dr)
    end 
    Param_Network.loss = loss
    Param_Network.model = nn
    return (dataTrain,dataTest,dataTrain_dyn,dataTest_dyn)
end





end