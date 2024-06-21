module RiFyFi_VDG
# ----------------------------------------------------
# --- Dependencies
# ---------------------------------------------------- 
using RFImpairmentsModels # Modeling imperfections
using OrderedCollections
using JSON  
using DigitalComm  
using FFTW
using Random 
using LinearAlgebra
using PGFPlotsX 
using ColorSchemes 
using MAT
using Statistics
using CSV 
using DelimitedFiles
using DataFrames
using Infiltrator
using DSP
using Plots



# Augmentation Module, add channel model on data
include("../../Augmentation/src/Augmentation.jl") 
using .Augmentation


include("./DatBinaryFiles.jl")
using .DatBinaryFiles
# ----------------------------------------------------
# --- Export 
# ---------------------------------------------------- 
export getKey, @loadKey
export setSynthetiquecsv
export loadCSV_Synthetic
export loadCSV_Synthetic_dynCFO
#export init_dynCFO
# ----------------------------------------------------
# --- Loading utils on keys 
# ---------------------------------------------------- 
include("utils_dict.jl")

# ----------------------------------------------------
# --- Modulator & Demodulator
# ---------------------------------------------------- 
include("tx_OFDM.jl") # Tx for OFDM modulation
include("tx_singleCarrier.jl") # Tx for single Carrier Modulation
include("tx_LoRa.jl") # Tx for LoRa modulation
include("rx.jl") # Receiver (non use)
include("pa_memory.jl") # functions for PA impairments with memory model
include("dynamic_CFO.jl") # functions to simulate a dynamic CFO 


# ----------------------------------------------------
# --- Setup Impairments
# ---------------------------------------------------- 
include("setup_impairments.jl")

include("Struct_data.jl")
export Data_Synth
export Data_Synth_construct
# ----------------------------------------------------
# --- Main call 
# ---------------------------------------------------- 

""" Function that create 4 csv files based 
    - on a fixed scenario RFF parameter define by RiFyFi_VDG.Data_Synth structure.
    - on a configuration (define RFF parameters randomly)
    CSV files: 
    - Training Data
    - Test Data 
    - Training Labels
    - Test Labels
     """
function setSynthetiquecsv(Param_Data)
    
    if Param_Data.RFF == "all_impairments_dynamic_cfo"
        
        (bigMatTrain,bigLabel_Train,bigMatTest,bigLabel_Test,bigMatTrain_dyn,bigLabel_Train_dyn,bigMatTest_dyn,bigLabel_Test_dyn,X) =  create_virtual_Database_dynCFO(Param_Data)
       
        @infiltrate
        bigLabels_Train = create_bigMat_Labels_Tx(bigLabel_Train)
        bigLabels_Test = create_bigMat_Labels_Tx(bigLabel_Test)

        bigLabels_Train_dyn = create_bigMat_Labels_Tx(bigLabel_Train_dyn)
        bigLabels_Test_dyn = create_bigMat_Labels_Tx(bigLabel_Test_dyn)

        if Param_Data.Augmentation_Value.augmentationType == "No_channel"
            savepath_CSVfiles = "/media/HDD/achillet/RF_Fingerprint/Database/RiFyFi_Database/CSV_Files/$(Param_Data.Modulation)/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)/$(Param_Data.E)_$(Param_Data.S)/$(Param_Data.E)_$(Param_Data.S)_$(Param_Data.C)_$(Param_Data.RFF)_$(Param_Data.nbSignals)_$(Param_Data.nameModel)"

        else 
            savepath_CSVfiles = "/media/HDD/achillet/RF_Fingerprint/Database/RiFyFi_Database/CSV_Files/$(Param_Data.Modulation)/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)/$(Param_Data.E)_$(Param_Data.S)/$(Param_Data.E)_$(Param_Data.S)_$(Param_Data.C)_$(Param_Data.RFF)_$(Param_Data.nbSignals)_$(Param_Data.nameModel)_$(Param_Data.Augmentation_Value.Channel)_$(Param_Data.Augmentation_Value.Channel_Test)_nbAugment_$(Param_Data.Augmentation_Value.nb_Augment)"
        end
        !ispath(savepath_CSVfiles) && mkpath(savepath_CSVfiles)    

        # Save in CSV files 
        open("$(savepath_CSVfiles)/bigMatTrain.csv","w") do io
            for i in 1:size(bigMatTrain)[3]
                writedlm(io,[vcat(bigMatTrain[:,:,i][:,1],bigMatTrain[:,:,i][:,2])])  
            end
        end

        open("$(savepath_CSVfiles)/bigMatTest.csv","w") do io
            for i in 1:size(bigMatTest)[3]
                writedlm(io,[vcat(bigMatTest[:,:,i][:,1],bigMatTest[:,:,i][:,2])])   
            end
        end

        open("$(savepath_CSVfiles)/bigLabelsTrain.csv","w") do io
            for i in 1:size(bigLabels_Train)[1]                 
                writedlm(io,[bigLabels_Train[i]])                          
            end
        end

        open("$(savepath_CSVfiles)/bigLabelsTest.csv","w") do io
            for i in 1:size(bigLabels_Test)[1]              
                writedlm(io,[bigLabels_Test[i]])                     
            end
        end
        #########################################################
        # --- CFO Dynamique -------------------------------------
        #########################################################

        open("$(savepath_CSVfiles)/bigMatTrain_dyn.csv","w") do io
            for i in 1:size(bigMatTrain_dyn)[3]
                writedlm(io,[vcat(bigMatTrain_dyn[:,:,i][:,1],bigMatTrain_dyn[:,:,i][:,2])])  
            end
        end
        open("$(savepath_CSVfiles)/bigLabelsTrain_dyn.csv","w") do io
            for i in 1:size(bigLabels_Train_dyn)[1]                 
                writedlm(io,[bigLabels_Train_dyn[i]])                          
            end
        end
        open("$(savepath_CSVfiles)/bigMatTest_dyn.csv","w") do io
            for i in 1:size(bigMatTest_dyn)[3]
                writedlm(io,[vcat(bigMatTest_dyn[:,:,i][:,1],bigMatTest_dyn[:,:,i][:,2])])   
            end
        end
        open("$(savepath_CSVfiles)/bigLabelsTest_dyn.csv","w") do io
            for i in 1:size(bigLabels_Test_dyn)[1]              
                writedlm(io,[bigLabels_Test_dyn[i]])                     
            end
        end
    
    
    else 
        (bigMatTrain,bigLabel_Train,bigMatTest,bigLabel_Test,X) = create_virtual_Database(Param_Data)
        bigLabels_Train = create_bigMat_Labels_Tx(bigLabel_Train)
        bigLabels_Test = create_bigMat_Labels_Tx(bigLabel_Test)
        DatBinaryFiles.writeComplexBinary(X[:,1],"./TX1")
        if Param_Data.Augmentation_Value.augmentationType == "No_channel"
            savepath_CSVfiles = "/media/HDD/achillet/RF_Fingerprint/Database/RiFyFi_Database/CSV_Files/$(Param_Data.Modulation)/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)/$(Param_Data.E)_$(Param_Data.S)/$(Param_Data.E)_$(Param_Data.S)_$(Param_Data.C)_$(Param_Data.RFF)_$(Param_Data.nbSignals)_$(Param_Data.nameModel)"
        else 
            savepath_CSVfiles = "/media/HDD/achillet/RF_Fingerprint/Database/RiFyFi_Database/CSV_Files/$(Param_Data.Modulation)/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)/$(Param_Data.E)_$(Param_Data.S)/$(Param_Data.E)_$(Param_Data.S)_$(Param_Data.C)_$(Param_Data.RFF)_$(Param_Data.nbSignals)_$(Param_Data.nameModel)_$(Param_Data.Augmentation_Value.Channel)_$(Param_Data.Augmentation_Value.Channel_Test)_nbAugment_$(Param_Data.Augmentation_Value.nb_Augment)"
        end
        !ispath(savepath_CSVfiles) && mkpath(savepath_CSVfiles)    

        # Save in CSV files 
        open("$(savepath_CSVfiles)/bigMatTrain.csv","w") do io
            for i in 1:size(bigMatTrain)[3]
                writedlm(io,[vcat(bigMatTrain[:,:,i][:,1],bigMatTrain[:,:,i][:,2])])  
            end
        end

        open("$(savepath_CSVfiles)/bigMatTest.csv","w") do io
            for i in 1:size(bigMatTest)[3]
                writedlm(io,[vcat(bigMatTest[:,:,i][:,1],bigMatTest[:,:,i][:,2])])   
            end
        end

        open("$(savepath_CSVfiles)/bigLabelsTrain.csv","w") do io
            for i in 1:size(bigLabels_Train)[1]                 
                writedlm(io,[bigLabels_Train[i]])                          
            end
        end

        open("$(savepath_CSVfiles)/bigLabelsTest.csv","w") do io
            for i in 1:size(bigLabels_Test)[1]              
                writedlm(io,[bigLabels_Test[i]])                     
            end
        end 

    end
end





""" 
Create train and test dataset (here matrixes and not dataloader) for a configuration decribed by RiFyFi_VDG.Data_Synth structure.
(X_train,Y_train,X_test,Y_test,tupTrain.X) = create_X_and_Y(Param_Data::RiFyFi_VDG.Data_Synth)
"""
function create_virtual_Database(Param_Data)
    shuffle = false
    @info "Create virtual Tx"
    # ----------------------------------------------------------------
    # --- Create configuration with parameters and random impairments
    # ----------------------------------------------------------------
    if Param_Data.configuration == "nothing"
        @info "Create Tx with random impairments"
        savePath_config="Configurations/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)/$(Param_Data.E)_$(Param_Data.RFF)_$(Param_Data.name)"
        !ispath(savePath_config) && mkpath(savePath_config)
        # --- Create configuration based on mean value and interval parameter
        configurationTrain = "$(savePath_config)/ConfigTrain.json"
        configurationTest = "$(savePath_config)/ConfigTest_$(Param_Data.S).json"
        createConfiguration(configurationTrain,Param_Data.Chunksize,Param_Data.nbTx,Param_Data.nbSignals,Param_Data.C,Param_Data.seed_model,Param_Data.seed_data)
        createConfiguration(configurationTest,Param_Data.Chunksize,Param_Data.nbTx,Param_Data.nbSignals,Param_Data.C,Param_Data.seed_modelTest,Param_Data.seed_dataTest)
        # --- Generate Database random database correspond to the parameters and configuration
        tupTrain = generateDatabase(configurationTrain,Param_Data.E,Param_Data.S,Param_Data.C,Param_Data.RFF,Param_Data.name,Param_Data.Modulation,Int(Param_Data.nbSignals*Param_Data.pourcentTrain),Param_Data)
        tupTest = generateDatabase(configurationTest,Param_Data.E,Param_Data.S,Param_Data.C,Param_Data.RFF,Param_Data.name,Param_Data.Modulation,Param_Data.nbSignals-Int(Param_Data.nbSignals*Param_Data.pourcentTrain),Param_Data)
        # --- Save configuration of the Database with value of impairments
        saveScenario("$(savePath_config)/scenario.json",tupTrain.dict_out)
    # ---------------------------------------------------------------------------
    # --- Create configuration with fixe RFF impairments define in scenario file
    # ---------------------------------------------------------------------------
    elseif Param_Data.configuration == "scenario" 
        # --- Generate Database with fixe impairments scenario
        @info "Create Tx based on scenario $(Param_Data.E) $(Param_Data.RFF) $(Param_Data.name)" 
        savePath_config="Configurations/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)/$(Param_Data.E)_$(Param_Data.RFF)_$(Param_Data.name)"
        !ispath(savePath_config) && mkpath(savePath_config)
        # --- Create configuration based on parameter
        configurationTrain = "$(savePath_config)/ConfigTrain.json"
        configurationTest = "$(savePath_config)/ConfigTest_$(Param_Data.S).json"
        createConfiguration(configurationTrain,Param_Data.Chunksize,Param_Data.nbTx,Param_Data.nbSignals,Param_Data.C,Param_Data.seed_model,Param_Data.seed_data)
        createConfiguration(configurationTest,Param_Data.Chunksize,Param_Data.nbTx,Param_Data.nbSignals,Param_Data.C,Param_Data.seed_modelTest,Param_Data.seed_dataTest)
        # --- Create database based on scenarios impairments previously defined 
        s_d_mismatch,s_d_cfo,s_d_phaseNoise,s_d_nonLinearPA =  reloadScenario("$(savePath_config)/scenario.json",Param_Data.RFF)
        tupTrain = generateDatabase(configurationTrain,Param_Data.E,Param_Data.S,Param_Data.C,Param_Data.RFF,Param_Data.name,Param_Data.Modulation,Int(Param_Data.nbSignals*Param_Data.pourcentTrain),Param_Data;s_d_mismatch, s_d_cfo, s_d_phaseNoise, s_d_nonLinearPA)
        if ispath("$(savePath_config)/scenarioTest.json")
            s_d_mismatch,s_d_cfo,s_d_phaseNoise,s_d_nonLinearPA =  reloadScenario("$(savePath_config)/scenarioTest.json",Param_Data.RFF)
            @info "load Test "
        else 
            s_d_mismatch,s_d_cfo,s_d_phaseNoise,s_d_nonLinearPA =  reloadScenario("$(savePath_config)/scenario.json",Param_Data.RFF)
        end
        tupTest = generateDatabase(configurationTest,Param_Data.E,Param_Data.S,Param_Data.C,Param_Data.RFF,Param_Data.name,Param_Data.Modulation,Param_Data.nbSignals-Int(Param_Data.nbSignals*Param_Data.pourcentTrain),Param_Data;s_d_mismatch, s_d_cfo, s_d_phaseNoise, s_d_nonLinearPA)
    end 
    # --- Compute dimension of Matrices 
    nbChunks = Param_Data.nbSignals * Param_Data.nbTx 
    nbTrain = Int(round(Param_Data.pourcentTrain * nbChunks)) 
    nbTest = nbChunks - nbTrain 
    nbTrain_per_radio =Int(nbTrain/Param_Data.nbTx) 
    nbTest_per_radio =Int(nbTest/Param_Data.nbTx)
    X_trainTemp = tupTrain.bigMat
    X_testTemp = tupTest.bigMat
    # --- Creation of label for each transmitter  
    if Param_Data.name =="1VsAll"
        Y_trainTemp = zeros(2,nbTrain)
        Y_testTemp = zeros(2,nbTest)
        for iR =1 :1:2
            Y_trainTemp[iR,(Param_Data.nbTx-1)*((iR-1)*nbTrain_per_radio)+1:(Param_Data.nbTx-2+iR)*nbTrain_per_radio] .= 1
            Y_testTemp[iR,(Param_Data.nbTx-1)*((iR-1)*nbTest_per_radio)+1:(Param_Data.nbTx-2+iR)*nbTest_per_radio] .= 1
        end
    else 
        Y_trainTemp = zeros(Param_Data.nbTx,nbTrain)
        Y_testTemp = zeros(Param_Data.nbTx,nbTest)
        for iR =1 :1:Param_Data.nbTx
            Y_trainTemp[iR,(iR-1)*nbTrain_per_radio+1:iR*nbTrain_per_radio] .= 1
            Y_testTemp[iR,(iR-1)*nbTest_per_radio+1:iR*nbTest_per_radio] .= 1
        end
    end 
    # --------------------------------------------------
    # --- Add Channel / Augmented Data 
    # --------------------------------------------------
    if (Param_Data.Augmentation_Value.augmentationType=="augment" || 
        Param_Data.Augmentation_Value.augmentationType=="same_channel" || 
        Param_Data.Augmentation_Value.augmentationType=="1channelTest" ||
        Param_Data.Augmentation_Value.augmentationType=="Change_channel" )
        @info "Augmentation * " Param_Data.Augmentation_Value.nb_Augment
        if Param_Data.Augmentation_Value.augmentationType=="augment" 
            N = Int(Param_Data.nbSignals * Param_Data.pourcentTrain) # on génère toujours nbSignaux par canal 
            ( X_train,Y_train)=Augmentation.Add_diff_Channel_train_test(X_trainTemp,Y_trainTemp,N,Param_Data.Augmentation_Value.Channel,Param_Data.Chunksize,Param_Data.Augmentation_Value.nb_Augment,Param_Data.nbTx,Param_Data.Augmentation_Value.seed_channel,Param_Data.Augmentation_Value.burstSize)
            @info size(X_train)
            # TEST
            N = Param_Data.nbSignals - N 
            nbAugment_Test = 100
            (X_test,Y_test)=Augmentation.Add_diff_Channel_train_test(X_testTemp,Y_testTemp,N,Param_Data.Augmentation_Value.Channel_Test,Param_Data.Chunksize,nbAugment_Test,Param_Data.nbTx,Param_Data.Augmentation_Value.seed_channel_test,Param_Data.Augmentation_Value.burstSize)
        elseif Param_Data.Augmentation_Value.augmentationType=="same_channel"
            N = Int(Param_Data.nbSignals * Param_Data.pourcentTrain) # on génère toujours nbSignaux par canal 
            ( X_train,Y_train)=Augmentation.Add_diff_Channel_train_test(X_trainTemp,Y_trainTemp,N,Param_Data.Augmentation_Value.Channel,Param_Data.Chunksize,Param_Data.Augmentation_Value.nb_Augment,Param_Data.nbTx,Param_Data.Augmentation_Value.seed_channel,Param_Data.Augmentation_Value.burstSize)
            # TEST
            N = Param_Data.nbSignals - N 
            nbAugment_Test = Param_Data.Augmentation_Value.nb_Augment
            (X_test,Y_test)=Augmentation.Add_diff_Channel_train_test(X_testTemp,Y_testTemp,N,Param_Data.Augmentation_Value.Channel,Param_Data.Chunksize,nbAugment_Test,Param_Data.nbTx,Param_Data.Augmentation_Value.seed_channel,Param_Data.Augmentation_Value.burstSize)
        elseif Param_Data.Augmentation_Value.augmentationType=="Change_channel"
            N = Int(Param_Data.nbSignals * Param_Data.pourcentTrain) # on génère toujours nbSignaux par canal 
            ( X_train,Y_train)=Augmentation.Add_diff_Channel_train_test(X_trainTemp,Y_trainTemp,N,Param_Data.Augmentation_Value.Channel,Param_Data.Chunksize,Param_Data.Augmentation_Value.nb_Augment,Param_Data.nbTx,Param_Data.Augmentation_Value.seed_channel,Param_Data.Augmentation_Value.burstSize)
            # TEST
            N = Param_Data.nbSignals - N 
            nbAugment_Test = Param_Data.Augmentation_Value.nb_Augment
            (X_test,Y_test)=Augmentation.AddChannelTest(X_testTemp,Y_testTemp,N,Param_Data.Augmentation_Value.Channel,Param_Data.Chunksize,nbAugment_Test,Param_Data.nbTx,Param_Data.Augmentation_Value.seed_channel,Param_Data.Augmentation_Value.burstSize)
        end  
    else
        Param_Data.Augmentation_Value.nb_Augment = 1
        nbAugment_Test =1
        X_train =X_trainTemp
        X_test=X_testTemp
        Y_train=Y_trainTemp
        Y_test=Y_testTemp
    end
    # ----------------------------------------
    # --- Shuffle signals with labels 
    # ----------------------------------------
    if shuffle
        X_trainS = zeros(Float32,Param_Data.Chunksize,2,nbTrain*Param_Data.Augmentation_Value.nb_Augment)
        Y_trainS = zeros(Float32,Param_Data.nbTx,nbTrain*Param_Data.Augmentation_Value.nb_Augment)
        X_testS = zeros(Float32,Param_Data.Chunksize,2,nbTest*nbAugment_Test)
        Y_testS = zeros(Float32,Param_Data.nbTx,nbTest*nbAugment_Test)
        indexes = randperm(Int(size(X_train,3)))
        for i =1 :1 :size(X_train,3)
            X_trainS[:,:,(i)] = X_train[:,:,(indexes[i])] 
            Y_trainS[:,(i)] = Y_train[:,(indexes[i])]
        end 
        indexes = randperm(Int(size(X_test,3)))
        for i =1 :1 :size(X_test,3)
            X_testS[:,:,(i)] = X_test[:,:,(indexes[i])] 
            Y_testS[:,(i)]  = Y_test[:,(indexes[i])]
        end 
        X_train=X_trainS
        Y_train=Y_trainS
        X_test=X_testS
        Y_test=Y_testS
    end 
    # ----------------------------------------
    # --- Normalized data
    # ----------------------------------------
    if Param_Data.Normalisation
        (moy,std_val) = (0,0)
        (moy,std_val) = preProcessing!(X_train,nothing,nothing)
        (moy,std_val) = preProcessing!(X_test,moy,std_val)
    end 
    return (X_train,Y_train,X_test,Y_test,tupTrain.X)
end



""" 
Create train and test dataset (here matrixes and not dataloader) for a configuration decribed by RiFyFi_VDG.Data_Synth structure.
(X_train,Y_train,X_test,Y_test,tupTrain.X) = create_X_and_Y(Param_Data::RiFyFi_VDG.Data_Synth)
"""
function create_virtual_Database_dynCFO(Param_Data)
    
    shuffle = true
    @info "Create virtual Tx"
    # ----------------------------------------------------------------
    # --- Create configuration with parameters and random impairments
    # ----------------------------------------------------------------
    if Param_Data.configuration == "nothing"
        @info "ERROR" 
    # ---------------------------------------------------------------------------
    # --- Create configuration with fixe RFF impairments define in scenario file
    # ---------------------------------------------------------------------------
    elseif Param_Data.configuration == "scenario" 
        # --- Generate Database with fixe impairments scenario
        @info "Create Tx based on scenario $(Param_Data.E) $(Param_Data.RFF) $(Param_Data.name)" 
        savePath_config="Configurations/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)/$(Param_Data.E)_$(Param_Data.RFF)_$(Param_Data.name)"
        !ispath(savePath_config) && mkpath(savePath_config)
        # --- Create configuration based on parameter
        configurationTrain = "$(savePath_config)/ConfigTrain.json"
        configurationTest = "$(savePath_config)/ConfigTest_$(Param_Data.S).json"
        createConfiguration(configurationTrain,Param_Data.Chunksize,Param_Data.nbTx,Param_Data.nbSignals,Param_Data.C,Param_Data.seed_model,Param_Data.seed_data)
        createConfiguration(configurationTest,Param_Data.Chunksize,Param_Data.nbTx,Param_Data.nbSignals,Param_Data.C,Param_Data.seed_modelTest,Param_Data.seed_dataTest)
        # --- Create database based on scenarios impairments previously defined 
        s_d_mismatch,s_d_cfo,s_d_phaseNoise,s_d_nonLinearPA =  reloadScenario("$(savePath_config)/scenario.json",Param_Data.RFF)
        tupTrain = generateDatabase(configurationTrain,Param_Data.E,Param_Data.S,Param_Data.C,"all_impairments",Param_Data.name,Param_Data.Modulation,Int(Param_Data.nbSignals*Param_Data.pourcentTrain),Param_Data;s_d_mismatch, s_d_cfo, s_d_phaseNoise, s_d_nonLinearPA)
        tupTrain_dyn = generateDatabase(configurationTrain,Param_Data.E,Param_Data.S,Param_Data.C,Param_Data.RFF,Param_Data.name,Param_Data.Modulation,Int(Param_Data.nbSignals*Param_Data.pourcentTrain),Param_Data;s_d_mismatch, s_d_cfo, s_d_phaseNoise, s_d_nonLinearPA)

        if ispath("$(savePath_config)/scenarioTest.json")
            s_d_mismatch,s_d_cfo,s_d_phaseNoise,s_d_nonLinearPA =  reloadScenario("$(savePath_config)/scenarioTest.json",Param_Data.RFF)
            @info "load Test "
        else 
            s_d_mismatch,s_d_cfo,s_d_phaseNoise,s_d_nonLinearPA =  reloadScenario("$(savePath_config)/scenario.json",Param_Data.RFF)
        end
        tupTest = generateDatabase(configurationTest,Param_Data.E,Param_Data.S,Param_Data.C,"all_impairments",Param_Data.name,Param_Data.Modulation,Param_Data.nbSignals-Int(Param_Data.nbSignals*Param_Data.pourcentTrain),Param_Data;s_d_mismatch, s_d_cfo, s_d_phaseNoise, s_d_nonLinearPA)
        tupTest_dyn = generateDatabase(configurationTest,Param_Data.E,Param_Data.S,Param_Data.C,Param_Data.RFF,Param_Data.name,Param_Data.Modulation,Param_Data.nbSignals-Int(Param_Data.nbSignals*Param_Data.pourcentTrain),Param_Data;s_d_mismatch, s_d_cfo, s_d_phaseNoise, s_d_nonLinearPA)
    else 
        @info "error"
    end 
    # --- Compute dimension of Matrices 
    nbChunks = Param_Data.nbSignals * Param_Data.nbTx 
    nbTrain = Int(round(Param_Data.pourcentTrain * nbChunks)) 
    nbTest = nbChunks - nbTrain 
    nbTrain_per_radio =Int(nbTrain/Param_Data.nbTx) 
    nbTest_per_radio =Int(nbTest/Param_Data.nbTx)

    X_trainTemp = tupTrain.bigMat
    X_testTemp = tupTest.bigMat

    X_trainTemp_dyn = tupTrain_dyn.bigMat
    X_testTemp_dyn = tupTest_dyn.bigMat
    # --- Creation of label for each transmitter  

    Y_trainTemp = zeros(Param_Data.nbTx,nbTrain)
    Y_testTemp = zeros(Param_Data.nbTx,nbTest)

    Y_trainTemp_dyn = zeros(Param_Data.nbTx,nbTrain)
    Y_testTemp_dyn = zeros(Param_Data.nbTx,nbTest)
    for iR =1 :1:Param_Data.nbTx
        Y_trainTemp[iR,(iR-1)*nbTrain_per_radio+1:iR*nbTrain_per_radio] .= 1
        Y_testTemp[iR,(iR-1)*nbTest_per_radio+1:iR*nbTest_per_radio] .= 1
        Y_trainTemp_dyn[iR,(iR-1)*nbTrain_per_radio+1:iR*nbTrain_per_radio] .= 1
        Y_testTemp_dyn[iR,(iR-1)*nbTest_per_radio+1:iR*nbTest_per_radio] .= 1
    end


    Param_Data.Augmentation_Value.nb_Augment = 1
    nbAugment_Test =1
    X_train=X_trainTemp
    X_test=X_testTemp
    Y_train=Y_trainTemp
    Y_test=Y_testTemp

    X_train_dyn=X_trainTemp_dyn
    X_test_dyn=X_testTemp_dyn
    Y_train_dyn=Y_trainTemp_dyn
    Y_test_dyn=Y_testTemp_dyn
    
    # ----------------------------------------
    # --- Shuffle signals with labels 
    # ----------------------------------------
    if shuffle
        X_trainS = zeros(Float32,Param_Data.Chunksize,2,nbTrain*Param_Data.Augmentation_Value.nb_Augment)
        Y_trainS = zeros(Float32,Param_Data.nbTx,nbTrain*Param_Data.Augmentation_Value.nb_Augment)
        X_testS = zeros(Float32,Param_Data.Chunksize,2,nbTest*nbAugment_Test)
        Y_testS = zeros(Float32,Param_Data.nbTx,nbTest*nbAugment_Test)
        X_trainS_dyn = zeros(Float32,Param_Data.Chunksize,2,nbTrain*Param_Data.Augmentation_Value.nb_Augment)
        Y_trainS_dyn = zeros(Float32,Param_Data.nbTx,nbTrain*Param_Data.Augmentation_Value.nb_Augment)
        X_testS_dyn = zeros(Float32,Param_Data.Chunksize,2,nbTest*nbAugment_Test)
        Y_testS_dyn = zeros(Float32,Param_Data.nbTx,nbTest*nbAugment_Test)
        indexes = randperm(Int(size(X_train,3)))
        for i =1 :1 :size(X_train,3)
            X_trainS[:,:,(i)] = X_train[:,:,(indexes[i])] 
            Y_trainS[:,(i)] = Y_train[:,(indexes[i])]
            X_trainS_dyn[:,:,(i)] = X_train_dyn[:,:,(indexes[i])] 
            Y_trainS_dyn[:,(i)] = Y_train_dyn[:,(indexes[i])]
        end 
        indexes = randperm(Int(size(X_test,3)))
        for i =1 :1 :size(X_test,3)
            X_testS[:,:,(i)] = X_test[:,:,(indexes[i])] 
            Y_testS[:,(i)]  = Y_test[:,(indexes[i])]
            X_testS_dyn[:,:,(i)] = X_test_dyn[:,:,(indexes[i])] 
            Y_testS_dyn[:,(i)]  = Y_test_dyn[:,(indexes[i])]
        end 
        X_train = X_trainS
        Y_train = Y_trainS
        X_test = X_testS
        Y_test = Y_testS

        X_train_dyn = X_trainS_dyn
        Y_train_dyn = Y_trainS_dyn
        X_test_dyn = X_testS_dyn
        Y_test_dyn = Y_testS_dyn
    end 
    # ----------------------------------------
    # --- Normalized data
    # ----------------------------------------
   # if Param_Data.Normalisation
   #     (moy,std_val) = (0,0)
   #     (moy,std_val) = preProcessing!(X_train,nothing,nothing)
   #     (moy,std_val) = preProcessing!(X_test,moy,std_val)
   # end 
    return (X_train,Y_train,X_test,Y_test,X_train_dyn,Y_train_dyn,X_test_dyn,Y_test_dyn,tupTrain.X)
end



""" Function that load data from the CSV file for Synthetic database based on configuration described by 
RiFyFi_VDG.Data_Synth struct
"""
function loadCSV_Synthetic(Param_Data)

    nbChunks=Int(Param_Data.nbTx*Param_Data.nbSignals)
    nbTrain = Int(round(Param_Data.pourcentTrain*nbChunks))
    nbTest = nbChunks - nbTrain
    if Param_Data.Augmentation_Value.augmentationType == "No_channel"
       savepath_CSVfiles = "/media/HDD/achillet/RF_Fingerprint/Database/RiFyFi_Database/CSV_Files/$(Param_Data.Modulation)/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)/$(Param_Data.E)_$(Param_Data.S)/$(Param_Data.E)_$(Param_Data.S)_$(Param_Data.C)_$(Param_Data.RFF)_$(Param_Data.nbSignals)_$(Param_Data.name)"
    else  
        savepath_CSVfiles = "/media/HDD/achillet/RF_Fingerprint/Database/RiFyFi_Database/CSV_Files/$(Param_Data.Modulation)/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)/$(Param_Data.E)_$(Param_Data.S)/$(Param_Data.E)_$(Param_Data.S)_$(Param_Data.C)_$(Param_Data.RFF)_$(Param_Data.nbSignals)_$(Param_Data.nameModel)_$(Param_Data.Augmentation_Value.Channel)_$(Param_Data.Augmentation_Value.Channel_Test)_nbAugment_$(Param_Data.Augmentation_Value.nb_Augment)"

        nbTrain = nbTrain * Param_Data.Augmentation_Value.nb_Augment
        if Param_Data.Augmentation_Value.augmentationType == "1channelTest"  
            nbTest = nbTest * 1
        elseif Param_Data.Augmentation_Value.augmentationType == "same_channel"   ||
            Param_Data.Augmentation_Value.augmentationType == "Change_channel"
            nbTest = nbTest * 1
        elseif Param_Data.Augmentation_Value.augmentationType == "augment"
            nbTest = nbTest * 100
        end 
    end 
    # Labels 
    fileLabelTest= "$(savepath_CSVfiles)/bigLabelsTest.csv"
    Y_testTemp = Matrix(DataFrame(CSV.File(fileLabelTest;types=Int64,header=false)))
    fileLabelTrain= "$(savepath_CSVfiles)/bigLabelsTrain.csv"
    Y_trainTemp = Matrix(DataFrame(CSV.File(fileLabelTrain;types=Int64,header=false)))
    # Data 
    fileDataTest= "$(savepath_CSVfiles)/bigMatTest.csv"
    X_testTemp = Matrix(DataFrame(CSV.File(fileDataTest;types=Float32,header=false)))
    fileDataTrain= "$(savepath_CSVfiles)/bigMatTrain.csv"
    X_trainTemp = Matrix(DataFrame(CSV.File(fileDataTrain;types=Float32,header=false)))
    X_train = zeros(Float32, Param_Data.Chunksize,2,nbTrain)
    X_test = zeros(Float32, Param_Data.Chunksize,2,nbTest)
    Y_train = zeros(Param_Data.nbTx,nbTrain)
    Y_test = zeros(Param_Data.nbTx,nbTest)

    for i in 1:size(X_trainTemp)[1]  
        X_train[:,1,i]=X_trainTemp[i,1:Param_Data.Chunksize]
        X_train[:,2,i]=X_trainTemp[i,Param_Data.Chunksize+1:Param_Data.Chunksize+Param_Data.Chunksize]
    end 
    for i in 1:size(X_testTemp)[1]  
        X_test[:,1,i]=X_testTemp[i,1:Param_Data.Chunksize]
        X_test[:,2,i]=X_testTemp[i,Param_Data.Chunksize+1:Param_Data.Chunksize+Param_Data.Chunksize]
    end 
    for i in 1:size(Y_trainTemp)[1]  
        Y_train[Y_trainTemp[i]+1,i]=1
    end 
    for i in 1:size(Y_testTemp)[1]  
        Y_test[Y_testTemp[i]+1,i]=1
    end 
    return (X_train,Y_train,X_test,Y_test)
end 



""" Function that load data from the CSV file for Synthetic database based on configuration described by 
RiFyFi_VDG.Data_Synth struct
"""
function loadCSV_Synthetic_dynCFO(Param_Data)

    nbChunks=Int(Param_Data.nbTx * Param_Data.nbSignals)
    nbTrain = Int(round(Param_Data.pourcentTrain * nbChunks))
    nbTest = nbChunks - nbTrain
    if Param_Data.Augmentation_Value.augmentationType == "No_channel"
       savepath_CSVfiles = "/media/HDD/achillet/RF_Fingerprint/Database/RiFyFi_Database/CSV_Files/$(Param_Data.Modulation)/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)/$(Param_Data.E)_$(Param_Data.S)/$(Param_Data.E)_$(Param_Data.S)_$(Param_Data.C)_$(Param_Data.RFF)_$(Param_Data.nbSignals)_$(Param_Data.name)"
    else  
        savepath_CSVfiles = "/media/HDD/achillet/RF_Fingerprint/Database/RiFyFi_Database/CSV_Files/$(Param_Data.Modulation)/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)/$(Param_Data.E)_$(Param_Data.S)/$(Param_Data.E)_$(Param_Data.S)_$(Param_Data.C)_$(Param_Data.RFF)_$(Param_Data.nbSignals)_$(Param_Data.nameModel)_$(Param_Data.Augmentation_Value.Channel)_$(Param_Data.Augmentation_Value.Channel_Test)_nbAugment_$(Param_Data.Augmentation_Value.nb_Augment)"

        nbTrain = nbTrain * Param_Data.Augmentation_Value.nb_Augment
        if Param_Data.Augmentation_Value.augmentationType == "1channelTest"  
            nbTest = nbTest * 1
        elseif Param_Data.Augmentation_Value.augmentationType == "same_channel"   ||
            Param_Data.Augmentation_Value.augmentationType == "Change_channel"
            nbTest = nbTest * 1
        elseif Param_Data.Augmentation_Value.augmentationType == "augment"
            nbTest = nbTest * 100
        end 
    end 
    # Labels 
    fileLabelTest= "$(savepath_CSVfiles)/bigLabelsTest_dyn.csv"
    Y_testTemp = Matrix(DataFrame(CSV.File(fileLabelTest;types=Int64,header=false)))
    fileLabelTrain= "$(savepath_CSVfiles)/bigLabelsTrain_dyn.csv"
    Y_trainTemp = Matrix(DataFrame(CSV.File(fileLabelTrain;types=Int64,header=false)))
    # Data 
    fileDataTest = "$(savepath_CSVfiles)/bigMatTest_dyn.csv"
    X_testTemp = Matrix(DataFrame(CSV.File(fileDataTest;types=Float32,header=false)))
    fileDataTrain= "$(savepath_CSVfiles)/bigMatTrain_dyn.csv"
    X_trainTemp = Matrix(DataFrame(CSV.File(fileDataTrain;types=Float32,header=false)))
    X_train = zeros(Float32, Param_Data.Chunksize,2,nbTrain)
    X_test = zeros(Float32, Param_Data.Chunksize,2,nbTest)
    Y_train = zeros(Param_Data.nbTx,nbTrain)
    Y_test = zeros(Param_Data.nbTx,nbTest)

    for i in 1:size(X_trainTemp)[1]  
        X_train[:,1,i]=X_trainTemp[i,1:Param_Data.Chunksize]
        X_train[:,2,i]=X_trainTemp[i,Param_Data.Chunksize+1:Param_Data.Chunksize+Param_Data.Chunksize]
    end 
    for i in 1:size(X_testTemp)[1]  
        X_test[:,1,i]=X_testTemp[i,1:Param_Data.Chunksize]
        X_test[:,2,i]=X_testTemp[i,Param_Data.Chunksize+1:Param_Data.Chunksize+Param_Data.Chunksize]
    end 
    for i in 1:size(Y_trainTemp)[1]  
        Y_train[Y_trainTemp[i]+1,i]=1
    end 
    for i in 1:size(Y_testTemp)[1]  
        Y_test[Y_testTemp[i]+1,i]=1
    end 
    return (X_train,Y_train,X_test,Y_test)
end 





""" Generates a datbase based on the configuration stored in the JSON file `filename`
"""
function generateDatabase(filename::String,E::String,S::String,C::String,RFF::String,name::String,Modulation::String,nbSignaux::Int,Param_Data;kw...)
    dict = loadConfiguration(filename)
    return generateDatabase(dict,E,S,C,RFF,name,Modulation,nbSignaux,Param_Data;kw...)
end

""" Create a default constructor if no input structure is given. Otherwise, returns the structure
"""
function instantiateImpairments(f::Function,structure::Union{AbstractVector{T},Nothing},dict,r) where T
    if isnothing(structure)
        # structure is not given, set the default constructor 
        return f(dict,r)
    else 
        # We have a structure in a vector, use it 
        return structure[r]
    end
end


""" Dispatch when dict is composed with symbols 
""" 
function generateDatabase(dict::AbstractDict{Symbol,Any},E,S,C,RFF,name,Modulation,nbSignaux,Param_Data;kw...)
    ds = string_dict(dict)
    generateDatabase(ds,E,S,C,RFF,name,Modulation,nbSignaux,Param_Data;kw...)
end

""" Generate a vector of RF impairments structure, based on the filename. The output can be used as input of generateDatabase. The file used as input should habe been generated from saveScenario, whose output is from generateDatabase
"""
function reloadScenario(filename,RFF)
    
    dict = loadConfiguration(filename)
    nbR = length(dict)
    s_d_mismatch = Vector{RFImpairmentsModels.IQMismatch}(undef,nbR)
 
    s_d_cfo = Vector{RFImpairmentsModels.CFO}(undef,nbR)

    s_d_phaseNoise = Vector{RFImpairmentsModels.PhaseNoiseModel}(undef,nbR)
    if RFF == "PA_memory" || RFF == "all_impairments_memory"
        s_d_nonLinearPA = Vector{RiFyFi_VDG.Memory_PowerAmplifier}(undef,nbR)
    else 
        s_d_nonLinearPA = Vector{RFImpairmentsModels.PowerAmplifier}(undef,nbR)
    end 
        for (key,value) in dict 
        # We need to now what is the radio we have 
        ind = parse(Int,split(key,"radio")[2])
        # Fill the structure 
        # Mismatch
        _tmp = value["s_mismatch"]
        s_d_mismatch[ind] = initIQMismatch(_tmp["g"],_tmp["ϕ"])
        # CFO 
        _tmp = value["s_cfo"]
       
        s_d_cfo[ind] = initCFO(_tmp["f"],_tmp["fs"],_tmp["ϕ"])
        
        # Phase noise 
        _tmp = value["s_phaseNoise"]
        s_d_phaseNoise[ind] = initPhaseNoise(:Wiener,;σ2=_tmp["σ2"]) # FIXME Name in constrcutor (new version of RFImpairmentsModels)
        # Non linear PA
        _tmp = value["s_nonLinearPA"]
        if RFF == "PA_memory" || RFF == "all_impairments_memory"
            
            #s_d_nonLinearPA = nothing
            s_d_nonLinearPA[ind] = initNonLinearPAmemory(;symbol_dict(_tmp)...)

        else
        s_d_nonLinearPA[ind] = initNonLinearPA(:Saleh;symbol_dict(_tmp)...)
        end
    end
    return (;s_d_mismatch,s_d_cfo,s_d_phaseNoise,s_d_nonLinearPA)
end



""" Core signal generation, using a Dict for configuration. Each impairment config can be overwritted by custom config
"""
function generateDatabase(dict::AbstractDict{String,Any},E,S,C,RFF,name,Modulation,nbSignaux,Param_Data;
        s_d_mismatch::Union{Vector{RFImpairmentsModels.IQMismatch},Nothing} = nothing, # Force config for IQ Mismatch
        s_d_cfo::Union{Vector{RFImpairmentsModels.CFO},Nothing} = nothing ,             # Force config for CFO 
        s_d_phaseNoise::Union{Vector{RFImpairmentsModels.PhaseNoiseModel},Nothing}= nothing , # Force config for PN 
        s_d_nonLinearPA::Union{Vector{RFImpairmentsModels.PowerAmplifier},Nothing,Vector{RiFyFi_VDG.Memory_PowerAmplifier}}= nothing
    )
    # ----------------------------------------------------
    # --- Create containers 
    # ----------------------------------------------------
    chunk_size = @loadKey dict "chunk_size"     # Corresponds to input layer of CNN
    nb_chunks  = nbSignaux                      # Number of chunks per radio
    nb_radios  = @loadKey dict "nb_radios"      # Number of radios
    seed_models = @loadKey dict "seed_models" 
    seed_data = @loadKey dict "seed_data" 
    tx_policy = @loadKey dict "tx_policy"  "random"
    Fingerprint_policy = @loadKey dict "Fingerprint_policy" "random"
    max_burst_size = @loadKey dict "max_burst_size"
    # ----------------------------------------------------
    # --- Fill matrix with data 
    # ---------------------------------------------------- 
    # --- First associate OFDM symbols with number of IQ samples
    if Modulation == "OFDM"
        tmp,_,_ = tx(1)
        @info "OFDM"
    elseif Modulation == "SC"
        tmp = singleCarrierTx(1)
        @info " Single Carrier Modulation "
    elseif Modulation == "LoRa"
        SF = 7
        BW =1 
        Fs = 1
        (tmp,_,_,_,_) = Gen_symb_SF(SF,1,BW,Fs,1)
        @info " LoRa Modulation "
    else 
        @info "ERREUR"
    end 
    size_symb = length(tmp)
    @infiltrate
    # We split in independent burst of duration max_burst_size
    nb_bursts = nb_chunks ÷ max_burst_size +1
    nb_symb_per_bursts = max_burst_size * chunk_size ÷ size_symb +1
    # We fill big matrix step by step: each burst create a mat_chunck
    cnt = 0
    # --- Deduce final big matrixes, taking into accound potential rounding due to bursts
    bigMat      = zeros(Float32,chunk_size,2,nb_chunks*nb_radios)
    bigLabels   = zeros(nb_radios,nb_chunks*nb_radios)
    dict_out    = Dict()
    X=zeros(ComplexF32,size_symb*nb_symb_per_bursts*nb_bursts,nb_radios)
    # --- Iterative generation 
#=
    X_brut = zeros(Float32,4,263040,2)
    savepath = "CSV_Files/sans_4_256/E1_S2/E1_S2_C1_all_impairments_1000_"
    brut1= "$(savepath)/X_brut1.csv"
    X_brut[1,:,:] = Matrix(DataFrame(CSV.File(brut1;types=Float64,header=false)))
=#
    for r ∈ 1 : nb_radios
        #= ## Used To permut the transmitter impairment
        if r==1 
            permut=3
            r=permut
        end    
        =#
        # ----------------------------------------------------
        # --- Instantiate Impairments models  
        # ---------------------------------------------------- 
        if E=="E1"
            Random.seed!(seed_models  ) # sans le + r on génère touours la même empreinte
            s_mismatch = instantiateImpairments(setup_IQMismatch,s_d_mismatch,dict,1)
            s_cfo       = instantiateImpairments(setup_CFO,s_d_cfo,dict,1)
            s_phaseNoise    = instantiateImpairments(setup_phaseNoise,s_d_phaseNoise,dict,1)
        else
            Random.seed!(seed_models  + r) # sans le + r on génère touours la même empreinte
            s_mismatch = instantiateImpairments(setup_IQMismatch,s_d_mismatch,dict,r)
            s_cfo       = instantiateImpairments(setup_CFO,s_d_cfo,dict,r)
            s_phaseNoise    = instantiateImpairments(setup_phaseNoise,s_d_phaseNoise,dict,r)
        end

        if E=="E3"
            s_phaseNoise.seed= -1 # seed relachée
        elseif E=="E1" 
            s_phaseNoise.seed= 1  # seed fixe pour une radio 
        else 
            s_phaseNoise.seed= r  # seed fixe pour une radio
        end

        # --- Instantiate PA
        if RFF=="PA_memory" || RFF=="all_impairments_memory"
            radio=r
            if r==1
                radio =1
            elseif r==2
                radio =4  #62
            elseif r==3
                radio =5  # 75
            elseif r==4
                radio =6 # 35
            elseif r==5
                radio =75  # 75
            elseif r==6
                radio =35 # 35
            end 

            if E=="E1"
            s_nonLinearPA        = instantiateImpairments(setup_nonLinearPA_memory,s_d_nonLinearPA,dict,1)
            else 
            s_nonLinearPA        = instantiateImpairments(setup_nonLinearPA_memory,s_d_nonLinearPA,dict,radio)
            end
        else
            if E=="E1"
                s_nonLinearPA        = instantiateImpairments(setup_nonLinearPA,s_d_nonLinearPA,dict,1)
            else 
                s_nonLinearPA    = instantiateImpairments(setup_nonLinearPA,s_d_nonLinearPA,dict,r)
            end 
        end 

        for k ∈ 1 : 1 : nb_bursts 
            if Fingerprint_policy == "random"
                # A new seed for Tx signal 
                if E=="E1" 
                    Random.seed!(seed_models ) # même empreinte pour chaque burst et chaque radio
                elseif E=="E2"
                    Random.seed!(seed_models + r ) # même empreinte pour chaque burst 
                else 
                    Random.seed!(seed_models + (r-1)* nb_bursts + r + k )
                end 
            end
            
            # We should start to random phase for PN 
            randomize_phaseNoise!(s_phaseNoise)
            # Random Phase Noise
            Random.seed!(seed_models + (r-1)* nb_bursts + r + k )   
            s_noise = setup_awgn(dict,r,k)

            # We calculate a new fading factor FIXME => Related to SNR ? 
         #   s_power = setup_rx_power(dict,r,s_noise,k)
            
            # ----------------------------------------------------
            # --- Create the pure tx signal
            # ----------------------------------------------------
            if tx_policy == "random"
                if S=="S1"
                    Random.seed!(seed_data) # toujours les mêmes données pour toutes les radios et identique à chaque burst
                elseif S=="S2"
                Random.seed!(seed_data + r) # toujours les mêmes données pour toutes les radios mais changent à chaque burst
                else
                    Random.seed!(seed_data + (r-1)* nb_bursts + r + k )
                end 
            end 
            if Modulation == "OFDM"
                (x,_,_) = tx(nb_symb_per_bursts)   
                x=x.*10 
            elseif Modulation == "SC"
                x = singleCarrierTx(nb_symb_per_bursts)
                x=x.*10 
            elseif Modulation == "LoRa"
                SF = 7
                BW =1 
                Fs = 1
                (x,_,_,_,_) = Gen_symb_SF(SF,nb_symb_per_bursts,BW,Fs,1)
                x=x.*10 
            else 
                @info "ERREUR"
            end 
            nbIQperBurst = size_symb*nb_symb_per_bursts
            X[(k-1)*nbIQperBurst+1:k*nbIQperBurst,r] = x
            # -------------------------------------------
            # -- Tx impairments avec seed Fingerprint
            # -------------------------------------------
            if Fingerprint_policy == "random"
                if E=="E1"
                    Random.seed!(seed_models )
                elseif E=="E2"
                    Random.seed!(seed_models + r )
                else 
                    Random.seed!(seed_models + (r-1)* nb_bursts + r + k )
                end 
            end
            # -------------------------------------------
            # --- To study the individual impact of impairments
            # -------------------------------------------
            if RFF == "imbalance"
                # IQ Mismatch 
                addIQMismatch!(x,s_mismatch)
            elseif RFF == "PN"
                # Add PN 
                addPhaseNoise!(x,s_phaseNoise) # Imperfection Dynamique 
            elseif RFF == "cfo"
                # Add CFO 
                RFImpairmentsModels.addCFO!(x,s_cfo)
            elseif RFF == "dynamic_cfo"
                add_dynamic_CFO!(x,s_cfo,Param_Data.dyn_value)
            elseif RFF == "PA_memory"
                # Non linear PA memory
                if E == "E1"
                    x = addNonLinearPA_memory(x,s_nonLinearPA,1)
                else
                    y = addNonLinearPA_memory(x,s_nonLinearPA,r)
                    PlotingAMAM_Saleh(x[1:2000],y[1:2000],RFF,r,E,S,C,name)
                    x = y
                end 
            elseif RFF == "PA"
                # Non linear PA Saleh
                    if E == "E1"
                        x = addNonLinearPA(x,s_nonLinearPA)
                    else
                        y = addNonLinearPA(x,s_nonLinearPA)
                    #   PlotingAMAM_Saleh(x[1:2000],y[1:2000],RFF,r,E,S,C,name)
                        x = y
                    end 
            # -------------------------------------------
            # --- To study the Conglomerate impact of impairments
            # -------------------------------------------
            elseif RFF == "all_impairments_dynamic_cfo"
                @info "dynamic CFO"
                # All Impairments 
                addIQMismatch!(x,s_mismatch)
                addPhaseNoise!(x,s_phaseNoise) # Imperfection Dynamique 
                add_dynamic_CFO!(x,s_cfo,Param_Data.dyn_value)
                if RFF=="all_impairments_memory"
                  x=addNonLinearPA_memory(x,s_nonLinearPA,r)
                else 
                    x=addNonLinearPA(x,s_nonLinearPA)
                end 
            else # All Impairments 
                addIQMismatch!(x,s_mismatch)
                addPhaseNoise!(x,s_phaseNoise) # Imperfection Dynamique 
                @info "stat" 
              #  if nbSignaux == 900
              RFImpairmentsModels.addCFO!(x,s_cfo)
              #  else 
              #      add_dynamic_CFO!(x,s_cfo)
              #  end 
                
                if RFF=="all_impairments_memory"
                    x=addNonLinearPA_memory(x,s_nonLinearPA,r)
                else 
                    x=addNonLinearPA(x,s_nonLinearPA)
                end 
            end 
            # ----------------------------------------------------
            # -- Noise model 
            # ----------------------------------------------------
            if C=="C1" # No noise
                out=x
            elseif C=="C3" # Add random noise between 10dB and 30dB
                Random.seed!(seed_models + (r-1)* nb_bursts + r + k )
                s_noise = Int(rand(1:3)*10)
                out,_ = addNoise(x,s_noise)
            else # C=="C2" || C=="C2_10dB" || C=="C2_20dB" || C=="C2_0dB"
                Random.seed!(seed_models + (r-1)* nb_bursts + r + k )
                out,_ = addNoise(x,s_noise)
            end 
        
    
            if k== nb_bursts
                rest= nb_chunks-(nb_bursts-1)*max_burst_size
                out=x[1:rest*chunk_size]
                fill_label!(bigLabels,rest,r,cnt)
                cnt = fill_data!(bigMat,out,chunk_size,rest,cnt)
            else
            # ----------------------------------------------------
            # Filling matrixes
            # --- Fill in matrixes
            # We reshape out in real/ imag 
            fill_label!(bigLabels,max_burst_size,r,cnt)
            cnt = fill_data!(bigMat,out,chunk_size,max_burst_size,cnt)
            end 
        end
        # --- Config for the radio 
        tup_radio = (; s_mismatch,s_cfo,s_phaseNoise,s_nonLinearPA)
        push!(dict_out,"radio$r"=>tup_radio)
    end
    
    return (;bigMat,bigLabels,dict,dict_out,X)

end


"""
fill bigMat (chunk_size x 2 x X) from out and split in real and imag part. 
Start bigMat fill @cnt and returns the updated pos of cnt (for next call)
"""
function fill_data!(bigMat,out,chunk_size,small_chunk,cnt)
    for n ∈ 1 : small_chunk
        for r ∈ 1 : chunk_size
            bigMat[r,1,cnt + n] = real(out[ (n-1)*chunk_size + r])
            bigMat[r,2,cnt + n] = imag(out[ (n-1)*chunk_size + r])
        end
    end
    return cnt + small_chunk
end
function fill_label!(bigLabels,small_chunk,r,cnt)
    for n ∈ 1 : small_chunk
        bigLabels[r,cnt + n] = 1 
    end
    return cnt + small_chunk
end


""" Create a random variable around 10^-digitPower with numDigits number
For instance a myRand(0,2) = 0.xy
myRand(-2,3) = 0.00xxx
myRand(nothing,y) = 0
"""
function myRand(digitPower,numDigits)
    if isnothing(digitPower)
        return 0
    else 
        randn(Float64) > 0 ? sign = 1 : sign = -1
        # left round is to avoid 0.xxyy00000001
        val = round(ceil(rand(Float64);digits=numDigits) * 10.0^(digitPower);digits= abs(digitPower)+numDigits )
        return sign * val
    end
end

""" Create a configuration and stores in filename. Code should be copy and paste for full customisation
"""
function createConfiguration(filename,ChunkSize,nbRadioTx,nbSignaux,C,seed_model,seed_data)
    tmp_σ = -7
    if C == "C2_0dB" 
        noize =0
    elseif C== "C2_5dB" 
        noize =5
    elseif C== "C2_10dB" 
        noize =10
    elseif C== "C2_15dB" 
        noize =15
    elseif C== "C2_20dB"
        noize=20
    elseif C== "C2_25dB" 
        noize =25
    elseif C == "C2"
        noize =30
    else 
        noize = 30
    end 
    # --- Radio mode 
    dict = OrderedDict(
                       :nb_radios => nbRadioTx,
                       :chunk_size  => ChunkSize,           # Slice of data feed to NN 
                       :nb_chunks  => nbSignaux,        # Per radios
                       :always_sync => false,         # Ensure each chunck is sync with OFDM symbol
                       :seed_data => seed_data,# 123456,          # To have a seed to generate radio parameters  # seed_data
                       :seed_models=> seed_model, #234567,
                       :max_burst_size => 64,#100,       # Number of chunks for one generated burst 
                       # --- Tx 
                       :physical_layer => "hadoc",
                       :tx_policy => "random",       # Transmitted signal policy: Random : each radio tranmsit different signal. Consistent :> each radio transmit sames burst
                       # --- Frequency mode 
                       :carrier_freq    => 2400e6,
                       :sampling_rate   => 4e6,
                       # --- Impairments ,
                       :with_iq_mismatch => true ,
                       :with_cfo         => true,
                       :with_phase_noise => true ,
                       :with_nonLinearPA => true ,
                       :with_channel     => false ,
                       :with_awgn        => true ,
                       :with_rx_power    => false,
                       # --- Range of parameters 
                       # One model per radio 
                       :nonLinearPA_models   => [:Saleh,:Saleh,:Saleh,:Saleh],
                       :nonLinearPA_random_range => (-2,3),#(-1,3),#(-2,3),    # config (-2,3) / config2(-1,3)    # (x,y) calls myRand
                       :nonLinearPA_base_saleh => [2.1587,1.1517,4.0033,9.1040],
                       # IQ Mismatch 
                       :base_g              => 1.5,   # Base for gain, in dB
                       :base_ϕ              => 3*π/180,   # Base for phase, in rad
                       :iq_random_range_gain => (-1,2), 
                       :iq_random_range_phase => (-3,2), # passer de -1 à -3
                       # CFO 
                       :base_cfo        => 300,   # In Hz 
                       :cfo_range   => (1,2),
                       # Phase noise 
                       :phase_noise_model => :Wiener,
                       :phase_noise_base_σ => 10.0^(tmp_σ),
                       :phase_noise_range_σ => (tmp_σ - 1,2) ,
                       # AWGN 
                       :awgn_snr_base => noize, # passage de 20 à 30
                       :awgn_range    =>(0,0),# (-1,2),# changé 0 en -1 # 0,2 leads to -9.9 to 9.9 =>=>> Divided by 2 in setup_awgn
                       # Channel model 
                       :channel_model => [:multipath,:multipath,:multipath,:multipath],
                       :channel_nb_taps => 5,
                       :channel_τ_m => 15 ,
                       :channel_fix_channel => false, # Channel is constant per radio
                       :channel_seed => 687,
                      )
    # --- Save
    open(filename,"w") do io 
        JSON.print(io,dict,1)
    end
    return string_dict(dict)
end


""" Save all the specific scenario parameters from generateDatabase. It takes the output of the function (field `dict_out`) as an input 
"""
function saveScenario(filename,dict_out)
    open(filename,"w") do io 
        JSON.print(io,dict_out,1)
    end
end

""" Apply a convolutional channel describe in vector s_channel to x. Mutate x with the modified data
"""
function applyChannel!(x::AbstractVector{Complex{T}},s_channel::Union{AbstractVector{T},AbstractVector{Complex{T}}})  where {T<:Real}
    if s_channel == [Complex{T}(1)]
        # Nothing to do, unitary channel. branch is done to speed up 
    else 
        if iseven(length(s_channel))
            # If channel is even, we have an issue to remove tails to we add a 0 
            @warn "Channel impulse response is Even, padd with 0 to have class 1 filter"
            push!(h,0)
        end
        # --- Apply Channel model
        # Convolution and tail truncation model
        δ = (length(s_channel)-1)÷2
        sigChan = DSP.conv(x,s_channel)[1+δ:end-δ] #FIXME Not sure if it works with channel. Need a +1 that we don't need if channel = 1...
        @show length(x), length(sigChan)
        x .= sigChan
    end
end

""" Load the configuration stored in the JSON file `filename`
"""
function loadConfiguration(filename)
    dict = open(filename,"r") do io 
        JSON.parse(io)
    end
    return dict 
end






end
