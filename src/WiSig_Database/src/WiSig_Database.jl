module WiSig_Database
using Random
using Statistics
using DelimitedFiles
using CSV
using DataFrames
using Infiltrator
include("../../Augmentation/src/Augmentation.jl")
using .Augmentation

include("utils.jl")

export setWiSigcsv
export create_X_Y_WiSig
export loadCSV_WiSig
include("Struct_data.jl")
export Data_WiSig


"""
Create train and test dataset (here matrixes and not dataloader) for a given distance in feet. If the keyword "all" is used the train and test sets uses all distances.
(X_train,X_test,Y_train,Y_test) = create_X_and_Y(filename,txs,rxs,days,equalized,ChunkSize,pourcentTrain,augmentationType,chanel) 
"""
function create_X_Y_WiSig(Param_Data) 
    nbChunks = Param_Data.nbSignals *Param_Data.nbTx * size(Param_Data.days,1) # size(tupTrain.bigMat,3) + size(tupTest.bigMat,3)
    nbTrain = Int(round(Param_Data.pourcentTrain*nbChunks)) 
    nbTest = nbChunks - nbTrain # size(tupTest.bigMat,3)

    shuffle=true
    ChunkSize = Param_Data.Chunksize
    # Loading Data and create Matrix for train and test 
    (X_train,X_test,Y_train,Y_test)= Pickle_to_matrix(Param_Data)
    # ----------------------------------------
    # --- Shuffle signals with labels 
    # ----------------------------------------
    if Param_Data.Augmentation_Value.augmentationType=="augment" 
        nbAugment = Param_Data.Augmentation_Value.nb_Augment
        @info nbAugment
        channel =Param_Data.Augmentation_Value.Channel
        seed_channel = Param_Data.Augmentation_Value.seed_channel
        burstSize = Param_Data.Augmentation_Value.burstSize
        N = Int(Param_Data.nbSignals * Param_Data.pourcentTrain) # on génère toujours nbSignaux par canal 
        ( X_trainTemp,Y_trainTemp)=Augmentation.Add_diff_Channel_train_test(X_train,Y_train,N,channel,Param_Data.Chunksize,Param_Data.Augmentation_Value.nb_Augment,Param_Data.nbTx,seed_channel,burstSize)
        @info size(X_train)
        # TEST
        X_train = X_trainTemp
        Y_train = Y_trainTemp
    end 
    
    if shuffle
        X_trainS = zeros(Float32,ChunkSize,2,nbTrain*Param_Data.Augmentation_Value.nb_Augment)
        Y_trainS = zeros(Float32,Param_Data.nbTx,nbTrain*Param_Data.Augmentation_Value.nb_Augment)
        X_testS = zeros(Float32,ChunkSize,2,nbTest)
        Y_testS = zeros(Float32,Param_Data.nbTx,nbTest)
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
        (moy,std_val) = preProcessing!(X_train,nothing,nothing)
        (moy,std_val) = preProcessing!(X_test,moy,std_val)
    end 
    if Param_Data.features == "Module_angle"
        X_train_Mod = zeros(Float32,ChunkSize,2,nbTrain)#*nb_Augment)
        X_test_Mod = zeros(Float32,ChunkSize,2,nbTest)#*nbAugment_Test)
        X_train_Mod[:,1,:] .= abs2.(X_train[:,1,:]+ im *X_train[:,2,:])
        X_train_Mod[:,2,:] .= angle.(X_train[:,1,:]+ im *X_train[:,2,:])
        X_train = X_train_Mod
        X_test = X_test_Mod
    end 
    return (X_train,Y_train,X_test,Y_test)
end



""" 
Apply normalisation to input data 
"""
function preProcessing!(X,moy,std_val)
    if isnothing(moy)
        moy_reel = mean(X[:,1,:])
        moy_ima = mean(X[:,2,:])
    else 
        moy_reel = real(moy)
        moy_ima = imag(moy)
    end 
    if isnothing(std_val)
        std_val_reel = std(X[:,1,:])
        std_val_ima = std(X[:,2,:])
    else 
        std_val_reel = real(std_val)
        std_val_ima = imag(std_val)
    end
    X[:,1,:] .= (X[:,1,:] .- moy_reel)./std_val_reel
    X[:,2,:] .= (X[:,2,:] .- moy_ima )./std_val_ima
    return (moy_reel+1im*moy_ima,std_val_reel+1im*std_val_ima)
end


""" Transforme la matrice des labels en un vecteur d'indice 0-(NbRadios-1) """
function create_bigMat_Labels_Tx(new_bigLabels)
    bigLabels   = zeros(Int,size(new_bigLabels)[2])
    for i in 1:size(new_bigLabels)[2]
        for j in 1:size(new_bigLabels)[1]
            if new_bigLabels[j,i] == 1
                bigLabels[i] = j-1;
            end
        end
    end

    return bigLabels
end


""" Function that load data from the CSV file for Synthetic database
"""
function loadCSV_WiSig(Param_Data)
  
    nbChunks=Int(Param_Data.nbTx*Param_Data.nbSignals * size(Param_Data.days,1))
    nbTrain = Int(round(Param_Data.pourcentTrain*nbChunks))
    nbTest = nbChunks - nbTrain

    suffix =  "$(Param_Data.txs)_$(Param_Data.rxs)_$(Param_Data.days)_$(Param_Data.equalized)_$(Param_Data.pourcentTrain)_$(Param_Data.Chunksize)"
    savepath = "./CSV_Files/WiSig/$(suffix)"    
    # Labels 
    fileLabelTest= "$(savepath)/bigLabelsTest_$suffix.csv"
    Y_testTemp = Matrix(DataFrame(CSV.File(fileLabelTest;types=Int64,header=false)))
    fileLabelTrain= "$(savepath)/bigLabelsTrain_$suffix.csv"
    Y_trainTemp = Matrix(DataFrame(CSV.File(fileLabelTrain;types=Int64,header=false)))
    # Data 
    fileDataTest= "$(savepath)/bigMatTest_$suffix.csv"
    X_testTemp = Matrix(DataFrame(CSV.File(fileDataTest;types=Float32,header=false)))
    fileDataTrain= "$(savepath)/bigMatTrain_$suffix.csv"
    X_trainTemp = Matrix(DataFrame(CSV.File(fileDataTrain;types=Float32,header=false)))
    X_train = zeros(Float32, Param_Data.Chunksize,2,nbTrain*Param_Data.Augmentation_Value.nb_Augment)
    X_test = zeros(Float32, Param_Data.Chunksize,2,nbTest)
    Y_train = zeros(Param_Data.nbTx,nbTrain*Param_Data.Augmentation_Value.nb_Augment)
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







""" 
Transform a vector of label (with each index a number of radio) into a matrix of 0 with a 1 per column associated to the radio index
"""
function transformLabels(Y,nbRadios)
    Z = zeros(Int,nbRadios,length(Y))
    for n in eachindex(Y)
        Z[1+Int(Y[n]),n] = 1 
    end 
    return Z
end



""" Function that create 4 csv files based on pkl file(s)
CSV files: 
- Training Data
- Test Data 
- Training Labels
- Test Labels """
function setWiSigcsv(Param_Data)
    
   
    (bigMatTrain,bigLabels_Train,bigMatTest,bigLabels_Test) = create_X_Y_WiSig(Param_Data)
    
    bigLabels_Train = create_bigMat_Labels_Tx(bigLabels_Train)
    bigLabels_Test = create_bigMat_Labels_Tx(bigLabels_Test)
    savepath = "CSV_Files/WiSig/$(Param_Data.txs)_$(Param_Data.rxs)_$(Param_Data.days)_$(Param_Data.equalized)_$(Param_Data.pourcentTrain)_$(Param_Data.Chunksize)"
    !ispath(savepath) && mkpath(savepath)    
    open("$(savepath)/bigMatTrain_$(Param_Data.txs)_$(Param_Data.rxs)_$(Param_Data.days)_$(Param_Data.equalized)_$(Param_Data.pourcentTrain)_$(Param_Data.Chunksize).csv","w") do io
        for i in 1:size(bigMatTrain)[3]
            writedlm(io,[vcat(bigMatTrain[:,:,i][:,1],bigMatTrain[:,:,i][:,2])])   
        end
    end

    open("$(savepath)/bigMatTest_$(Param_Data.txs)_$(Param_Data.rxs)_$(Param_Data.days)_$(Param_Data.equalized)_$(Param_Data.pourcentTrain)_$(Param_Data.Chunksize).csv","w") do io
        for i in 1:size(bigMatTest)[3]
            writedlm(io,[vcat(bigMatTest[:,:,i][:,1],bigMatTest[:,:,i][:,2])])    
        end
    end
    open("$(savepath)/bigLabelsTrain_$(Param_Data.txs)_$(Param_Data.rxs)_$(Param_Data.days)_$(Param_Data.equalized)_$(Param_Data.pourcentTrain)_$(Param_Data.Chunksize).csv","w") do io
        for i in 1:size(bigLabels_Train)[1]                 
            writedlm(io,[bigLabels_Train[i]])                           
        end
    end
    open("$(savepath)/bigLabelsTest_$(Param_Data.txs)_$(Param_Data.rxs)_$(Param_Data.days)_$(Param_Data.equalized)_$(Param_Data.pourcentTrain)_$(Param_Data.Chunksize).csv","w") do io
        for i in 1:size(bigLabels_Test)[1]              
            writedlm(io,[bigLabels_Test[i]])                          
        end
    end
end


end 