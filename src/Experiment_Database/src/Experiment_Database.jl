module Experiment_Database
using Preferences
using Random
using Statistics
using Infiltrator
using DelimitedFiles
using CSV
using DataFrames
using FFTW
include("Utils.jl")
using .Utils
include("../../Augmentation/src/Augmentation.jl")
using .Augmentation
include("Struct_data.jl")
include("DatBinaryFiles.jl")

export Data_Exp
export setExpcsv

export loadCSV_Exp





function create_X_Y_Exp(Param_Data)
Param_Data.shuffle=false
    if Param_Data.run== "1"
        Run ="2024-06-03-17h/"
    elseif Param_Data.run== "1bis"
        Run ="2024-07-04-14h/" 
    elseif Param_Data.run== "2"
        Run="2024-06-12-17h/"
    elseif Param_Data.run== "3"
        Run="2024-06-04-16h/"
    elseif Param_Data.run== "4"
        Run="2024-07-05-12h/"
    elseif Param_Data.run== "4bis"
        Run="2024-07-09-12h/"
    elseif Param_Data.run== "5"
        Run="2024-07-04-16h/"
    elseif Param_Data.run== "6"
        Run ="2024-06-06-12h/"
    end 
   

List_files = readdir("$(Param_Data.File_Path)$(Run)$(Param_Data.Type_of_sig)/DatFile_Cut")
name_Tx = []
nbTrain = Int(Param_Data.nbSignals*Param_Data.pourcentTrain)
nbTest = Param_Data.nbSignals - nbTrain
cntTrain = 0
cntTest  = 0
if Param_Data.nbTx==4
    List_files=List_files[1],List_files[3],List_files[4],List_files[5]
end 

if Param_Data.permutation == true
    permutation = randperm(Param_Data.nbSignals)
    trainSet = permutation[ 1 : nbTrain]
    testSet  = permutation[ 1 + nbTrain : end]
else 
    if Param_Data.Type_of_sig=="Payload"
        trainSet = (1:1:nbTrain)
        testSet  = (300000-nbTest+1:1:300000)
    else 
        trainSet = (1:1:nbTrain)
        testSet  = (60000-nbTest+1:1:60000)
    end 
  #  testSet  = (nbTrain+1:1:nbTest+nbTrain)
 #=   for i = 1 : 1: nbTrain
        push!(trainSet,i)
    end
    if Param_Data.Test =="1"
        for i = nbTrain+1 : 1: nbTest+nbTrain
            push!(testSet,i)
        end 
    else 
        for i = 125000-nbTest+1 : 1: 125000
            push!(testSet,i)
        end 
    end =#
end 
Param_Data.nbTx = Int(size(List_files,1))
X_trainTemp = zeros(Float32,Param_Data.Chunksize,2,nbTrain  * Param_Data.nbTx) 
Y_trainTemp = zeros(Float32,Param_Data.nbTx,nbTrain  * Param_Data.nbTx)
X_testTemp  = zeros(Float32,Param_Data.Chunksize,2,nbTest * Param_Data.nbTx) 
Y_testTemp  = zeros(Float32,Param_Data.nbTx,nbTest * Param_Data.nbTx)

cntLabel = 1


for i =1 :1 : Param_Data.nbTx
    
    filename = "$(Param_Data.File_Path)$(Run)$(Param_Data.Type_of_sig)/DatFile_Cut/$(List_files[i])"
    push!(name_Tx,split(List_files[i],"_")[1])
    signal = load_Exp_Signal(filename,Param_Data.nbSignals,Param_Data.Chunksize)

    if Param_Data.noise !== nothing
        signal = Augmentation.Add_Noise_Rxside(signal,Param_Data.noise)
    end 
    for n in eachindex(trainSet)
        cPos = trainSet[n]
        X_trainTemp[:,1,cntTrain + n] = real( signal[ (cPos-1)*Param_Data.Chunksize .+ (1:Param_Data.Chunksize)] )
        X_trainTemp[:,2,cntTrain + n] = imag( signal[ (cPos-1)*Param_Data.Chunksize .+ (1:Param_Data.Chunksize)] )
        Y_trainTemp[cntLabel ,cntTrain + n] = 1
    end
    for n in eachindex(testSet)
        cPos = testSet[n]
        X_testTemp[:,1,cntTest + n] = real( signal[ (cPos-1)*Param_Data.Chunksize .+ (1:Param_Data.Chunksize)]  )
        X_testTemp[:,2,cntTest + n] = imag( signal[ (cPos-1)*Param_Data.Chunksize .+ (1:Param_Data.Chunksize)]  )
        Y_testTemp[cntLabel,cntTest+ n] = 1
    end
#=
     (moy,std_val) = (nothing,nothing)
    (P_Train,_,_)=preProcessing!(X_train[:,:,cntTrain+1:cntTrain+nbTrain],moy,std_val)
    X_train[:,:,cntTrain+1:cntTrain+nbTrain]=P_Train[:,:,:]
    (P_Test,_,_)=preProcessing!(X_test[:,:,cntTest+1:cntTest+nbTest],moy,std_val)
    X_test[:,:,cntTest+1:cntTest+nbTest]=P_Test[:,:,:]
=#
    cntTrain += nbTrain
    cntTest  += nbTest
    cntLabel += 1
end 
#=
if Param_Data.augmentationType=="augment" || augmentationType=="same_channel" || augmentationType=="1channelTest"
    nbAugment = Param_Data.Augmentation_Value.nb_Augment
    @info nbAugment
    channel =Param_Data.Augmentation_Value.Channel
    channel_Test =Param_Data.Augmentation_Value.Channel_Test
    seed_channel = Param_Data.Augmentation_Value.seed_channel
    seed_channel_test = Param_Data.Augmentation_Value.seed_channel_test
    burstSize = Param_Data.Augmentation_Value.burstSize
    if augmentationType=="augment" 
        N = Int(nbSignaux * pourcentTrain) # on génère toujours nbSignaux par canal 
        ( X_train,Y_train)=Augmentation.Add_diff_Channel_train_test(X_trainTemp,Y_trainTemp,N,channel,ChunkSize,nbAugment,nbRadioTx,seed_channel,burstSize)
        @info size(X_train)
        # TEST
        N = nbSignaux - N 
        nbAugment_Test = 100
        (X_test,Y_test)=Augmentation.Add_diff_Channel_train_test(X_testTemp,Y_testTemp,N,channel_Test,ChunkSize,nbAugment_Test,nbRadioTx,seed_channel_test,burstSize)
    elseif augmentationType=="same_channel"
        N = Int(nbSignaux * pourcentTrain) # on génère toujours nbSignaux par canal 
        ( X_train,Y_train)=Augmentation.Add_diff_Channel_train_test(X_trainTemp,Y_trainTemp,N,channel,ChunkSize,nbAugment,nbRadioTx,seed_channel,burstSize)
        # TEST
        N = nbSignaux - N 
        nbAugment_Test = nbAugment
        seed_channel_test = seed_channel
        (X_test,Y_test)=Augmentation.Add_diff_Channel_train_test(X_testTemp,Y_testTemp,N,channel,ChunkSize,nbAugment_Test,nbRadioTx,seed_channel_test,burstSize)
    end 
    else
        =#
    nbAugment = 1
    X_train =X_trainTemp
    X_test=X_testTemp
    Y_train=Y_trainTemp
    Y_test=Y_testTemp
    nbAugment_Test =1
#end
    if Param_Data.shuffle
        
        X_trainS = zeros(Float32,Param_Data.Chunksize,2,nbTrain*Param_Data.nbTx)#)*Param_Data.Augmentation_Value.nb_Augment)
        Y_trainS = zeros(Float32,Param_Data.nbTx,nbTrain*Param_Data.nbTx)#*Param_Data.Augmentation_Value.nb_Augment)
        X_testS = zeros(Float32,Param_Data.Chunksize,2,nbTest*Param_Data.nbTx)#*nbAugment_Test)
        Y_testS = zeros(Float32,Param_Data.nbTx,nbTest*Param_Data.nbTx)#*nbAugment_Test)
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
 
 
    (moy,std_val) = (nothing,nothing)
    
    preProcessing!(X_train,moy,std_val)
    (moy,std_val) = preProcessing!(X_test,moy,std_val)
    

return (X_train,Y_train,X_test,Y_test,moy,std_val)
end 

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
    return (X,moy_reel+1im*moy_ima,std_val_reel+1im*std_val_ima)
end



function loadCSV_Exp(Param_Data)
    nbChunks=Int(Param_Data.nbTx*Param_Data.nbSignals )
    nbTrain = Int(round(Param_Data.pourcentTrain*nbChunks))
    nbTest = nbChunks - nbTrain
    #if augmentationType == "No_channel"
    suffix =  "$(Param_Data.Type_of_sig)_$(Param_Data.Chunksize)"
    if Param_Data.permutation == true
    savepath = "./CSV_Files/Experiment/Run$(Param_Data.run)_Test$(Param_Data.Test)_permut_$(Param_Data.nbTx)_$(Param_Data.nbSignals)"    
    else 
        if Param_Data.noise == nothing
            savepath = "./CSV_Files/Experiment/Run$(Param_Data.run)_Test$(Param_Data.Test)_$(Param_Data.nbTx)_$(Param_Data.nbSignals)"    
        else 
            savepath = "./CSV_Files/Experiment/Run$(Param_Data.run)_Test$(Param_Data.Test)_$(Param_Data.nbTx)_$(Param_Data.nbSignals)_$(Param_Data.noise)"    
        end 
    end 
    #=else  
        channel = Param_Data.Augmentation_Value.Channel
        channel_Test = Param_Data.Augmentation_Value.Channel_Test
        nbAugment = Param_Data.Augmentation_Value.nb_Augment
        savepath = "./CSV_Files/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)/$(Param_Data.E)_$(Param_Data.S)/$(Param_Data.E)_$(Param_Data.S)_$(Param_Data.C)_$(Param_Data.RFF)_$(Param_Data.nbSignals)_$(Param_Data.nameModel)_$(Param_Data.Augmentation_Value.Channel)_$(Param_Data.Augmentation_Value.Channel_Test)_nbAugment_$(Param_Data.Augmentation_Value.nb_Augment)"
        nbTrain = nbTrain * nbAugment
        if augmentationType == "1channelTest"  
            nbTest = nbTest * 1
        elseif augmentationType == "same_channel"  
            nbTest = nbTest * 1
        else augmentationType == "augment"
            nbTest = nbTest * 100
        end 
    end
    =#
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

function setExpcsv(Param_Data)

    (bigMatTrain,bigLabels_Train,bigMatTest,bigLabels_Test) = create_X_Y_Exp(Param_Data)
    bigLabels_Train = create_bigMat_Labels_Tx(bigLabels_Train)
    bigLabels_Test = create_bigMat_Labels_Tx(bigLabels_Test)
    if Param_Data.permutation == true
        savepath = "./CSV_Files/Experiment/Run$(Param_Data.run)_Test$(Param_Data.Test)_permut_$(Param_Data.nbTx)_$(Param_Data.nbSignals)"    
    else 
        if Param_Data.noise == nothing
        savepath = "./CSV_Files/Experiment/Run$(Param_Data.run)_Test$(Param_Data.Test)_$(Param_Data.nbTx)_$(Param_Data.nbSignals)"    
        else 
        savepath = "./CSV_Files/Experiment/Run$(Param_Data.run)_Test$(Param_Data.Test)_$(Param_Data.nbTx)_$(Param_Data.nbSignals)_$(Param_Data.noise)"    
        end 
    end 
    !ispath(savepath) && mkpath(savepath)    
    open("$(savepath)/bigMatTrain_$(Param_Data.Type_of_sig)_$(Param_Data.Chunksize).csv","w") do io
        for i in 1:size(bigMatTrain)[3]
            writedlm(io,[vcat(bigMatTrain[:,:,i][:,1],bigMatTrain[:,:,i][:,2])])   
        end
    end

    open("$(savepath)/bigMatTest_$(Param_Data.Type_of_sig)_$(Param_Data.Chunksize).csv","w") do io
        for i in 1:size(bigMatTest)[3]
            writedlm(io,[vcat(bigMatTest[:,:,i][:,1],bigMatTest[:,:,i][:,2])])    
        end
    end

    open("$(savepath)/bigLabelsTrain_$(Param_Data.Type_of_sig)_$(Param_Data.Chunksize).csv","w") do io
        for i in 1:size(bigLabels_Train)[1]                 
            writedlm(io,[bigLabels_Train[i]])                           
        end
    end

    open("$(savepath)/bigLabelsTest_$(Param_Data.Type_of_sig)_$(Param_Data.Chunksize).csv","w") do io
        for i in 1:size(bigLabels_Test)[1]              
            writedlm(io,[bigLabels_Test[i]])                          
        end
    end
end 



function load_Exp_Signal(filename,nbSignals,Chunksize)
    Data_Vector= DatBinaryFiles.readComplexBinary(filename)
   # Cutted_Data_Vector = Data_Vector[1000001:end-1000000]
  #  Cutted_Data_Vector_good_size = Data_Vector[1:nbSignals*Chunksize]
    return Data_Vector #Cutted_Data_Vector
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


end