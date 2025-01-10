module Oracle_Database
using Preferences
using Random
using Statistics
using Infiltrator
using DelimitedFiles
using CSV
using DataFrames
include("../../Augmentation/src/Augmentation.jl")
using .Augmentation
include("Struct_data.jl")

export create_X_Y_Oracle
export Data_Oracle
export  getDistances,
        getRadioIdent,
        loadSignal,
        getFilenames,
        loadDistance

export setOraclecsv
export loadCSV_Oracle
""" 
Create train and test dataset (here matrixes and not dataloader) for a given distance in feet. If the keyword "all" is used the train and test sets uses all distances.
(X_train,Y_train,X_test,Y_test) = create_X_and_Y(Param_Data)
"""
function create_X_Y_Oracle(Param_Data) 
    shuffle =true
    # ----------------------------------------------------
    # --- Creating distance vector 
    # ---------------------------------------------------- 
    if Param_Data.distance == "all"  || Param_Data.distance == "all_ft"
        distanceVect = getDistances() 
    else 
        distanceVect = Param_Data.distance
    end
    nV = length(distanceVect)
    # ----------------------------------------------------
    # --- Config 
    # ---------------------------------------------------- 
    # --- Constant parameters 
    nbSeq          = Param_Data.nbSignals * nV #60_000 * nV# Number of sequence per radio 
    nbSeqD         = nbSeq ÷ nV # Number of sequence per radio per distance
    nbSeq          = nbSeqD * nV
    nbRadio        = Param_Data.nbTx #6#16 # Number of radio to be identified 
    trainTestRatio = Param_Data.pourcentTrain # 10% for test 
    # ----------------------------------------------------
    # --- Train and test sets
    # ---------------------------------------------------- 
    # --- Define train and test partitionning 
    nbTrain = Int(nbSeqD * trainTestRatio)
    nbTest  = nbSeqD - nbTrain
    permutation = randperm(nbSeqD)
    trainSet = permutation[ 1 : nbTrain]
    testSet  = permutation[ 1 + nbTrain : end]
    cntTrain = 0
    cntTest  = 0
    X_train = zeros(Float32,Param_Data.Chunksize,2,nbTrain * nV * nbRadio) 
    Y_train = zeros(Float32,nbRadio,nbTrain * nV * nbRadio)
    X_test  = zeros(Float32,Param_Data.Chunksize,2,nbTest * nV * nbRadio) 
    Y_test  = zeros(Float32,nbRadio,nbTest * nV * nbRadio)
    # ----------------------------------------------------
    # --- Folding matrixes
    # ---------------------------------------------------- 
    for feet_distance in distanceVect
        @info "preparing $feet_distance"
        name_file = getFilenames(feet_distance,Param_Data.run) 
        cntLabel = 1 # Restart label 
        for k = 1:1:nbRadio
            filename = name_file[k]
            # --- Load signal 
            signal = loadSignal(filename)
            # --- Train dataset 
            for n in eachindex(trainSet)
                cPos = trainSet[n]
                X_train[:,1,cntTrain + n] = real( signal[ (cPos-1)*Param_Data.Chunksize .+ (1:Param_Data.Chunksize)] )
                X_train[:,2,cntTrain + n] = imag( signal[ (cPos-1)*Param_Data.Chunksize .+ (1:Param_Data.Chunksize)] )
                Y_train[cntLabel ,cntTrain + n] = 1
            end
            for n in eachindex(testSet)
                cPos = testSet[n]
                X_test[:,1,cntTest + n] = real( signal[ (cPos-1)*Param_Data.Chunksize .+ (1:Param_Data.Chunksize)]  )
                X_test[:,2,cntTest + n] = imag( signal[ (cPos-1)*Param_Data.Chunksize .+ (1:Param_Data.Chunksize)]  )
                Y_test[cntLabel,cntTest+ n] = 1
            end
            # --- Position in matrixes for next radio 
            cntTrain += nbTrain
            cntTest  += nbTest
            cntLabel += 1
        end
        if shuffle
            X_trainS = zeros(Float32,Param_Data.Chunksize,2,nbTrain*Param_Data.nbTx*nV)#)*Param_Data.Augmentation_Value.nb_Augment)
            Y_trainS = zeros(Float32,Param_Data.nbTx,nbTrain*Param_Data.nbTx*nV)#*Param_Data.Augmentation_Value.nb_Augment)
            X_testS = zeros(Float32,Param_Data.Chunksize,2,nbTest*Param_Data.nbTx*nV)#*nbAugment_Test)
            Y_testS = zeros(Float32,Param_Data.nbTx,nbTest*Param_Data.nbTx*nV)#*nbAugment_Test)
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
    end
    # --- Normalisation 
    (moy,std_val) = (nothing,nothing)
    preProcessing!(X_train,moy,std_val)
    (moy,std_val) = preProcessing!(X_test,moy,std_val)
    return (X_train,Y_train,X_test,Y_test,moy,std_val)
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



# ----------------------------------------------------
# --- Database path definition 
# ---------------------------------------------------- 
# We use Preferences for store and load ORACLE database path
function set_database_path(dbPath)
    # Loading at global scope ?
    # Preferences.set_preferences!(Preferences.get_uuid(@__MODULE__),"databasePath" => dbPath, export_prefs=true)
    #@set_preferences!("databasePath"=>dbPath)
    databasePath = "/media/HDD/achillet/RF_Fingerprint/Database/KRI-16Devices-RawData/"
 
    @info "ORACLE database path is now at $dbPath.\n Reload Julia session to take the new path into account"
end


const PATH_DATABASE = "/media/HDD/achillet/RF_Fingerprint/Database/KRI-16Devices-RawData/"
# --- Check it exists 
function __init__()
    # --- Check the folder exists
    if !isdir(PATH_DATABASE)
        @warn "Path for the ORACLE DATABASE is not found. It means that the signal will probably not be usable for the learning processings.\n Use OracleRFFingerprint_Database.set_database_path(PATH) to set the appropriate path (with PATH pointing to the ORACLE database end should ending with .../KRI-16Devices-RawData/) and reload Julia\n"
    end
end

# ----------------------------------------------------
# --- Methods definition
# ---------------------------------------------------- 
"""
Returns the complex signal associated to the data in `nom_fichier`
The data is not post processed.
sigId = loadSignal(filename)
- nom_fichier  : Name of the database [String]
- sigId : Complex signal with IQ samples [Vector{Complex{Float64}]
"""
function loadSignal(filename::String) 
    # --- Open file 
    f= open(filename, "r")
    k = read(f)
    taille = length(k)
    Nb_Ite = taille÷8;
    currWord = zeros(UInt8,8)

    # buffer = zeros(ComplexF32,Nb_Ite÷2)
    buffer = Vector{ComplexF32}(undef,Nb_Ite÷2)
    cnt = 1
    @inbounds @simd for i in 1 : 1 : Nb_Ite
        for j = 1 : 8 
            currWord[j] = k[8*(i-1)+j]
        end
        Nv = reinterpret(Float64,currWord)
        cV = Float32(Nv[1])
        if (i%2 == 1)
            buffer[cnt] = cV
        else
            buffer[cnt] += 1im*cV
            cnt += 1
        end
    end
    return buffer
end

function loadSignal2(filename)
    # --- Final output 
    nbSeg = filesize(filename) ÷ 16
    # --- Open the file 
    f= open(filename, "r")
    # --- Pre-alloc output (as F64)
    out = Vector{Complex{Float64}}(undef,nbSeg)
    # --- Fill the array
    read!(f,out)
    # --- Returns as Float32
    return ComplexF32.(out)
end



""" 
Loads all the signals from a given distance and 
 radioMatrix = loadDistance("2feet")
returns a matrix of size nSamples x nRadio where 
nSamples is the samples recorded per radio 
nRadio is the number of radio.
Each column corresponds to a different radio
"""
function loadDistance(feet_distance)
    # --- Getting the filenames 
    nM = getFilenames(feet_distance)
    # --- Deduce number of radios 
    nbRadio = length(nM)
    # --- Get number of samples per radio 
    nbSeg = filesize(nM[1]) ÷ 16
    radioMatrix = zeros(ComplexF32,nbSeg,nbRadio)
    #  Iterate on file
    for (m,f) in enumerate(nM) 
        # --- Get the signal from matrix 
        currSig = loadSignal2(f)
        # --- Populate the column of the matrix 
        radioMatrix[:,m] .= currSig
    end
    return radioMatrix
end

""" 
Returns a Vector of String containing all radio identifiers 
"""
function getRadioIdent()
    return id_radio = [
                "3123D7B", "3123D7D", "3123D7E", "3123D52",
                "3123D54", "3123D58", "3123D64", "3123D65",
                "3123D70", "3123D76", "3123D78", "3123D79", 
                "3123D80", "3123D89", "3123EFE", "3124E4A"]
end 


""" 
Returns an array of string with all the filename of the file with the IQ samples for a given distance 
filenames = getFilenames(feet_distance)
- feet_distance : Desired distance as string ("2ft")
- filenames : Array of string with all the files related to the distance 
""" 
function getFilenames(feet_distance,run)
    id_radio = getRadioIdent()
       # --- Instantiate all radio names 
    name_file = String[]
    for id in id_radio
        push!(name_file, PATH_DATABASE * feet_distance * "/WiFi_air_X310_" * id * "_" * feet_distance *"_run$(run).sigmf-data")
    end
    return name_file
end


""" 
Returns an array with all the supported distance, as strings (like "2ft","16ft",...)
"""
function getDistances()
    return ["2ft","8ft","14ft","20ft","26ft","32ft","32ft","38ft","44ft","50ft","56ft","62ft"]
end



""" Function that create 4 csv files based on pkl file(s)
CSV files: 
- Training Data
- Test Data 
- Training Labels
- Test Labels """
function setOraclecsv(Param_Data)
    
    (bigMatTrain,bigLabels_Train,bigMatTest,bigLabels_Test) = create_X_Y_Oracle(Param_Data)

    bigLabels_Train = create_bigMat_Labels_Tx(bigLabels_Train)
    bigLabels_Test = create_bigMat_Labels_Tx(bigLabels_Test)
    savepath = "CSV_Files/Oracle/$(Param_Data.distance)_$(Param_Data.nbTx)_$(Param_Data.nbSignals)"
    !ispath(savepath) && mkpath(savepath)    
    open("$(savepath)/bigMatTrain_$(Param_Data.distance)_$(Param_Data.Chunksize).csv","w") do io
        for i in 1:size(bigMatTrain)[3]
            writedlm(io,[vcat(bigMatTrain[:,:,i][:,1],bigMatTrain[:,:,i][:,2])])   
        end
    end

    open("$(savepath)/bigMatTest_$(Param_Data.distance)_$(Param_Data.Chunksize).csv","w") do io
        for i in 1:size(bigMatTest)[3]
            writedlm(io,[vcat(bigMatTest[:,:,i][:,1],bigMatTest[:,:,i][:,2])])    
        end
    end

    open("$(savepath)/bigLabelsTrain_$(Param_Data.distance)_$(Param_Data.Chunksize).csv","w") do io
        for i in 1:size(bigLabels_Train)[1]                 
            writedlm(io,[bigLabels_Train[i]])                           
        end
    end

    open("$(savepath)/bigLabelsTest_$(Param_Data.distance)_$(Param_Data.Chunksize).csv","w") do io
        for i in 1:size(bigLabels_Test)[1]              
            writedlm(io,[bigLabels_Test[i]])                          
        end
    end
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


function loadCSV_Oracle(Param_Data)
    nbChunks=Int(Param_Data.nbTx*Param_Data.nbSignals * size(Param_Data.distance,1))
    nbTrain = Int(round(Param_Data.pourcentTrain*nbChunks))
    nbTest = nbChunks - nbTrain
    suffix =  "$(Param_Data.distance)_$(Param_Data.Chunksize)"
    savepath = "./CSV_Files/Oracle/$(Param_Data.distance)_$(Param_Data.nbTx)_$(Param_Data.nbSignals)"    
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


end 