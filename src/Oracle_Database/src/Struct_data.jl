Base.@kwdef mutable struct Data_Oracle 
    distance ="2ft"
    run="1"
    File_name::String = "/media/HDD/achillet/RF_Fingerprint/Database/KRI-16Devices-RawData/"
    name::String = "Oracle"
    nbTx::Int = 6
    nbSignals::Int = 1000
    Chunksize::Int = 256
    features::String= "IQsamples"
    Normalisation::Bool = true
    pourcentTrain::Float64 =0.9
    Augmentation_Value::Data_Augmented = Data_Augmented() 
end 


function Data_Oracle(; kwargs...)
    if haskey(kwargs, :distance)
        distance = kwargs[:distance]
    else
        distance="2ft"
    end
    if haskey(kwargs, :run)
        run = kwargs[:run]
    else
        run="1"
    end
    if haskey(kwargs, :File_name)
        File_name = kwargs[:File_name]
    else
        File_name="/media/HDD/achillet/RF_Fingerprint/Database/KRI-16Devices-RawData/"
    end

if haskey(kwargs, :name)
    name = kwargs[:name]
else
    name="Oracle"
end

if haskey(kwargs, :nbTx)
    nbTx = kwargs[:nbTx]
else 
    nbTx = 6
end

if haskey(kwargs, :nbSignals)
    nbSignals = kwargs[:nbSignals]
else 
    nbSignals = 1000
end

if haskey(kwargs, :Chunksize)
    Chunksize = kwargs[:Chunksize]
else 
    Chunksize = 256
end

if haskey(kwargs, :features)
    features = kwargs[:features]
else 
    features = "IQsamples"
end
if haskey(kwargs, :Normalisation)
    Normalisation = kwargs[:Normalisation]
else 
    Normalisation = true
end
if haskey(kwargs, :pourcentTrain)
    pourcentTrain = kwargs[:pourcentTrain]
else 
    pourcentTrain = 0.9
end

if haskey(kwargs, :Augmentation_Value)
    Augmentation_Value = kwargs[:Augmentation_Value]
else 
    Augmentation_Value = Data_Augmented() 
end
    return Data_Oracle(distance,run,File_name,name,nbTx,nbSignals,Chunksize,features,Normalisation,pourcentTrain,Augmentation_Value)
end