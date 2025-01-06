Base.@kwdef mutable struct Data_Exp 
    run="1"
    Test="1"
    noise=nothing
    permutation=false
    shuffle=true 
    Type_of_sig = "Preamble"
    File_Path::String = "/media/redinblack/ANR_RedInBlack/rffExperiment/"
    name::String = "Exp"
    nbTx::Int = 5
    nbSignals::Int = 10000
    Chunksize::Int = 256
    features::String= "IQsamples"
    Normalisation::Bool = true
    pourcentTrain::Float64 =0.9
    Augmentation_Value::Data_Augmented = Data_Augmented() 
end 


function Data_Exp(; kwargs...)
    if haskey(kwargs, :run)
        run = kwargs[:run]
    else
        run="1"
    end
    if haskey(kwargs, :Test)
        Test = kwargs[:Test]
    else
        Test="1"
    end
    if haskey(kwargs, :noise)
        noise = kwargs[:noise]
    else
        noise=nothing
    end
    if haskey(kwargs, :permutation)
        permutation = kwargs[:permutation]
    else
        permutation=false
    end
    if haskey(kwargs, :shuffle)
        shuffle = kwargs[:shuffle]
    else
        shuffle=true
    end
    if haskey(kwargs, :Type_of_sig)
        Type_of_sig = kwargs[:Type_of_sig]
    else
        Type_of_sig="Preamble"
    end
    if haskey(kwargs, :File_Path)
        File_Path = kwargs[:File_Path]
    else
        File_Path="/media/redinblack/ANR_RedInBlack/rffExperiment/"
    end

if haskey(kwargs, :name)
    name = kwargs[:name]
else
    name="Exp"
end

if haskey(kwargs, :nbTx)
    nbTx = kwargs[:nbTx]
else 
    nbTx = 5
end

if haskey(kwargs, :nbSignals)
    nbSignals = kwargs[:nbSignals]
else 
    nbSignals = 100000
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
    return Data_Exp(run,Test,noise,permutation,shuffle,Type_of_sig,File_Path,name,nbTx,nbSignals,Chunksize,features,Normalisation,pourcentTrain,Augmentation_Value)
end