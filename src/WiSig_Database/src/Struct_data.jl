
Base.@kwdef mutable struct Data_WiSig
    File_name::String = "DataBases/WiSig/ManySig/pkl_wifi_ManySig/ManySig.pkl"
    name::String = "WiSig"
    nbTx::Int = 6
    nbSignals::Int = 1000
    Chunksize::Int = 256
    features::String= "IQsamples"
    txs = 1:6
    rxs = 1
    days = 1
    equalized= 2
    Normalisation::Bool = true
    pourcentTrain::Float64 =0.9
    Augmentation_Value::Data_Augmented = Augmentation.Data_Augmented() 
end



function Data_WiSig(; kwargs...)
    if haskey(kwargs, :File_name)
        File_name = kwargs[:File_name]
    else
        File_name="DataBases/WiSig/ManySig/pkl_wifi_ManySig/ManySig.pkl"
    end

    if haskey(kwargs, :name)
        name = kwargs[:name]
    else
        name="WiSig"
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


    if haskey(kwargs, :txs)
        txs = kwargs[:txs]
    else 
        txs = 1:6
    end
    if haskey(kwargs, :rxs)
        rxs = kwargs[:rxs]
    else 
        rxs = 1
    end
    if haskey(kwargs, :days)
        days = kwargs[:days]
    else 
        days = 1
    end
    if haskey(kwargs, :equalized)
        equalized = kwargs[:equalized]
    else 
        equalized =2
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
    return Data_WiSig(File_name,name, nbTx, nbSignals, Chunksize,features, txs,rxs,days,equalized,Normalisation,pourcentTrain,Augmentation_Value)
end

