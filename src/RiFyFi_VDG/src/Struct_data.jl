Base.@kwdef mutable struct Data_Synth
    name::String = ""
    nameModel::String = ""
    nbTx::Int = 4
    nbSignals::Int = 12000
    Chunksize::Int = 256
    features::String= "IQsamples"
    S::String = "S1"
    E::String = "E2"
    C::String = "C2_20dB"
    RFF::String = "all_impairments"
    Normalisation::Bool = true
    pourcentTrain::Float64 =0.9
    configuration::String  = "non"
    seed_data::Int = 1234
    seed_model::Int = 2345
    seed_dataTest::Int = 1234
    seed_modelTest::Int = 2345
    Augmentation_Value = Data_Augmented_construct() 
end


function Data_Synth_construct(; kwargs...)
    if haskey(kwargs, :name)
        name = kwargs[:name]
    else
        name=""
    end

    if haskey(kwargs, :nameModel)
        nameModel = kwargs[:nameModel]
    else
        nameModel=""
    end

    if haskey(kwargs, :nbTx)
        nbTx = kwargs[:nbTx]
    else 
        nbTx = 4
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

    if haskey(kwargs, :S)
        S = kwargs[:S]
    else 
        S = "S1"
    end

    if haskey(kwargs, :E)
        E = kwargs[:E]
    else 
        E = "E2"
    end
    if haskey(kwargs, :C)
        C = kwargs[:C]
    else 
        C = "C2_20dB"
    end
    if haskey(kwargs, :RFF)
        RFF = kwargs[:RFF]
    else 
        RFF = "all_impairments"
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
    if haskey(kwargs, :configuration)
        configuration = kwargs[:configuration]
    else 
        configuration = "non"
    end

    if haskey(kwargs, :seed_data)
        seed_data = kwargs[:seed_data]
    else 
        seed_data = 1234
    end

    if haskey(kwargs, :seed_model)
        seed_model = kwargs[:seed_model]
    else 
        seed_model = 2345
    end

    if haskey(kwargs, :seed_dataTest)
        seed_dataTest = kwargs[:seed_dataTest]
    elseif S == "S1" || S== "S2"
        seed_dataTest = seed_data
    else 
        seed_dataTest = 9999246912
    end

    if haskey(kwargs, :seed_modelTest)
        seed_modelTest = kwargs[:seed_modelTest]
    elseif E == "E1" || E == "E2"
        seed_modelTest = seed_model 
    else 
        seed_modelTest = 15987654321
    end 

    if haskey(kwargs, :Augmentation_Value)
        Augmentation_Value = kwargs[:Augmentation_Value]
    else 
        Augmentation_Value = Data_Augmented_construct() 
    end
return Data_Synth(name,nameModel,nbTx,nbSignals,Chunksize,features,S,E,C,RFF,Normalisation,pourcentTrain,configuration,seed_data,seed_model,seed_dataTest,seed_modelTest,Augmentation_Value)
end

#=

Base.@kwdef mutable struct Data_WiSig
    File_name::String = "../My_WiSig_ManySig/pkl_wifi_ManySig/ManySig.pkl"
    name::String = "WiSig"
    nbTx::Int = 6
    nbSignals::Int = 1000
    Chunksize::Int = 256
    features::String= "IQsamples"
    txs = 1:6
    rxs = 1
    days = 1
    equalized= 1
    Normalisation::Bool = true
    pourcentTrain::Float64 =0.9
    Augmentation_Value::Data_Augmented = Data_Augmented() 
end



function Data_WiSig(; kwargs...)
    if haskey(kwargs, :File_name)
        File_name = kwargs[:File_name]
    else
        File_name="../My_WiSig_ManySig/pkl_wifi_ManySig/ManySig.pkl"
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
        equalized =1
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

=#  
