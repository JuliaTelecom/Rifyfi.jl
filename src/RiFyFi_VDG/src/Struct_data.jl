Base.@kwdef mutable struct Data_Synth
    name::String = ""  # Name use to save the configuration
    nameModel::String = ""   # Name use to save the configuration
    nbTx::Int = 4 # Nb transmitters in the database 
    nbSignals::Int = 12000 # Nb signals per transmitter for Train + test
    Chunksize::Int = 256 # Size of packet
    features::String= "IQsamples" # In this version always use "IQsamples" 
    S::String = "S1" # S described the type of signal trame use in database (S1 preamble, S2 MAC address, S3 Payload)
    E::String = "E3" # E3 always to have a dynamic Phase noise impairments and different impairments for each transmitter
    C::String = "C2_20dB" # C described the level of noise 
    RFF::String = "all_impairments" # define the impairments which are activate 
    Normalisation::Bool = true # Normalise dataset
    pourcentTrain::Float64 =0.9 # Define the proportion of the Train /Test sets
    configuration::String  = "nothing" # use "scenario" to reload a predefined scenario or "nothing" to let the generator create a random scenario 
    seed_data::Int = 1234 # Define the seed to ensure random and reproductible result
    seed_model::Int = 2345
    seed_dataTest::Int = 1234
    seed_modelTest::Int = 2345
    Modulation::String = "OFDM" # define the Modulation
    dyn_value::Int = 0
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

    if haskey(kwargs, :Modulation)
        Modulation = kwargs[:Modulation]
    else 
        Modulation = ""
    end 

    if haskey(kwargs, :dyn_value)
        dyn_value = kwargs[:dyn_value]
    else 
        dyn_value = 0
    end 

    if haskey(kwargs, :Augmentation_Value)
        Augmentation_Value = kwargs[:Augmentation_Value]
    else 
        Augmentation_Value = Data_Augmented_construct() 
    end
return Data_Synth(name,nameModel,nbTx,nbSignals,Chunksize,features,S,E,C,RFF,Normalisation,pourcentTrain,configuration,seed_data,seed_model,seed_dataTest,seed_modelTest,Modulation,dyn_value,Augmentation_Value)
end
