Base.@kwdef mutable struct Data_Augmented
    augmentationType::String = "No_channel"
    Channel::String = "mulitpath"
    Channel_Test::String = "mulitpath"
    nb_Augment::Int = 1
    seed_channel::Int = 12
    seed_channel_test::Int = 12
    burstSize::Int =64
end



function Data_Augmented_construct(; kwargs...)
    if haskey(kwargs, :augmentationType)
        augmentationType = kwargs[:augmentationType]
    else
        augmentationType="No_channel"
    end

    if haskey(kwargs, :Channel)
        Channel = kwargs[:Channel]
    else 
        Channel = "mulitpath"
    end

    if haskey(kwargs, :Channel_Test)
        Channel_Test = kwargs[:Channel_Test]
    else 
        Channel_Test = "mulitpath"
    end

    if haskey(kwargs, :nb_Augment)
        nb_Augment = kwargs[:nb_Augment]
    else 
        nb_Augment = 1
    end

    if haskey(kwargs, :seed_channel)
        seed_channel = kwargs[:seed_channel]
    else 
        seed_channel = 12
    end

    if haskey(kwargs, :seed_channel_test)
        seed_channel_test = kwargs[:seed_channel_test]
    else 
        seed_channel_test = 999999999
    end
    if haskey(kwargs, :burstSize)
        burstSize = kwargs[:burstSize]
    else 
        burstSize = 64
    end
    return Data_Augmented(augmentationType,Channel, Channel_Test, nb_Augment, seed_channel, seed_channel_test,burstSize)
end
