
# arguments for the `train` function 
Base.@kwdef mutable struct Args
    η = 1e-4            # learning rate e-5
    dr = 0
    λ = 0               # L2 regularizer param, implemented as weight decay
    batchsize = 100     # batch size
    epochs = 700        # number of epochs
    seed = 12           # set seed > 0 for reproducibility
    use_cuda = true     # if true use cuda (if available)
    infotime = 1 	    # report every `infotime` epochs
    checktime = 0       # Save the model every `checktime` epochs. Set to 0 for no checkpoints.
    tblogger = true     # log training with tensorboard
    tInit       = 0.0 
    timings    = zeros(epochs) # Store timings of train 
end



function Args_construct(; kwargs...)
    if haskey(kwargs, :η)
        η = kwargs[:η]
    else
        η=1e-4
    end

    if haskey(kwargs, :dr)
        dr = kwargs[:dr]
    else
        dr=0
    end
    if haskey(kwargs, :λ)
        λ = kwargs[:λ]
    else 
        λ = 0
    end

    if haskey(kwargs, :batchsize)
        batchsize = kwargs[:batchsize]
    else 
        batchsize = 100
    end

    if haskey(kwargs, :epochs)
        epochs = kwargs[:epochs]
    else 
        epochs = 1000
    end

    if haskey(kwargs, :seed)
        seed = kwargs[:seed]
    else 
        seed = 12
    end
    if haskey(kwargs, :use_cuda)
        use_cuda = kwargs[:use_cuda]
    else 
        use_cuda = true
    end
    if haskey(kwargs, :infotime)
        infotime = kwargs[:infotime]
    else 
        infotime = 1
    end
    if haskey(kwargs, :checktime)
        checktime = kwargs[:checktime]
    else 
        checktime = 0
    end
    if haskey(kwargs, :tblogger)
        tblogger = kwargs[:tblogger]
    else 
        tblogger = true
    end
    if haskey(kwargs, :tInit)
        tInit = kwargs[:tInit]
    else 
        tInit = 0.0
    end
    if haskey(kwargs, :timings)
        timings = kwargs[:timings]
    else 
        timings  = zeros(epochs)
    end
    return Args(η,dr,λ,batchsize,epochs,seed,use_cuda,infotime,checktime,tblogger,tInit,timings)
end



Base.@kwdef mutable struct Network_struct
    Networkname::String = "AlexNet"
    NbClass::Int = 4
    Chunksize::Int = 256
    NbSignals::Int = 1000
    Seed_Network = 12
    Train_args::Args =  Args()
    model  = initAlexNet(256,4,Train_args.dr)[1]
    loss = initAlexNet(256,4,Train_args.dr)[2]
end


function Network_struct(; kwargs...)
    if haskey(kwargs, :Networkname)
        Networkname = kwargs[:Networkname]
    else
        Networkname="AlexNet"
    end

    if haskey(kwargs, :NbClass)
        NbClass = kwargs[:NbClass]
    else 
        NbClass = 4
    end

    if haskey(kwargs, :Chunksize)
        Chunksize = kwargs[:Chunksize]
    else 
        Chunksize = 256
    end

    if haskey(kwargs, :NbSignals)
        NbSignals = kwargs[:NbSignals]
    else 
        NbSignals = 1000
    end

    if haskey(kwargs, :Seed_Network)
        Seed_Network = kwargs[:Seed_Network]
    else 
        Seed_Network = 12
    end

    if haskey(kwargs, :Train_args)
        Train_args = kwargs[:Train_args]
    else 
        Train_args = Args()
    end
    if haskey(kwargs, :model)
        model = kwargs[:model]
        loss =kwargs[:loss]
    else 
        model,loss = initAlexNet(Chunksize,NbClass,Train_args.dr)
    end


    return Network_struct(Networkname,NbClass,Chunksize,NbSignals,Seed_Network,Train_args,model,loss) 
end







