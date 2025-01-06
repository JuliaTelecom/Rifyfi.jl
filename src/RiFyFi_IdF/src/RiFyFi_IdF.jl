module RiFyFi_IdF


using Flux   
using CUDA   
using ProgressMeter: @showprogress
using Flux.Losses: crossentropy
using Flux.Optimise: Optimiser, WeightDecay
using Flux: onehotbatch, onecold
using Random
using Statistics
using DataFrames
using DelimitedFiles 
using BSON
using Infiltrator
export inference
export loadCNN

export initAlexNet
export initGDA
export initNewCNN
export initWiSig


function initWiSig(x,nbRadioTx,dr)# tiré de WiSig  
m = Chain(
        #    x -> reshape(x, (size(x)[1], 2, 1, size(x)[3])),
            Conv((3,2), 1 => 8, pad=SamePad(), relu),
            MaxPool((2,1)),
            Conv((3,2), 8 => 16, pad=SamePad(), relu),
            MaxPool((2,1)),
            Conv((3,2), 16 => 16, pad=SamePad(), relu),
            MaxPool((2,1)),
            Conv((3,1), 16 => 32, pad=SamePad(), relu),
            MaxPool((2,1)),
            Conv((3,1), 32 => 16, pad=SamePad(), relu),
            MaxPool((2,1)),
            Flux.flatten,
            Dense(256, 100, relu),
            Dense(100, 80, relu),
            Dropout(dr),
            Dense(80,nbRadioTx),
            Flux.softmax
        )
        testmode!(m, false) # Important sinon le CNN ne prend pas en compte le drop out = surapprentissage
        loss(ŷ, y)= crossentropy(ŷ, y)
        return (m,loss)
end

function initNewCNN(x,nbRadioTx,dr)# tiré de Oracle 
    m = Chain(
        #----------------------------------------------------
        # --- Convolutionnal layers
        # ---------------------------------------------------- 
        # Bloc 1 
        Conv((7,), 2 => 50, pad=SamePad(), relu),
        Conv((7,), 50 => 50, pad=SamePad(), relu),
        Flux.flatten,
        Dense(50*x, 256, relu), # 1024 = 8 * 128
        Dropout(dr),
        Dense(256, 80, relu),
        Dropout(dr),
        Dense(80,nbRadioTx), # Couche non indiquée sur la doc mais obligatoire pour avoir un Softmax correct
        softmax
       )
    testmode!(m, false) # Important sinon le CNN ne prend pas en compte le drop out = surapprentissage
    loss(ŷ, y)= crossentropy(ŷ, y)
    return (m,loss)
end

function initAlexNet(x,nbRadioTx,dr)
    #dr = 0 #Dropout rate
    sizeEnd= Int(x*128/2^4)
    m = Chain(
              # ----------------------------------------------------
              # --- Convolutionnal layers
              # ---------------------------------------------------- 
              # Bloc 1 
              Conv((7,), 2 => 128, pad=SamePad(), relu),                    
              Conv((5,), 128 => 128, pad=SamePad(), relu),                 
              MaxPool((2,)),
              # Bloc 2
              Conv((7,), 128 => 128, pad=SamePad(), relu),
              Conv((5,), 128 => 128, pad=SamePad(), relu),
              MaxPool((2,)),
              # Bloc 3
              Conv((7,), 128 => 128, pad=SamePad(), relu),
              Conv((5,), 128 => 128, pad=SamePad(), relu),
              MaxPool((2,)),
              # # Bloc 4
              Conv((7,), 128 => 128, pad=SamePad(), relu),
              Conv((5,), 128 => 128, pad=SamePad(), relu),
              MaxPool((2,)),
              # ----------------------------------------------------
              # --- Dense perceptron
              # ---------------------------------------------------- 
              Flux.flatten,
              Dense(sizeEnd, 256, relu),  
              Dropout(dr),
              Dense(256, 128, relu),
              Dropout(dr),
              # ----------------------------------------------------
              # --- To classes
              # ---------------------------------------------------- 
              Dense(128,nbRadioTx), 
              Flux.softmax
             )
    testmode!(m, false) #prise en compte du drop out pour éviter le surapprentissage
    loss(ŷ, y)= crossentropy(ŷ, y)
    return (m,loss)
end






function initGDA(x,nbRadioTx,dr)
   # dr = 0.5 #Dropout rate
    m = Chain(
            Conv((10,), 2 => 64, pad=SamePad(), relu),
            MaxPool((2,)),
            Conv((10,), 64 => 32, pad=SamePad(), relu),
            MaxPool((2,)),
            Conv((10,), 32 => 16, pad=SamePad(), relu),
            MaxPool((2,)),
            Flux.flatten,
            Dense(512,64),
            Dense(64,4),
            Dense(4,nbRadioTx),
            Flux.softmax
             )
    testmode!(m, false) # Important sinon le CNN ne prend pas en compte le drop out = surapprentissage
    loss(ŷ, y)= crossentropy(ŷ, y)
    return (m,loss)
end



function loadCNN(cnnPath)
    # --- Loading data 
    dict = BSON.load(cnnPath,@__MODULE__) 
    # --- Exporting variables 
    model     = dict[:model]
    testAcc   = dict[:testAcc]
    trainAcc  = dict[:trainAcc]
    testLoss  = dict[:testLoss]
    trainLoss = dict[:trainLoss]
    #=moy = 0 
        std = 1
    try 
        moy = dict[:moy] 
    catch exception
    end
    try 
        std= dict[:std] 
    catch exception
    end =#
    #return (model=model,testAcc=testAcc,trainAcc=trainAcc,testLoss=testLoss,trainLoss=trainLoss,moy=moy,std=std)
    return (model=model,testAcc=testAcc,trainAcc=trainAcc,testLoss=testLoss,trainLoss=trainLoss)
end

""" 
Use model to infer dataset and return the labeled radio. Different from model[dataset] as it uses GPU and dataLoader with batch 
Ŷ,Y = inference(model,dataset)
Returns 2 vectors 
- The estimated labeled radio 
- The true labeled radio
""" 
function inference(model,dataset,device)
    #CUDA.functional() ? device = gpu : device = cpu 
    mm = model |> device 
    Ŷ = Int[]
    Y = Int[]
    
    for (x,label) in dataset
      #  xG = reshape(x, (256,2,1,(size(x))[3])) |> device # To GPU 
      #  xG = reshape(x, (size(x)[1], 2, 1, size(x)[3])) 

        xG = x |> device # To GPU 
        r = mm(xG) |> cpu # Infer model in GPU. Output is soft matrix 
        y = onecold(label)
        ŷ = onecold(r) # Hard decision to get radio label 
        push!(Ŷ,ŷ...)
        push!(Y,y...)
        
    end
        return Ŷ,Y
end



include("customTrain.jl")
export customTrain!
include("Struct_Network.jl")
export Network_struct
export Args
export Args_construct

end 


