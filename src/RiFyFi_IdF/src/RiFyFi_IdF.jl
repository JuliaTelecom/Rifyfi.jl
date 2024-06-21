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

export inference
export initAlexNet
export loadCNN
export initShen
export initMerchant2021A
export initResNet
export initGDA
export initFeng
export initDense
export DenseEmma
export F1Score

function ResNet(x,nbRadioTx,dr)
    m = Chain(
        Chain([
        Conv((7,), 2 => 64, pad=3, stride=2, bias=false),  # 9_408 parameters
        BatchNorm(64, relu),              # 128 parameters, plus 128
        MaxPool((3,), pad=1, stride=2),
        Parallel(
            Chain(
            Conv((3,), 64 => 64, pad=1, bias=false),  # 36_864 parameters
            BatchNorm(64, relu),          # 128 parameters, plus 128
            Conv((3,), 64 => 64, pad=1, bias=false),  # 36_864 parameters
            BatchNorm(64),                # 128 parameters, plus 128
            ),
            identity,
        ),
        Parallel(
            Chain(
            Conv((3,), 64 => 64, pad=1, bias=false),  # 36_864 parameters
            BatchNorm(64, relu),          # 128 parameters, plus 128
            Conv((3,), 64 => 64, pad=1, bias=false),  # 36_864 parameters
            BatchNorm(64),                # 128 parameters, plus 128
            ),
            identity,
        ),
        Parallel(
            Chain(
            Conv((3,), 64 => 128, pad=1, stride=2, bias=false),  # 73_728 parameters
            BatchNorm(128, relu),         # 256 parameters, plus 256
            Conv((3,), 128 => 128, pad=1, bias=false),  # 147_456 parameters
            BatchNorm(128),               # 256 parameters, plus 256
            ),
            Chain([
            Conv((1, ), 64 => 128, stride=2, bias=false),  # 8_192 parameters
            BatchNorm(128),               # 256 parameters, plus 256
            ]),
        ),
        Parallel(
            Chain(
            Conv((3,), 128 => 128, pad=1, bias=false),  # 147_456 parameters
            BatchNorm(128, relu),         # 256 parameters, plus 256
            Conv((3,), 128 => 128, pad=1, bias=false),  # 147_456 parameters
            BatchNorm(128),               # 256 parameters, plus 256
            ),
            identity,
        ),
        Parallel(
            Chain(
            Conv((3,), 128 => 256, pad=1, stride=2, bias=false),  # 294_912 parameters
            BatchNorm(256, relu),         # 512 parameters, plus 512
            Conv((3,), 256 => 256, pad=1, bias=false),  # 589_824 parameters
            BatchNorm(256),               # 512 parameters, plus 512
            ),
            Chain([
            Conv((1,), 128 => 256, stride=2, bias=false),  # 32_768 parameters
            BatchNorm(256),               # 512 parameters, plus 512
            ]),
        ),
        Parallel(
            Chain(
            Conv((3,), 256 => 256, pad=1, bias=false),  # 589_824 parameters
            BatchNorm(256, relu),         # 512 parameters, plus 512
            Conv((3,), 256 => 256, pad=1, bias=false),  # 589_824 parameters
            BatchNorm(256),               # 512 parameters, plus 512
            ),
            identity,
        ),
        Parallel(
            Chain(
            Conv((3,), 256 => 512, pad=1, stride=2, bias=false),  # 1_179_648 parameters
            BatchNorm(512, relu),         # 1_024 parameters, plus 1_024
            Conv((3,), 512 => 512, pad=1, bias=false),  # 2_359_296 parameters
            BatchNorm(512),               # 1_024 parameters, plus 1_024
            ),
            Chain([
            Conv((1, ), 256 => 512, stride=2, bias=false),  # 131_072 parameters
            BatchNorm(512),               # 1_024 parameters, plus 1_024
            ]),
        ),
        Parallel(
            Chain(
            Conv((3,), 512 => 512, pad=1, bias=false),  # 2_359_296 parameters
            BatchNorm(512, relu),         # 1_024 parameters, plus 1_024
            Conv((3,), 512 => 512, pad=1, bias=false),  # 2_359_296 parameters
            BatchNorm(512),               # 1_024 parameters, plus 1_024
            ),
            identity,
        ),
        ]),
        Chain(
            AdaptiveMeanPool((1,)),
            Flux.flatten,
            Dense(512 => x),               # 513_000 parameters
            Flux.softmax
        ),
    )
return m
end

#Merchant2018
function initMerchant2021A(x,nbRadioTx,dr) 
    sizeEnd= Int(x*128/2^6)
   # dr = 0.5 #Dropout rate
    m = Chain(
        Conv((19,), 2 => 128, pad=SamePad(), relu),  
        MaxPool((2,)), 
        Conv((15,), 128 => 32, pad=SamePad(), relu),  
        MaxPool((2,)), 
        Conv((11,), 32 => 16, pad=SamePad(), relu),  
        MaxPool((2,)), 
        Flux.flatten,
        Dense(sizeEnd,128), 
        Dropout(dr),
        Dense(128,16), #     
        Dropout(dr),  
        Dense(16,nbRadioTx), #           
        Flux.softmax
             )
    testmode!(m, false) # Important sinon le CNN ne prend pas en compte le drop out = surapprentissage
    loss(ŷ, y)= crossentropy(ŷ, y)
    return (m,loss)end

function initFeng(x,nbRadioTx,dr)
        #dr = 0.5 #Dropout rate
        m = Chain(
            Conv((1,), 2 => 32, relu; bias = false),
            MaxPool((2,)),
            Conv((13,), 32 => 32, pad=SamePad(), relu),
            MaxPool((2,)),
            Conv((13,), 32 => 16, pad=SamePad(), relu),
            MaxPool((2,)),
            Flux.flatten,
            Dense(512,32),
            Dropout(dr),
            Dense(32,x),
            Flux.softmax
                 )
        testmode!(m, false) # Important sinon le CNN ne prend pas en compte le drop out = surapprentissage
        return m
end



#Shen ou anciennement Infocom21
function initShen(x,nbRadioTx,dr)
    sizeEnd= Int(x*32/2^5)
   # dr = 0.5 #Dropout rate
    m = Chain(
            Conv((6, ), 2 => 8, pad=SamePad(), relu),  
            MaxPool((4,1)),
            Conv((6,), 8 => 16, pad=SamePad(), relu),  
            MaxPool((4,1)),
            Conv((6,), 16 => 32, pad=SamePad(), relu),  
            Flux.flatten,
            Dense(sizeEnd,nbRadioTx), 
            #Dropout(dr),
            Flux.softmax
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


function initDense(x,nbRadioTx,dr)
m = Chain(
    Flux.flatten,
    Dense(512 => 3, tanh),   # activation function inside layer
    BatchNorm(3),
    Dense(3 => nbRadioTx),
    softmax)  
    testmode!(m, false) #prise en compte du drop out pour éviter le surapprentissage
    loss(ŷ, y)= crossentropy(ŷ, y)
    return (m,loss)
end 


function DenseEmma(x,nbRadioTx,dr)
    m = Chain(
            #x -> reshape(x, (size(x)[1], 2, 1, size(x)[3])),
            Flux.flatten,
            #Dropout(dr),
            Dense(512, 15),
            leakyrelu,
            #leakyrelu,
            Dense(15, 10),
            selu,
            Dense(10,nbRadioTx),
            Flux.softmax
        )
        testmode!(m, false) #prise en compte du drop out pour éviter le surapprentissage
        loss(ŷ, y)= crossentropy(ŷ, y)
        return (m,loss)
    end 

#=
    function DenseEmma(x,nbRadioTx,dr)
        m = Chain(
                #x -> reshape(x, (size(x)[1], 2, 1, size(x)[3])),
                Flux.flatten,
                #Dropout(dr),
                Dense(512, 1000),
                Dropout(dr),
                leakyrelu,
                #leakyrelu,
                Dense(1000,nbRadioTx),
                Flux.softmax
            )
            testmode!(m, false) #prise en compte du drop out pour éviter le surapprentissage
            loss(ŷ, y)= crossentropy(ŷ, y)
            return (m,loss)
        end 
=#

function initGDA(x,nbRadioTx,dr)
   # dr = 0.5 #Dropout rate
    m = Chain(
            Conv((10,), 2 => 64, pad=SamePad(), relu),
           # MaxPool((2,)),
            Conv((10,), 64 => 32, pad=SamePad(), relu),
           # MaxPool((2,)),
            Conv((10,), 32 => 16, pad=SamePad(), relu),
           # MaxPool((2,)),
            Flux.flatten,
            Dense(4096,64),
            Dense(64,4),
            Dense(4,x),
            Flux.softmax
             )
    testmode!(m, false) # Important sinon le CNN ne prend pas en compte le drop out = surapprentissage
    loss(ŷ, y)= crossentropy(ŷ, y)
    return (m,loss)
end



function loadCNN(cnnPath)
    # --- Loading data 
   # dict = BSON.parse(cnnPath)
    
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
       # xG = reshape(x, (256,2,1,(size(x))[3])) |> device # To GPU 
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