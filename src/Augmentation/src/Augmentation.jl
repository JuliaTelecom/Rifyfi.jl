module Augmentation

using DataFrames
using Random
using DigitalComm
using DSP
using Distributed

include("Struct_Augmentation.jl")
export Data_Augmented
export Data_Augmented_construct
export Add_diff_Chanel_train_test
export getChannel


function Add_diff_Channel_train_test(bigMat, bigLabels, N, channel, ChunkSize, pourcentAugment, nbRadioTx,seed,burstSize)
    nbChunk = Int(size(bigMat,3)  )
    nbsignauxParRadio = Int(nbChunk/nbRadioTx)
    (snrRange,cfoRange,τₘ,nbTaps) = AugmentationParameters()
    Augmented_MatTemp = zeros(Float32,ChunkSize,2,nbChunk*pourcentAugment)
    Augmented_Labels = zeros(Float32,size(bigLabels,1),size(bigLabels,2)*pourcentAugment)
    concatVec = zeros(ComplexF32,ChunkSize*nbChunk)
    concaten!(concatVec,1:nbChunk,bigMat,ChunkSize)
    concatMat = reshape(concatVec,Int(ChunkSize*nbChunk/nbRadioTx),nbRadioTx)
    indice = 1
    #concatMat = bigMat #reshape(concatVec,Int(ChunkSize*nbChunk/nbRadioTx),nbRadioTx)
    @info "Create set with augmentation ..."     

    for iR = 1 : 1 : nbRadioTx 
        burstSizeTemp = burstSize
        # --- Signal associated to radio 
        sig = @views concatMat[:,iR] # 256 * nbBurstParRadio = nbsignauxParRadio
        nbBurstParRadio = Int(floor(size(sig,1)/(burstSizeTemp*ChunkSize)) +1)
        # --- Call the augmenting // feeding job 
        # donné augmentée
        X_mp = zeros(Float32,ChunkSize,2,Int(pourcentAugment*N))
        Y_mp = zeros(Float32,nbRadioTx,Int(pourcentAugment*N))
        # -- Working vector to augment data
        sigAugmented = zeros(ComplexF32,Int(ChunkSize*N))
        # --- Iterative data augmentation chunk après chunk
        for iN = 1 : 1 : Int(nbBurstParRadio)

            # Integrer le pourcentage d'augmentation 
            if iN == nbBurstParRadio
                burstSizeTemp = nbsignauxParRadio- (iN-1)*burstSize
            end
            subSigAugmented = zeros(ComplexF32,Int(ChunkSize*burstSizeTemp))
            subSig = @views sig[  Int((iN-1)*ChunkSize*burstSize) .+ (1:Int(ChunkSize*burstSizeTemp))]
            for j = 1 : 1 : pourcentAugment  
                Random.seed!(seed * iR + j )    
                #--- Data augmentation 
                # First get unique impairment model
                σ   = choose(snrRange)
                f   = choose(cfoRange) / 5e6 # Normalized CFO, wrt sampling rate 
                if channel == "multipath"
                    cir = getChannel(τₘ;model=:multipath,nbTaps)
                elseif channel == "etu"
                    cir = getChannel(τₘ;model=:etu,nbTaps)
                elseif channel == "eva"
                    cir = getChannel(τₘ;model=:eva,nbTaps)
                end
                # Augment the data
                signalAugmentation!(subSigAugmented,channel,"OfflineV1",subSig,cir,f,σ)
                sig_real= IQsample_real(subSigAugmented)
                sig_imag = IQsample_ima(subSigAugmented)
                # --- Populate segment 
                for n = 1 : 1 : Int(burstSizeTemp)    
                    X_mp[:,1,indice+n-1] = sig_real[Int(ChunkSize*(n-1))+1:Int(ChunkSize*n)]
                    X_mp[:,2,indice+n-1] = sig_imag[Int(ChunkSize*(n-1))+1:Int(ChunkSize*n)]
                    Y_mp[iR ,indice+n-1] = 1
                end
                indice+=burstSizeTemp                 
            end
        end
        indice = 1
        Augmented_MatTemp[:,:,(nbsignauxParRadio*pourcentAugment)*(iR-1)+1:(nbsignauxParRadio*pourcentAugment)*(iR-1)+nbsignauxParRadio*pourcentAugment] = X_mp
        Augmented_Labels[:,(nbsignauxParRadio*pourcentAugment)*(iR-1)+1:(nbsignauxParRadio*pourcentAugment)*(iR-1)+nbsignauxParRadio*pourcentAugment] = Y_mp
    end
    X = Augmented_MatTemp
    Y = Augmented_Labels
# --- Ajout de la Normalisation si besoin
return (X,Y)
end

function AugmentationParameters()
    # ----------------------------------------------------
    # --- Range of parameters 
    # ---------------------------------------------------- 
    # Define the range of the additive white noise in dB
    snrRange = (0:30)
    # Define the range of the added CFO in Hz 
    cfoRange = (-1e3:1e1)
    # Define the max delay spread of the CIR 
    # CP value in OFDM here
    τₘ = 36 
    # We canont define a pure random multipath channel so we define here the number of taps we will have. the position of the bins will be set randomly between 1 and τₘ
    nbTaps = 8 
    return    snrRange,cfoRange,τₘ,nbTaps

end
"""Permet de concatener en le signal I + iQ """
function concaten!(tmp_concat,indexList,bigMat,ChunkSize)
    j = 1
    for i in indexList
        tmp_concat[(j-1)*ChunkSize+1:j*ChunkSize] =  bigMat[:,:,i][:,1] + bigMat[:,:,i][:,2]im
        j = j+1
    end
end


""" 
Transform a vector of label (with each index a number of radio) into a matrix of 0 with a 1 per column associated to the radio index
"""
function transformLabels(Y,nbRadios)
    Z = zeros(Int,nbRadios,length(Y))
    for n in eachindex(Y)
        Z[1+Int(Y[n]),n] = 1 
    end 
    return Z
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


""" 
Create a databse of signal impaired with channel model to avoid overfitting 
The purpose is to get 4 matrixes: 2 for the train and 2 for the test 
For the train, we have 2 matrixes 
- One matrix with IQ signals, of size 128 x 2 x N with 128 the batch size and 2 as we have real and imag parts. N corresponds to nR x nS with nR the number of radios (16) and nS the number of chunks per radio
- One matrix with the label a nR x N matrix. This is a sparse matrix with only one 1 per column at the radio index position.
For the test we have 2 matrixes exactly as before but with a nS smaller (9 times lower in test)

Create train and test dataset (here matrixes and not dataloader) for a given distance in feet. If the keyword "all" is used the train and test sets uses all distances.
(X_train,Y_train,X_test,Y_test) = create_X_and_Y(distance)
"""


""" Return a pure random CIR sequence with maximal delay spread τ
TODO: Sotchastic model ? 
"""
function getChannel(τ::Number;model=:none,nbTaps=1)
    if model == :none 
        # No channel, no attenuation, no phase => it is 1
        cir = 1
    elseif model == :randn
        # Pure random sequence, without profile
        cir = randn(ComplexF32,τ)
    elseif model== :multipath 
        # Select the tap of interest 
        # We will have energy at this locations only
        id   = choose(collect(1:τ),nbTaps)
        # Populate the CIR
        cir = zeros(ComplexF32,τ)
        cir[id] .= randn(ComplexF32,nbTaps)
    elseif model == :etu 
        channelModel = initChannel("constetu",2.4e9,5e6,0)
      #  Random.seed!()
        channelReal  = DigitalComm.getChannel(0,channelModel)
        cir = channelReal.cir
    elseif model == :eva 
        channelModel = initChannel("consteva",2.4e9,5e6,0)
      #  Random.seed!()
        channelReal  = DigitalComm.getChannel(0,channelModel)
        cir = channelReal.cir
    end
    return cir
end
#




""" Get one element in the array, randomly
"""
@inline function choose(a::AbstractVector)
    return a[rand(1:end)]
end

"""" Get several element in the array; all differents
""" 
function choose(a::AbstractVector{T},numb::Number) where T
    @assert numb < length(a) "Unable to select randomly more elements ($numb) than the length of input vector ($(length(a)))"
    # Everyday I shuffling 
    b = shuffle(a)
    # Keep only numb elements
    return b[1:numb]
end





"""" 
Augment the input signal, based on CIR impulse response, CFO value and noise variance
""" 
function signalAugmentation(sig::AbstractVector,chanel,augmentationType,h::AbstractVector,cfo::Number,σ::Number)
    sigOut = similar(sig)
    signalAugmentation!(sigOut,chanel,augmentationType,sig,h,cfo,σ)
    return sigOut
end
function signalAugmentation!(sigOut,chanel,augmentationType,sig::AbstractVector,h::AbstractVector,cfo::Number,σ::Number)
    # --- Tx CFO 
    addCFO!(sigOut,ComplexF32.(sig),cfo,1,0)
    # --- Apply Channel model
    # Convolution and tail truncation model 
    δ = length(h)÷2
    if augmentationType =="OnlineV2" && chanel=="multipath" || chanel=="eva"
        sigChan = DSP.conv(sigOut,h)[1+δ:end-δ]
    else
        sigChan = DSP.conv(sigOut,h)[1+δ:end-δ+1]
    end 
    # --- AWGN 
    
    addNoise!(ComplexF32.(sigOut),ComplexF32.(sigChan),σ)
    return sigOut
end


function IQsample_real(data)
    return real(data)
end

function IQsample_ima(data)
    return imag(data)
end



end # module
