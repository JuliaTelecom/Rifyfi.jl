using Plots
using PGFPlotsX 
""" 
Input is (128*96_000) x 2
Need to transform into 128 x 2 x 96_000 
r11 i11 
r12 i12 
r13 i13 
r21 i21 
r22 i22 
r23 i23 
...
with r_ij i is the radio index and j the time slot 
We need to have 
[r11 i11   [r21 i21 
 r12 i12    r22 i22 
 r13 i13]   r23 i23] ...
""" 
function customReshape(X::Matrix{T},batchSize) where T
    # In 
    sI1 = size(X,1)
    sI2 = size(X,2) # = 2 
    # Out 
    sO1 = batchSize # Batch size 
    sO3 = sI1 ÷ sO1   # Number of batch 
    sO2 = 2           # Number of channels, (I/Q
    #d
    xs = Array{T}(undef,sO1,sO2,sO3)
    for n in 1 : sO3 
        xs[:,:,n] .= X[ (n-1)*sO1 .+ (1:sO1),:]
    end
    return xs
end

""" 
Convert matrix of label to a vector 
For learning, the labels are in a matrix (nRadio x nTrials) where nRadio is the number of radios and nTrials is the number of batches. The matrix is full of zero with only one 1 for each column at the line index associated to the radio index.
The function provides a vector of size nTrials with each index of the vector a Int between 1 and nRadio. 
"""
convertDataLabelToVect(y) = [findfirst(x .== 1) for x in eachcol(y)]

""" 
Evaluating performance on test dataset 
Returns a vector with radio identifier and the probability to have find the good radio index 
"""
function evalP(nn,data,label)
    # ----------------------------------------------------
    # --- Getting output of nn
    # ---------------------------------------------------- 
    # We want to have the output of our neural network based on one batch 
    # As stated in the doc, the last dimension is the one of the batch i.e we should have 
    # (128 x 2 x 1) as channel is the real / imag alternation
    nBatch = size(data,3)
    allL = zeros(Int,nBatch)
    d = zeros(Float32,size(data,1),size(data,2),1)
    for i in 1 : nBatch
        # Getting the (128x2x1) element 
        d[:,:,1] .= data[:,:,i] 
        # Compute the NN => Output is Vector of size 16 
        l = nn(d)
        # We have the soft proba, switch to decision 
        c = argmax(l)[1]
        # Store decision
        allL[i] = c 
    end
    # --- Compute proba 
    v = convertDataLabelToVect(label) 
    p = sum( allL .== v) / nBatch
    # Output
    return allL,p
end

## utility functions
""" 
Returns the number of parameters to be estimated in a Flux network 
"""
num_params(model) = sum(length, Flux.params(model)) 


""" 
Round a number around 4 digits 
""" 
round4(x) = round(x, digits=4)

""" 
Compute confusion matrix for a given model and a specific dataset 
Infer dataset on model and compute confusionMatrix thanks to supervised labels
""" 
#=
function confusionMatrix(model,dataTest,labels;device=gpu)
    # --- Getting parameters 
    # --- Get number of batch 
    nbBatch = size(dataTest,3)
    # --- get number of class 
    nbClass = size(labels,1)
    # --- Getting labels 
    # Convert the label matrix to a vector with the position 
    vL = convertDataLabelToVect(labels)
    # --- Check we have same labels and data 
    @assert size(labels,2) == nbBatch "Labels and data have not the same batch size"
    # --- Init matrix 
    confMatrix = zeros(nbClass,nbClass)
    # --- Inference
    mm = model |> device
    # Here we have a matrix of size nbClass x nbBatch with soft decisions
    # --- Compute confusion matrix 
    for n in 1 : 1 : nbBatch
        # --- Output
        dd = dataTest[:,:,n:n] |> device
        dO = mm(dd) |> cpu
        # Convert the soft decision into a hard labelisation 
        cO = dO |> argmax   # Chosen radio after inference 
        cL = vL[n]          # Labeled radio
        # --- Update matrix
        # x Axis are labeled radio 
        # y Axis are chosen radios
        confMatrix[cL,cO] += 1
    end 
    # --- Normalize to have probability 
    for l in 1 : nbClass 
        confMatrix[l,:] = confMatrix[l,:] ./ sum(confMatrix[l,:])
    end
    return confMatrix
end
=#
""" 
Compute confusion matrix when inputs are the estimated labels and the true labels 
"""
function confusionMatrix(l̂::AbstractArray,l::AbstractArray,nbRadios::Number)
    confMatrix = zeros(nbRadios,nbRadios)
    @assert size(l̂) == size(l) "Estimated radio label and true radio label must have the same size: here we have $(length(l̂)) and $(length(l))"
    for n ∈ eachindex(l) 
        confMatrix[l[n],l̂[n]] += 1 
    end
    # --- Normalize to have probability 
    for n in 1 : 1 : nbRadios
        confMatrix[n,:] = confMatrix[n,:] ./ sum(confMatrix[n,:])
    end
    
    return confMatrix 
end

function plotSigPArt2(signal, Param_Data,Name_sig)
    # N = length(signal)
    N = 1000
    N = 16384
    Fs = 5.2608e6
    #Fs = 5e6
    Ts = 1/Fs
    xAx = Ts*(N-1):Ts:2*Ts*(N-1)
    burst=64
    sig= vec(signal[:,1,burst+1:burst+burst])
    plt = plot(xAx*1000,sig,label="")
    xlabel!("Time index [ms]")
    ylabel!("Real part")
    plt |> display
    savefig(plt,"Chap3/$(Name_sig).pdf")

end 

function plotSig(signal, Param_Data,Name_sig)
    # N = length(signal)
    N = 1000
    N = 16384
    Fs = 5.2608e6
    #Fs = 5e6
    Ts = 1/Fs
    xAx = 0:Ts:Ts*(N-1) 
    burst=64
    sig= vec(signal[:,1,1:burst])

    plt = plot(xAx*1000,sig,label="")
    ymin!=0
    xlabel!("Time index [ms]")
    ylabel!("Real part")
    plt |> display
    savefig(plt,"Chap3/$(Name_sig).pdf")
    #pltnew=plot(real(signal[1:N]),imag(signal[1:N]),seriestype = :scatter)
#=
    sigF = abs2.(fftshift(fft(signal[1:N])))
    xAx = ((0:1:N-1)/N .- 0.5)*Fs
    plt = plot(xAx,10*log10.(sigF),label="")
    xlabel!("Frequency [Hz]")
    ylabel!("Magnitude [dB]")
    plt |> display
    =#
end


function plotPN(signal)
    N=1000
    Fs = 5.2608e6
    #Fs = 5e6
    Ts = 1/Fs
    xAx = 0:Ts:Ts*(N-1) 
    dictMarker  = ["square*","triangle*","diamond*","*","pentagon*","rect","otimes","triangle*"];
    # --- Dictionnary for colors 
    dictColor   = ColorSchemes.tableau_superfishel_stone
    @pgf a = Axis({
                height      ="3in",             # Size of Latex object, adapted to IEEE papers 
                width       ="4in",
                grid,
                xlabel      = "Time [s]",       # X axis name 
                ylabel      = "F1 score",       # Y axis name  
                legend_style="{at={(1,0)},anchor=south east,legend cell align=left,align=left,draw=white!15!black}"         # Legend, 2 first parameters are important: we anchor the legend in bottom right (south east) and locate it in bottom right of the figure (1,0)
                },
    );



    @pgf push!(a,Plot({color=dictColor[1],mark=dictMarker[1]},Table([xAx[:],signal[1:1:1000]])))
    @pgf push!(a, LegendEntry("GPU")) # Train)
 
    pgfsave("Results/PN_1.tex",a)

end 


function plotSig_brut(signal, Param_Data,Name_sig)
    # N = length(signal)
    N = 1000
    N = 16384
    Fs = 5.2608e6#5.2e9
    #Fs = 5e6
    Ts = 1/Fs
    #xAx= Ts*(N-1):Ts:2*Ts*(N-1)
    #xAx = 0:Ts:2*Ts*(N-1)
    xAx = 0:Ts:Ts*(N-1) 
    plt = plot(xAx*1000,real(fft(P[1:N])),label="")
    xAx= Ts*(N-1):Ts:2*Ts*(N-1)
    plt = plot!(xAx*1000,real(signal[N+1:N+N]),color=([RGB(1.0,0.6824,0.2039)]),label="")
    xlabel!("Time index [ms]")
    ylabel!("Real part")
    plt |> display
    savefig(plt,"Chap3/$(Name_sig).pdf")

end 




#=

function plotSig_brut(signal, Param_Data,Name_sig)
    # N = length(signal)
    N = 1000
    N = 16384
    Fs = 5.2e9
    #Fs = 5e6
    Ts = 1/Fs
    #xAx= Ts*(N-1):Ts:2*Ts*(N-1)
    xAx = 0:Ts:2*Ts*(N-1)
  #  xAx = 0:Ts:Ts*(N-1) 
    plt = plot(xAx*1000000,real(signal[1:N+N-1]),color=([RGB(0.5,0.4,0.3)]),label="")
    plt = plot(xAx*1000000,real(signal[1:N+N-1]),color=([RGB(0.5,0.4,0.3)]),label="")
    xlabel!("Time index [μs]")
    ylabel!("Real part")
    plt |> display
    savefig(plt,"Chap3/$(Name_sig).pdf")

end 

=#

"""
Returns accuracy estimation (in percent), based on onecold estimator (label vector)
"""
getAccuracy(a,b) = sum( a .== b) / length(b)*100


function getF1_score(l̂::AbstractArray,l::AbstractArray,nbRadios::Number)
    
    tp = zeros(nbClass) # true positive
    fp = zeros(nbClass) # false positive 
    fn = zeros(nbClass) # false negative
    nbClass=nbRadios
    ŷ = onecold(l̂)
    y = onecold(l)
    # --- Compute metrics for each class 
    for c = 1 : 1 : nbClass
        cTrue = findall(y .== c)  # Corresponds to true radio c 
        cGuess = findall(ŷ .== c) # Guessed as radio c 
        # True positive 
        # We have c and we guess c 
        tp[c] += sum(ŷ[cTrue] .== c)
        # False positive  (equivalent to false alarm in detection theory)
        # We guess c but it was not c 
        fp[c] += sum(y[cGuess] .!= c)
        # False negative 
        # We predict that it is not c but it was c 
        fn[c] += sum(ŷ[cTrue] .!= c)
    end 

# Macro average approach, calculate precision and recall per class  and average
# Mirror to what is done in Gegelati, with same argument 
# (chosen instead of the global f1 score as it gives an equal weight to
# the f1 score of each class, no matter its ratio within the observed
# population)



replace_nan!(x) = isnan(x) ? 0 : x

# filtrer 

precision = (tp ./ (tp .+ fp)) 
recall    = (tp ./ (tp .+ fn))

F1        = replace_nan!.(2 ./ (1 ./precision + 1 ./recall))|> mean

end 
