
module Utils 


function timeToIndex(tBeg,fe)
    Int(round(tBeg*fe))
end
function indexToTime(idx,fe)
    return idx * fe 
end

function extractFromTimes(tBeg,tFinal,sig,fe)
    indexBeg = Int(round(tBeg*fe))
    indexEnd = Int(round(tFinal*fe))
    @assert indexBeg > 1 "Input time must be positive"
    @assert indexEnd < length(sig) "Output time must be lower than signal duration"
    return sig[indexBeg:indexEnd]
end


function scale_int(sigId)
    xR = maximum(abs.(real(sigId)))
    xI = maximum(abs.(imag(sigId)))
    scale = Float32(1-2^-15)
    sentBuffer = (real(sigId)/xR + im*imag(sigId)/xI) * scale
    return sentBuffer
end


# ----------------------------------------------------
# --- Dependencies 
# ---------------------------------------------------- 
using FFTW 
using Images, Interpolations
using DSP


# --- Exportation 
# ---------------------------------------------------- 
export getSpectrum
export getWaterfall 
export getWelch

# ----------------------------------------------------
# --- Main calls
# ---------------------------------------------------- 
"""
Compute the periodogram of the input signal `sig` sampled at the frequency `fs`. The additional parameter N can be used to restrict the PSD on N points
"""
function getSpectrum(fs,signal;N=nothing)
    # --- Restrict to input dimension 
    if !isnothing(N)
        # ----------------------------------------------------
        # --- Compute Welch with no 50% Overlap and add 
        # ----------------------------------------------------
        nbSeg = length(signal) ÷ N
        window = DSP.blackman(N) 
        S = zeros(ComplexF32,N) 
        n = 0
        while true 
            _sig = signal[ n .+ (1:N)]
            _X   = sqrt(1/length(_sig)) * fftshift(fft(_sig .* window))   
            S  .+= _X 
            n += N ÷ 2
            if (n + N) > length(signal)
                break 
            end
        end
        S = abs2.(S)
    else 
        # ----------------------------------------------------
        # --- Compute periodogram
        # ----------------------------------------------------
        # --- Compute classic PSD 
        N = length(signal)
        window = DSP.blackman(N)
        X   = sqrt(1/length(signal)) * fftshift(fft(signal .* window))
        S = abs2.(X)
    end
    S   = 10*log10.(S)
    # --- Calculate axis 
    freqAx = collect(((0:N-1)./N .- 0.5)*fs);
	return (freqAx,S);
end


function getWaterfall(fe,sig;sizeFFT=1024,outSize=(1024,1024))
    # --- Calculate the T/F Matrix 
    nbSeg   = length(sig) ÷ sizeFFT;
    ss      =  @views sig[1:nbSeg*sizeFFT];
    ss      = reshape(ss,sizeFFT,nbSeg)
    sMatrix = zeros(Float64,sizeFFT,nbSeg);
    for iN = 1 : 1 : nbSeg 
#        tup =  periodogram(ss[:,iN])       
#        sMatrix[:,iN] = tup[2]
        sMatrix[:,iN] = abs2.(fftshift(fft(ss[:,iN])));
    end
    # --- Reduce the output image to match the one in parameters 
    sMatrix = 10*log10.(imresize(sMatrix,outSize[1],outSize[2]))
    # --- Calculate the axis for the T/F
    fAx = collect(((0:1:outSize[1]-1)./outSize[1] .- 0.5) .* fe);
    duration = nbSeg * sizeFFT / fe 
    tAx = range(0,stop=duration,length=outSize[2])
    return tAx,fAx,sMatrix;
end
getWaterfall(sig;sizeFFT=1024) = getWaterfall(1,sig;sizeFFT=sizeFFT);



""" 
---
Calculate the average power of the input signal 
σ	= 1 / N Σ | x[n] | ^2 
# --- Syntax 
σ	= avgPower(x);
# --- Input parameters 
- x	  : Input signal [Array{Any}]
# --- Output parameters 
- σ	  : Estimated power [Float64]
# --- 
# v 1.0 - Robin Gerzaguet.
"""
function avgPower(x)
	return 1/length(x) * sum( abs2.(x) ) ;
end
export avgPower


function Plot_PSD(signal)
    signal=X[:,1]
    N = 10_000
    Fs = 5e6
    Ts = 1/Fs
    sigF = abs2.(fftshift(fft(signal[1:N])/N))
    xAx = ((0:1:N-1)/N .- 0.5)*Fs
    siz=maximum(sigF)
    plt = plot(xAx,10*log10.(sigF/siz),label="")


end 




end # module
