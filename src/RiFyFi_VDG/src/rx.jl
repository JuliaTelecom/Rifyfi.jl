
""" Convert a time domain OFDM signal (one symbol) to a constellation vector and a bin ary decoded sequence
"""
function decode(sigRx::Vector{Complex{T}}) where T

    # --- The configuration 
    (nFFT,nCP,sizeSymb,mcs,crcSize,sampleRate,carrierFreq,nbBitsUncoded,inputFEC,outputFEC,allocatedSubcarriers,pilotSubcarriers,payloadSubcarriers,pilotVal,pilotTime,packetSize,nbRepeat,nbPayloadSubcarriers,nbPilotSubcarriers) = getConfig()
    nbP = length(pilotSubcarriers)
    chanCoeffs = zeros(Complex{T},nFFT)
    qamEq = zeros(Complex{T},length(payloadSubcarriers))


    # --- Frequency domain
    sig = sigRx[1+nCP:end]
    @assert length(sig) == nFFT "Size of input vector ($(length(sig))) is not equal to FFT size ($nFFT)"
    cacheDemod = fft(sig)

    # --- CHEST 
    # ----------------------------------------------------
    # --- Channel Estimation 
    # ---------------------------------------------------- 
    # --- Getting channel elements @ pilot position
    for n ∈ 1:1:nbP
        sc = pilotSubcarriers[n]
        chanCoeffs[sc] = cacheDemod[sc] / pilotVal[sc] #FIXME Optimize @e310, this is 1
    end
    # ----------------------------------------------------
    # --- Channel Interpolation 
    # ---------------------------------------------------- 
    # Interpolation
    for ic = 1:1:nbP-1
        # --- Circular pilot managment 
        # We now have sc and nextSc as the 2 pilot subcarriers 
        sc = pilotSubcarriers[ic]
        nextSc = pilotSubcarriers[ic+1]
        nbGap = nextSc - sc
        for n ∈ 1:nbGap
            # --- Direct mapping 
            chanCoeffs[sc + n] = chanCoeffs[nextSc]
            # --- Linear interpolation 
            # chanCoeffs[sc+n] = (chanCoeffs[nextSc] - chanCoeffs[sc]) / nbGap * n + chanCoeffs[sc]
        end
    end
    # --- Last carrier is special (circular)
    sc = pilotSubcarriers[end]
    nextSc = pilotSubcarriers[1]
    nbGap = nFFT - sc + nextSc
    for n ∈ 1:nbGap
        chanCoeffs[mod(sc + n - 1, nFFT)+1] = (chanCoeffs[nextSc] - chanCoeffs[sc]) / nbGap * n + chanCoeffs[sc]
    end
    # ----------------------------------------------------
    # --- Equalization
    # ---------------------------------------------------- 
    for cnt = 1:1:length(payloadSubcarriers)
        ic = payloadSubcarriers[cnt]
        # qamEq[cnt] = cacheDemod[ic] / chanCoeffs[ic]
        qamEq[cnt] = cacheDemod[ic] 
    end

    # ----------------------------------------------------
    # --- To binary stream
    # ---------------------------------------------------- 
    # --- Coded binary sequence 
    bitRx = zeros(UInt8,length(payloadSubcarriers)*Int(log2(mcs)))
    bitDemappingQAM!(bitRx, mcs, qamEq[:]);
    # --- Apply De-Interleaver
    bitDeinterl = similar(bitRx)
    deinterleave!(bitDeinterl, bitRx)
    # --- Hamming decoder
    bitDec = zeros(Int8,nbBitsUncoded)
    hammingDecode!(bitDec, bitDeinterl);

    return qamEq,bitDec,chanCoeffs


end

