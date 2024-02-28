

"""
sqrtRaisedCosine(N,β,ovS)\\
Returns the Finite Impulse Response of a Square Root Raised Cosine (SRRC) filter.
The filter is defined by its span (evaluated in number of symbol N), its Roll-Off factor and its oversampling factor. The span corresponds to the number of symbol affected by filter before and after the center point. \\
Output is a Complex{Float64} array of size L= 2KN+1 \\
SRRC definition is based on [1]\\
[1]	 3GPP TS 25.104 V6.8.0 (2004-12). http://www.3gpp.org/ftp/Specs/archive/25_series/25.104/25104-680.zip\\ 
Parameters 
- N	  : Symbol span 
- β  : Roll-off factor (Float64)
- ovS	  : Oversampling rate 
"""
function sqrtRaisedCosine(N,β,ovS)
    # --- Final size of filter
    nbTaps	= 2 * N * ovS + 1;
    # --- Init output
    h			= zeros(Float64,nbTaps);
    counter		= 0;
    # --- Iterative SRRC definition
    for k = -N*ovS : 1 : N*ovS
        counter		 = counter + 1;
        if k == 0
            ## First singular point at t=0
            h[counter]  = (1-β) + 4*β/pi;
        elseif abs(k) == ovS / (4*β);
            ## Second possible singular point
            h[counter]  = β/sqrt(2)*( (1+2/pi)sin(pi/(4β))+(1-2/pi)cos(pi/(4β)));
        else
            ## Classic SRRC formulation (see [1])
            h[counter]  = ( sin(pi*k/ovS*(1-β)) + 4β*k/ovS*cos(pi*k/ovS*(1+β))) / (pi*k/ovS*(1- (4β*k/ovS)^2) );
        end
    end
    # --- Normalize power of filter
    #return h / maximum(abs.(h))
    return h / sqrt(sum(abs2.(h)))
end

""" Create a transmitted signal with `nbSymb` symbols with constellation with `mcs` constellation size (with log2(mcs) bits per symbol i.e 4 for QPSK and 16 for QAM-16), filtered by a square root raised cosine with oversampling factor `ovs` and Roll-off factor `β` span on `2N+1` symbols 
Return the generated bit sequence and the complex signal 
""" 
function singleCarrierTx(nbSymb; mcs = 4,ovS = 4, β=0.3, N=6)
    # Number of bits per symbol 
    bps = Int(log2(mcs))
    # Generate bit sequence 
    bitSeq = genBitSequence(bps * nbSymb) 
    # Modulate 
    qamSeq = bitMappingQAM(mcs,bitSeq)
    # Generate filter 
    h = sqrtRaisedCosine(N,β,ovS)
    # Oversample and filter 
    support = zeros(ComplexF64, nbSymb * ovS)
    support[1:ovS:end] .= qamSeq 
    # Filter data 
    dataFiltered = DSP.conv(h,support) 
    return (bitSeq,dataFiltered)
end


function singleCarrierTx(nbSymb)
    mcs = 4
    ovS = 4
    β=0.3 
    N=6
    (bitSeq,dataFiltered) = singleCarrierTx(nbSymb; mcs ,ovS, β, N)
    alpha = N * ovS 
    # size_h = 2 * N * ovS + 1
    x= dataFiltered[1+alpha: end- alpha]
 return x
end 

""" Demodulate the transmitted signal with a synchronisation delay `delay`
Returns the Rx QAm sequence and the decoded bit sequence """ 
function singleCarrierRx(y; delay=0,mcs = 4,ovS = 8, β=0.3, N=6)
    # Filter in Rx 
    h = sqrtRaisedCosine(N,β,ovS)
    dataFiltered = DSP.conv(h,y) 
    # Decision 
    # assuming match filter 
    tails = length(h)-1
    constRx = dataFiltered[1+delay+tails:ovS:end-tails] 
    # Demodulation 
    bitDec = bitDemappingQAM(mcs,constRx) 
    return (bitDec,constRx)
end


# Minimal Chain 
# b,signal = RiFyFi.RiFyFi_VDG.singleCarrierTx(120)
# b̄,qamRx = RiFyFi.RiFyFi_VDG.singleCarrierRx(signal);
# ber = sum(xor.(b̄,b)) / length(b)
