# module Tx

"""
Generate LoRa symbols 
signal, bitstsream_preFEC, symbols_preFEC, bitstsream_postFEC, symbols_postFEC = Gen_symb_SF(SF,N,BW,Fs,inverse)
# Input
- SF: Spreading factor of the CSS modulation
- N: Number of block to transmit, a block correspond to SF LoRa symbols
- BW: Bandwidth used for transmission
- Fs: sampling frequency
- inverse: 0 to send upchirp, 1 for downchirp
# Output
- signal: Baseband  signal emitted from the transmitter
- bitstsream_preFEC: randomly generated binary data, it is the data that need to be sent
- symbols_preFEC: Conversion of bitstsream_preFEC into symbols, each symbols correspond to 4 bits
- bitstsream_postFEC: Results of the application of Hamming code and interleaving
- symbols_postFEC: Conversion of bitstsream_postFEC into symbols, each symbols correspond to SF bits

"""
function Gen_symb_SF(SF,N,BW,Fs,inverse)
    total_sym = N*SF;           # Number of LoRa Symbols
    A=rand(4,total_sym);
    A=(A.<1/2); # Binary elements
    bitstsream_preFEC = A;

    InputH = zeros(7,total_sym)
    # Hamming coding (7,4)
    for i ∈ 1:total_sym 
        InputH[:,i] = hamming74Encode(A[:,i])
    end
    OutputFEC = floor.(Int, InputH);

    OutputInter = zeros(Int,N*7,SF);
    for k ∈ 1:N # interleaver
        OutputInter[(k-1)*7 + 1 : 7*k,:] = OutputFEC[:,SF*(k-1) + 1 : SF*k]
    end

    # Symb=gray2dec(OutputInter); # Gray coding
    LoRa_symbols = grayencode.(bin2dec(OutputInter))
    LoRa_symbols_mat=LoRa_symbols';
    signal=[];
    num_samples = Int64(floor(Fs*(2^SF)/BW));  # Number of samples
    for i = 1:(N*7)
        out_sym = LoRa_Modulation(SF,BW,Fs,num_samples,LoRa_symbols_mat[i],inverse);
        signal = append!(signal, out_sym);
    end
    symbols_preFEC = bin2dec(bitstsream_preFEC')
    symbols_postFEC = bin2dec(OutputInter)
    return Complex.(signal),bitstsream_preFEC, symbols_preFEC, OutputInter, symbols_postFEC
end

"""
Do the CSS modulation
"""
function LoRa_Modulation(SF,BW,Fs,num_samples,symbol,inverse)

    #initialization
    phase = 0;
    Frequency_Offset = (Fs/2) - (BW/2);

    shift = symbol;
    out_preamble = zeros(ComplexF64,num_samples);

    for k = 1:num_samples
   
        # output the complex signal
        out_preamble[k] = cos(phase) + 1im*sin(phase);
    
        # Frequency from cyclic shift
        f = BW*shift/(2^SF);
        if(inverse == 1)
               f = BW - f;
        end
    
        # apply Frequency offset away from DC
        f = f + Frequency_Offset;
    
        # Increase the phase according to frequency
        phase = phase + 2*pi*f/Fs;
        if phase > pi
            phase = phase - 2*pi;
        end
    
        # update cyclic shift
        shift = shift + BW/Fs;
        if shift >= (2^SF)
            shift = shift - 2^SF;
        end
    end
    return out_preamble
end


"""
"""
function hamming_mat(total_sym, A)
    InputH = zeros(7,total_sym)
    for i ∈ 1:total_sym #encodage de hamming (ajout des bits de parités: mots de 4 bits --> mots de 7 bits)
        InputH[:,i] = hamming74Encode(A[:,i])
    end
    InputH = floor.(Int, InputH);
    return InputH
end

"""
"""
function hamming74Encode(x)
    # --- Matrix encoder
    # --- Matrix encoder 
    A = UInt8.([ 0 1 1;1 0 1;1 1 0;1 1 1 ]);
    G = [ Matrix{UInt8}([1 0 0 0 ; 0 1 0 0;0 0 1 0;0 0 0 1]) A ]
    # --- Init output  
    nL	= length(x)÷4;
    out	= zeros(UInt8,7);
    y	= zeros(UInt8,nL*7);
    # --- 
    for iN = 1 : 1 : nL 
        # --- Get 4 bits
        subM	  = x[(iN-1)*4 .+ (1:4)];
        # --- Apply encoder 
        out .=  mod.((subM' * G)[:],2);
        # --- Additionnal parity bit 
        parity	  = mod(sum(out),2);
        y[(iN-1)*7 .+ (1:7)] .= out;
    end
    return y;
end

"""
"""
function interleaver(N, SF, InputH)
    InputInter = zeros(Int,N*7,SF);
    for k ∈ 1:N #interleaver
        InputInter[(k-1)*7 + 1 : 7*k,:] = InputH[:,SF*(k-1) + 1 : SF*k]
    end
    return InputInter
end

# end # module



"""" 
g = grayencode(n)\\
Convert the integer `n` as its value with Gray encoding \\
Inputs : 
- n : Input integer 
Outputs : 
- g : Gray-coded integer
Example: grayencode(2) = 3
"""
grayencode(n::Integer) = n ⊻ (n >> 1)


"""
n = graydecode(n) 
Convert the gray encoded word `n` back
Inputs : 
- g : Gray-coded integer
Outputs : 
- n : Input integer 
"""
function graydecode(n::Integer)
    r = n
    while (n >>= 1) != 0
        r ⊻= n
    end
    return r
end

""" 
n = bin2dec(data)
Convert a binary vector into its integer representation. The input should be a vector with the first element the MSB and the last element the LSB. \\
Example : bin2dec([0;0;1]) = 1; bin2dec([1;0;1;0])=10 \\
If the input is a matrix, the conversion is down per line (e.g bin2dec([1 0 1 0 ; 1 1 1 0]) = [10;14]
"""
function bin2dec(data::AbstractMatrix)
    pow2 = [2^(k-1) for k in (size(data,2)):-1:1]
    dataout = [sum(data[k,:] .* pow2) for k ∈ 1:size(data,1)]
    return dataout
end
bin2dec(data::AbstractVector) = bin2dec(data')

"""
Binary representation of Int on a given number of bits. MSB in pos 1 and LSB at end 
"""
function dec2bin(input::Vector{Int},n::Int)
    Output_bin = zeros(Int, length(input), n)
    for i ∈ eachindex(input)
        c = bitstring(input[i])
        data = [Int(c[j]-48) for j ∈ length(c)-(n-1):length(c)]
        Output_bin[i,:] = data
    end
    return Output_bin
end

