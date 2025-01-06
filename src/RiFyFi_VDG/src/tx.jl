# ---------------------------------------------------- 
# --- Loading configuration  
# ---------------------------------------------------- 
""" getConfig
---  
Return all the necessary internal variable associated to the desired configuration 
# --- Syntax 
 (nFFT,nCP,sizeSymb,mcs,crcSize,sampleRate,carrierFreq,nbBitsUncoded,inputFEC,outputFEC,allocatedSubcarriers,pilotSubcarriers,payloadSubcarriers,pilotVal,pilotTime,packetSize,nbRepeat)= getConfig();
# --- Input parameters 
- []
# --- Output parameters 
- nFFT		                    : FFT size [Int]
- nCP		                    : CP size [Int]
- sizeSymb	                    : Symbol size (nFFT + nCP) [Int]
- mcs		                    : Constellation size [Int] 
- fecType	                    : Nature of encoder [String] 
- crcSize	                    : Size of appended CRC [Int] 
- sampleRate                    : Sampling rate [Float32] 
- carrierFreq                   : Carrier frequency [Float32] 
- nbBitsUncoded                 : Number of bits in a packet  [Int]
- InputFEC                      : Number of payload bit (before coding) [Int]
- outputFEC                     : Number of bit after encoding [Int]
- allocatedSubcarriers          : Frequency allocation (payload+ pilots ) [Array{Int}]
- pilotSubcarriers	            : Frequency allocation of pilots [Array{Int}]
- payloadSubcarriers            : Allocation frequency for payload [Array{Int}]
- pilotVal			            : Value of pilots in frequency domain  [Array{Complex{Float32}}] 
- pilotTime			            : Value of pilot in time domain [Array{Complex{Float32}}] 
# --- 
# v 1.0 - Robin Gerzaguet.
"""
function getConfig()
	# --- Numerology
	nFFT				= 512;
	nCP					= 36;
	sizeSymb 			= nFFT + nCP;
	mcs 				= 4;
	# --- Channel coding
    crcSize	  = 16;
	# --- Radio parameters
	sampleRate			= 5.2608e6
	carrierFreq			= 5.2e9;
	# --- Packet parameters
	packetSize			 = 10;  	# Packet is 10 words
	nbBitsElemUncoded	 = 24; 		# Word is 24 bits 
	nbBitsUncoded 	     = nbBitsElemUncoded * packetSize 
	inputFEC 			 = nbBitsElemUncoded*packetSize+crcSize;
	outputFEC            = inputFEC ÷ 4 * 7
    nbBits = outputFEC
	nbRepeat			 = 1; 
	# --- Frequency allocatio
	# 80 + and 80 -
	# What we want
	nbDesiredPayloadSubcarriers     = Int(floor(nbBits/log2(mcs)));
	nbDesiredPilots 			    = Int(nbDesiredPayloadSubcarriers ÷ 2 );
	nbDesiredSubcarriers 		    = nbDesiredPayloadSubcarriers + nbDesiredPilots;
	# Deduce space between pilots, and subcarrier allocation
	fPSpace 				  = Int(floor(nbDesiredSubcarriers/nbDesiredPilots));
	allocatedSubcarriers 	  = [collect(2:Int(nbDesiredSubcarriers/2)+2);collect(nFFT-Int(nbDesiredSubcarriers/2)+1:nFFT)];
    pilotSubcarriers 		  = allocatedSubcarriers[1:fPSpace:end];
	# Final configuration
	payloadSubcarriers		  = setdiff(allocatedSubcarriers,pilotSubcarriers);
	allocatedSubcarriers 	  = sort([payloadSubcarriers;pilotSubcarriers])
	# ----------------------------------------------------
	# --- Pilot position correction 
	# ---------------------------------------------------- 
	payloadSubcarriers = [3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31, 33, 34, 36, 37, 39, 40, 42, 43, 45, 46, 48, 49, 51, 52, 54, 55, 57, 58, 60, 61, 63, 64, 66, 67, 69, 70, 72, 73, 75, 76, 78, 79, 81, 82, 84, 85, 87, 88, 90, 91, 93, 94, 96, 97, 99, 100, 102, 103, 105, 106, 108, 109, 111, 112, 114, 115, 117, 118, 120, 121, 123, 124, 126, 127, 129, 130, 132, 133, 135, 136, 138, 139, 141, 142, 144, 145, 147, 148, 150, 151, 153, 154, 156, 157, 159, 160, 162, 163, 165, 166, 168, 169, 346, 347, 349, 350, 352, 353, 355, 356, 358, 359, 361, 362, 364, 365, 367, 368, 370, 371, 373, 374, 376, 377, 379, 380, 382, 383, 385, 386, 388, 389, 391, 392, 394, 395, 397, 398, 400, 401, 403, 404, 406, 407, 409, 410, 412, 413, 415, 416, 418, 419, 421, 422, 424, 425, 427, 428, 430, 431, 433, 434, 436, 437, 439, 440, 442, 443, 445, 446, 448, 449, 451, 452, 454, 455, 457, 458, 460, 461, 463, 464, 466, 467, 469, 470, 472, 473, 475, 476, 478, 479, 481, 482, 484, 485, 487, 488, 490, 491, 493, 494, 496, 497, 499, 500, 502, 503, 505, 506, 508, 509, 511, 512];
	pilotSubcarriers = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74, 77, 80, 83, 86, 89, 92, 95, 98, 101, 104, 107, 110, 113, 116, 119, 122, 125, 128, 131, 134, 137, 140, 143, 146, 149, 152, 155, 158, 161, 164, 167, 170, 345, 348, 351, 354, 357, 360, 363, 366, 369, 372, 375, 378, 381, 384, 387, 390, 393, 396, 399, 402, 405, 408, 411, 414, 417, 420, 423, 426, 429, 432, 435, 438, 441, 444, 447, 450, 453, 456, 459, 462, 465, 468, 471, 474, 477, 480, 483, 486, 489, 492, 495, 498, 501, 504, 507, 510];
	allocatedSubcarriers = sort([pilotSubcarriers;payloadSubcarriers])
    # pilotVal .= 0;
    pilotVal = zeros(ComplexF32,nFFT)
    pilotVal[pilotSubcarriers] .= 1;
    #pilotVal[pilotSubcarriers] .= 1+1im;
    
    # pilotVal= ComplexF32.(genZCSequence(nFFT,pilotSubcarriers,1,0)) # A changer (idx autre que 1) si on veux une séquence de pilote différente pour chaque radio 

	# Metrics
    # We hardcode them to be sure MVector can be used @
	nbPayloadSubcarriers 	  = 224
	nbPilotSubcarriers 		  = 113
	nbSubcarriers			  = nbPayloadSubcarriers + nbPilotSubcarriers

	# In time domain
	pilotSeq 			      = reshape(pilotVal,nFFT,1);
	pilotTime			      = ofdmSigGen(pilotSeq,nFFT,nCP,collect(1:nFFT));
	# ---  Return 
	return (nFFT,nCP,sizeSymb,mcs,crcSize,sampleRate,carrierFreq,nbBitsUncoded,inputFEC,outputFEC,allocatedSubcarriers,pilotSubcarriers,payloadSubcarriers,pilotVal,pilotTime,packetSize,nbRepeat,nbPayloadSubcarriers,nbPilotSubcarriers) 
end

""" Apply OFDM modulator to generate nbSymb based on configuration from `getConfig`
"""
function tx(nbSymb)
    # --- Config 
    (nFFT,nCP,sizeSymb,mcs,crcSize,sampleRate,carrierFreq,nbBitsUncoded,inputFEC,outputFEC,allocatedSubcarriers,pilotSubcarriers,payloadSubcarriers,pilotVal,pilotTime,packetSize,nbRepeat) = getConfig()
    # --- Init buffers for binary and QAM data
    (bitSeq, bitCRC, bitEnc, bitInterl, qamPayload, qamSeq) = init_qamMod()
    # --- Init modulator 
    (sigId, container_modulator...) = init_modulator()
    # sigAll = Vector{eltype(sigId)}(undef,0)
    sigAll = zeros(eltype(sigId),nbSymb*sizeSymb)
    qamAll = zeros(eltype(qamSeq),length(payloadSubcarriers),size(qamSeq,2)*nbSymb)
    bitAll = zeros(eltype(bitSeq),length(bitSeq),nbSymb)
    # --- Iterative generation
    for iN = 1  : 1 : nbSymb 
        # --- Generate one packet (i.e one symbol)
        genBitSequence!(bitSeq,nbBitsUncoded);
        # --- Adding CRC
        addCRC!(bitCRC,bitSeq);
        # --- Apply FEC encoder
        hammingEncode!(bitEnc, bitCRC);
        # --- Interleaver 
        interleave!(bitInterl,bitEnc);
        # --- Map to MCS
        bitMappingQAM!(qamPayload,mcs, bitInterl);
        # ----------------------------------------------------
        # --- Payload generation
        # ----------------------------------------------------
        qamSeq[payloadSubcarriers] 	.= qamPayload;
        ofdmMod!(sigId, qamSeq, container_modulator...)
        # ----------------------------------------------------
        # --- Push into main container 
        # ---------------------------------------------------- 
        sigAll[ (iN-1)*sizeSymb.+(1:sizeSymb)] .= sigId
        qamAll[:,iN] .= qamPayload
        bitAll[:,iN] .= bitSeq
    end
    return (sigAll,qamAll,bitAll)
end


# ----------------------------------------------------
# --- CRC 
# ---------------------------------------------------- 
const gx = [1;1;0;0;0;0;0;0;0;0;0;0;0;0;1;0;1]; 

function addCRC!(out,data)
    crcSize = 16
	lenR = length(data);
	out .= 0;
	out[1:lenR] .= data;
	@inbounds @simd for i ∈ 1 : lenR 
		if out[i] == 1 
			for n ∈ 0: crcSize
				out[i+n] = xor(out[i+n],gx[1+n]);
			end		
		end
	end
	@inbounds @simd for i ∈ 1 : lenR 
		out[i] = data[i];
	end
end
function addCRC(data::Vector{T}) where T 
    out = zeros(T,length(data) + 16)
    addCRC!(out,data)
    return out
end


function checkCRC(data0)
    crcSize = 16
	data = copy(data0)
	# gx = getCRCPoly(crcSize);
	lenR = length(data);    # length of the received codeword
	# lenGW = length(gx);  # length of the generator
	# @show lenGW, crcSize
	@inbounds @simd  for i = 1 : lenR - crcSize
		if data[i] == 1
			# data[i:i+lenGW-1] = xor.(data[i:i+lenGW-1],gx);
			for n ∈ 0: crcSize
				data[i+n] = xor(data[i+n],gx[1+n]);
			end
		end
	end
	# syndrome is now equal to the remainder of xor division
	syndrome = data[ lenR - crcSize + 1: lenR];
	if all(syndrome.== 0x00);
		err = true;
	else
		err = false;
	end
	return err;
end

# ----------------------------------------------------
# --- Hamming
# ---------------------------------------------------- 
function hammingEncode!(y,x)
    # --- Matrix encoder
    nL	= length(x)÷4;
    # --- 
    @inbounds @simd for iN = 1 : 1 : nL 
        # --- Spacing parameters 
        Δx  = (iN-1)*4;
        Δy  = (iN-1)*7;
        # --- Get 4 bits
        for n ∈ 1 : 4 
            y[ Δy + n] = x[Δx + n];
        end 
        # --- Add parity bits
        y[Δy  + 5] = x[Δx + 2] ⊻ x[Δx + 3] ⊻ x[Δx + 4];
        y[Δy  + 6] = x[Δx + 1] ⊻ x[Δx + 3] ⊻ x[Δx + 4];
        y[Δy  + 7] = x[Δx + 1] ⊻ x[Δx + 2] ⊻ x[Δx + 4];
    end
    return y;
end

function hammingDecode!(x,y)
    nL	= length(y)÷7
    cnt = 0
    @inbounds @simd for n ∈ 1 : 1 : nL 
        # --- Calculate 3 equations to deduce syndrome 
        s0 = y[ (n-1)*7 + 4] ⊻  y[ (n-1)*7 + 5] ⊻ y[ (n-1)*7 + 6] ⊻ y[ (n-1)*7 + 7]
        s1 = y[ (n-1)*7 + 2] ⊻  y[ (n-1)*7 + 3] ⊻ y[ (n-1)*7 + 6] ⊻ y[ (n-1)*7 + 7]
        s2 = y[ (n-1)*7 + 1] ⊻  y[ (n-1)*7 + 3] ⊻ y[ (n-1)*7 + 5] ⊻ y[ (n-1)*7 + 7]
        # --- Syndrome calculation 
        pos = s0 << 2 + s1 << 1 + s2
        # --- Switch is syndrome is non-null
        if pos > 0
            bitflip!(y,(n-1)*7 +pos)
            cnt += 1
        end
        for k ∈ 1 : 1 : 4
            x[(n-1)*4 + k] = y[(n-1)*7 + k]
        end
    end
    return cnt
end

function hammingDecodeFull!(x,y)
    nL	= length(y)÷7
    H = [0 0 0 1 1 1 1; 0 1 1 0 0 1 1; 1 0 1 0 1 0 1]
    cnt = 0
    for n ∈ 1 : 1 : nL 
        tmp = @views y[(n-1)*7 .+ (1:7)]
        syndrome = mod.( H * tmp,2)
        pos = sum( syndrome .* [4;2;1])
        if pos != 0
            bitflip!(tmp,pos);
            cnt += 1
        end
        x[(n-1)*4 .+ (1:4)] = tmp[1:4];
    end
    return cnt
end





function hammingDecode(y::Vector{T}) where T
    x = zeros(T,length(y)÷7*4);
    hammingDecode!(x,y);
    return x;
end
function hammingEncode(x::Vector{T}) where T
    y = zeros(T,length(x)÷4*7) 
    hammingEncode!(y,x)
    return y
end

@inline function bitflip!(in,index)
    in[index] == 1 ? in[index] = 0 : in[index]=1;
end

# ----------------------------------------------------
# --- Interleaver 
# ---------------------------------------------------- 
const interl = 1 .+ [0;64;128;192;256;320;384;1;65;129;193;257;321;385;2;66;130;194;258;322;386;3;67;131;195;259;323;387;4;68;132;196;260;324;388;5;69;133;197;261;325;389;6;70;134;198;262;326;390;7;71;135;199;263;327;391;8;72;136;200;264;328;392;9;73;137;201;265;329;393;10;74;138;202;266;330;394;11;75;139;203;267;331;395;12;76;140;204;268;332;396;13;77;141;205;269;333;397;14;78;142;206;270;334;398;15;79;143;207;271;335;399;16;80;144;208;272;336;400;17;81;145;209;273;337;401;18;82;146;210;274;338;402;19;83;147;211;275;339;403;20;84;148;212;276;340;404;21;85;149;213;277;341;405;22;86;150;214;278;342;406;23;87;151;215;279;343;407;24;88;152;216;280;344;408;25;89;153;217;281;345;409;26;90;154;218;282;346;410;27;91;155;219;283;347;411;28;92;156;220;284;348;412;29;93;157;221;285;349;413;30;94;158;222;286;350;414;31;95;159;223;287;351;415;32;96;160;224;288;352;416;33;97;161;225;289;353;417;34;98;162;226;290;354;418;35;99;163;227;291;355;419;36;100;164;228;292;356;420;37;101;165;229;293;357;421;38;102;166;230;294;358;422;39;103;167;231;295;359;423;40;104;168;232;296;360;424;41;105;169;233;297;361;425;42;106;170;234;298;362;426;43;107;171;235;299;363;427;44;108;172;236;300;364;428;45;109;173;237;301;365;429;46;110;174;238;302;366;430;47;111;175;239;303;367;431;48;112;176;240;304;368;432;49;113;177;241;305;369;433;50;114;178;242;306;370;434;51;115;179;243;307;371;435;52;116;180;244;308;372;436;53;117;181;245;309;373;437;54;118;182;246;310;374;438;55;119;183;247;311;375;439;56;120;184;248;312;376;440;57;121;185;249;313;377;441;58;122;186;250;314;378;442;59;123;187;251;315;379;443;60;124;188;252;316;380;444;61;125;189;253;317;381;445;62;126;190;254;318;382;446;63;127;191;255;319;383;447]


function interleave!(y,x)
    @assert size(x) == size(interl) "Interlaved data should be same length as interleaver"
    @assert size(x) == size(y) "Error in input and ouput size in interleaver stage";
    @inbounds @simd for n ∈ (1:length(x))
        y[interl[n]] = x[n];
    end
end


function interleave(x::Vector{T}) where T 
    y = zeros(T,length(x))
    interleave!(y,x)
    return y 
end


function deinterleave!(y,x)
    @assert size(x) == size(interl) "Interlaved data should be same length as interleaver"
    @assert size(x) == size(y) "Error in input and ouput size in interleaver stage";
    @inbounds @simd for n ∈ (1:length(x))
        y[n] = x[interl[n]];
    end
end


function deinterleave(x::Vector{T}) where T 
    y = zeros(T,length(x))
    deinterleave!(y,x)
    return y 
end


# ----------------------------------------------------
# --- Modulation 
# ---------------------------------------------------- 
function init_qamMod()
	# --- Get configuration 
(nFFT,nCP,sizeSymb,mcs,crcSize,sampleRate,carrierFreq,nbBitsUncoded,inputFEC,outputFEC,allocatedSubcarriers,pilotSubcarriers,payloadSubcarriers,pilotVal,pilotTime,packetSize,nbRepeat)= getConfig()
    # --- Bit sequence generation
    bitSeq            = zeros(UInt8, nbBitsUncoded);
    # --- CRC adder 
    bitCRC      = zeros(UInt8, nbBitsUncoded + crcSize);
    # --- Encoder 
    bitEnc      = zeros(UInt8, (nbBitsUncoded + crcSize) ÷ 4 * 7)
    # --- Interleaver 
    bitInterl = similar(bitEnc);
    # --- Init matrix of QAM symbol
    qamPayload  = zeros(Complex{Float32}, length(payloadSubcarriers));
    qamAll  		  = zeros(Complex{Float32}, nFFT);
    qamAll[pilotSubcarriers] .= @view pilotVal[pilotSubcarriers];
    # --- Full return 
    return (bitSeq, bitCRC, bitEnc, bitInterl, qamPayload, qamAll)
end

    


function init_modulator()
	# --- Get configuration 
(nFFT,nCP,sizeSymb,mcs,crcSize,sampleRate,carrierFreq,nbBitsUncoded,inputFEC,outputFEC,allocatedSubcarriers,pilotSubcarriers,payloadSubcarriers,pilotVal,pilotTime,packetSize,nbRepeat)= getConfig()
    # --- OFDM 
    rangeFFT  = 1:nFFT;
    qamAll  		  = zeros(Complex{Float32}, nFFT);
    sigId     = zeros(Complex{Float32}, nFFT + nCP);
    sigRadio  = zeros(Complex{Cfloat}, nFFT + nCP);
    cacheSig  = zeros(Complex{Float32}, nFFT);
    planMod   = plan_ifft(qamAll;flags=FFTW.PATIENT); ;
    sigAll    = Vector{Complex{Float32}}(undef,0);
    # Full retirn 
    # use as (sigId, container_modulator)
    return (sigId, nFFT, nCP, planMod, cacheSig)
end

function ofdmMod!(sigId,qamAll,nFFT,nCP,planMod,cacheSig)
    # --- IFFT 
    mul!(cacheSig,planMod,qamAll);
    # --- Inserting CP 
    @inbounds @simd for n ∈ 1:nCP
        sigId[n] = cacheSig[nFFT-nCP+n];
    end
    # --- Copy FFT
    @inbounds @simd for n ∈ 1:nFFT
        sigId[nCP+n] = cacheSig[n];
    end
end
