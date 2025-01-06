export addNonLinearPA_memory
export initNonLinearPAmemory

# This structure should be integrated in RFImpairmentsModels as a <: PowerAmplifier
struct Memory_PowerAmplifier 
    coefficients::Array{ComplexF64}  # The array of all the coefficients (from Mikko)
    order::Int # Non linear order 
    memory::Vector{Int} # Memory effect size LPA 
    backoff::Float64  # Power backoff (to be mirrored by Matlab implem)
    bandwidth::Float64 # Band for wich the PA is aligned
    Gain_tx::Int64
end

function initNonLinearPAmemory(;coefficients,Gain_tx,order,memory,bandwidth,backoff)
    coefficientsComp = zeros(ComplexF64,order)
    for i= 1 :1 : order
        coefficientsComp[i] = coefficients[i]["re"] + im *coefficients[i]["im"]

    end 
    coefficients = coefficientsComp
   pa = Memory_PowerAmplifier(coefficients,order,memory,backoff,bandwidth,Gain_tx)
    return pa
end

function setup_nonLinearPA_memory(dict,indexRadio)
  with_nonLinearPA = @loadKey dict "with_nonLinearPA" false 
  if with_nonLinearPA
  file= matopen("../My_Signal_Synthesis/PA_memory_models.mat")
  parameters=read(file)["parameters"]
  param = parameters[indexRadio];
  order = Int(param["pa"]["P"]); # nonlinearity order of the PA
  coefficients = zeros(ComplexF64,order)
  coefficients=param["pa"]["coeff"][:,1]
  memory = param["pa"]["Lpa"][1,:]; # memory order of the PA
  Gain_tx = 1; # Gain à 1 pour toute les radio c'est plus simple
  backoff =5
  bandwidth= param["Fs"]
  pa = Memory_PowerAmplifier(coefficients,order,memory,backoff,bandwidth,Gain_tx)
  return pa
  else 
  # No PA, linear model 
    return initNonLinearPA(:Linear)
  end
end

#pa = parameters[r]["pa"] # on réccupère les paramètres du PA de la radio numéro r
    
#memory          = [1,2,2,2,2]
#order           = 9
#backoff         = 10 # In dB
#bandwidth       = 120e6 
#pa_coefficients = zeros(ComplexF64,sum(memory)) # To be changed with correct coefficients



# Init the structure 
#pa = Memory_PowerAmplifier(pa_coefficients,order,memory,backoff,bandwidth)



function addNonLinearPA_memory(x::AbstractVector,pa::Memory_PowerAmplifier,r)
    # Assuming that input signal is 20Mhz we need to upsampled it 
    # -> We use the classic DSP.jl policy here
    v = Int(pa.bandwidth ÷ 20e6)
    y = resample(x,v)
    # Apply the memory PA
    y = memoryPA(y,pa,r)
    # Downsample the signal back to what we need 
    z = resample(y,1/v)
    return z
end


function memoryPA(input,pa::Memory_PowerAmplifier,r)
   
    # A changer pour réccupére les donner la la structure pa 
    # Memory polynomial PA model
    L_in = length(input);
    M = pa.memory; # memory order of the PA
    P = pa.order; # nonlinearity order of the PA   
    backoff =pa.backoff

    # Scale input power according to the specified back-off  
    Pin = mean(abs.(input).^2);  # store the input power, and scale back to it after the TX model
    scale_input = 1/sqrt(10^(backoff/10)*Pin); # 0 dB back-off corresponds to input power=1
    input = scale_input*input;
    
   
    # Generate static basis functions
    PHI = zeros(ComplexF64,L_in,Int((P+1)/2));
    for p=1:Int((P+1)/2)
        PHI[:,p] = input.*abs.(input).^(2*(p-1)); # puissance paire de 0 à 8
        # PHI de(:p) de 0 à 9 par pas de 2 autant de ligne que taille de input,
        # et 5 colones (1,3,5,7,9) non linéarité d'odre 9 avec 5 composantes
        # (puissance impaires) (x , x^3 , x^5 , ... suivant les colonnes)
    end
    
    # Generate memory terms
    R = zeros(ComplexF64,Int(L_in+maximum(M)-1),Int(sum(M)));
    for pp = 1:Int((P+1)/2)
        for m = 1: Int(M[pp])
            R[m:L_in+m-1,Int(sum(M[1:pp-1])+m)] = PHI[1:L_in,pp]; # sum faut juste séléctionner la colonne suivantes 
        end
    end
    
    output = R[1:L_in,:] * pa.coefficients * (1.05-r*0.005); # produit scalaire (multiplication et somme)
    
    # TX specific gain (unity gain on average)
    output = pa.Gain_tx * output;
    
    # Scale power back to original
    output = 1/scale_input * output;
    return output
end



# ----------------------------------------------------
# --- Example 
# ---------------------------------------------------- 
# Emulate an OFDM signal 
# x,_ = tx(100)
# Emulate Non linear memory PA 
# y   = addNonLinearPA(x,pa)
