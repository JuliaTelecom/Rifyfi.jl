
""" Create a configuration for IQ mismatch for a given radio based on configuation in dict 
"""
function setup_IQMismatch(dict,r)
    # --- Loading config
    with_iq_mismatch        = @loadKey dict "with_iq_mismatch" false
    if !with_iq_mismatch
        # No Mismatch, set identity impairments
        g = -Inf
        ϕ = 0
    else 
        base_g = @loadKey dict "base_g" -Inf
        base_ϕ = @loadKey dict "base_ϕ" 0
        iq_random_range_gain    = @loadKey dict "iq_random_range_gain" (nothing,0)
        iq_random_range_phase   = @loadKey dict "iq_random_range_phase" (nothing,0)
        g = base_g   + myRand(iq_random_range_gain...)
        ϕ = base_ϕ   + myRand(iq_random_range_phase...)
    end
    # Calling constructor from RFImpairmentsModels
    return initIQMismatch(g,ϕ)
end

""" Create a configuration for CFO for a given radio based on configuation in dict 
"""
function setup_CFO(dict,r)
    
    with_cfo = @loadKey dict "with_cfo" false 
    sampling_rate = @loadKey dict "sampling_rate" 1
    if with_cfo
        base_cfo = @loadKey dict "base_cfo" 0
        cfo_range = @loadKey dict "cfo_range" (nothing,0)
        cfo = base_cfo + myRand(cfo_range...)
        return initCFO(cfo,sampling_rate)
    else 
        # --- No CFO return identity transform
        return initCFO(0,sampling_rate)
    end
end





""" Create a configuration for phase noise for a given radio based on configuation in dict 
"""
function setup_phaseNoise(dict::Dict{String},r)
    # Seed 
    # seed_models     = @loadKey dict "seed_models" 12345
    # Random.seed!(seed_models + r + 1000)
    with_phase_noise = @loadKey dict "with_phase_noise" false
    if with_phase_noise
        phase_noise_model   = @loadKey dict "phase_noise_model" :Wiener
        phase_noise_base_σ  = @loadKey dict "phase_noise_base_σ" 0
        phase_noise_range_σ = @loadKey dict "phase_noise_range_σ" (nothing,0)
        σ2 = phase_noise_base_σ + myRand(phase_noise_range_σ...)
        return initPhaseNoise(Symbol(phase_noise_model);σ2)
    else
        # No PN, default constructor 
        return initPhaseNoise(:None;σ2=0)
    end
end


""" Create a configuration for AWGN for a given radio based on configuation in dict 
"""
function setup_awgn(dict,r,k)
    with_noise = @loadKey dict "with_awgn" false
    if with_noise 
        awgn_snr_base = @loadKey dict "awgn_snr_base" 80
        awgn_range = @loadKey dict "awgn_range" (nothing,0)
        snr = awgn_snr_base + 1/2*myRand(awgn_range...)
        return snr
    else 
        return 80.0
    end
end

function setup_nonLinearPA(dict,indexRadio)
    with_nonLinearPA = @loadKey dict "with_nonLinearPA" false 
    if with_nonLinearPA
        # pa_models = @loadKey dict "nonLinearPA_models" :Linear
        if !haskey(dict,"nonLinearPA_models")
            # Specific error message as for non linear PA we don't want to infer linear 
            @error "Non linear PA model is set but no model has been provided. Be sure \"nonLinearPA_models\" key is provided in dict"
        end
        pa_models = @loadKey dict "nonLinearPA_models" # No fallback to be sure it crash as we should use indexRadio (covered by previous test)
        #FIXME Code will only works with Saleh
        model = Symbol(pa_models[1])###
        pa_random_range = @loadKey dict "nonLinearPA_random_range" (nothing,0)
        if  model == :Saleh
            pa_base_saleh = @loadKey dict "nonLinearPA_base_saleh" [0,0,0,0]
            # --- Base model for Saleh 
            range_pa = map( x-> x + myRand(pa_random_range...),pa_base_saleh)
            α_AM,β_AM,α_PM,β_PM = range_pa
            return initNonLinearPA(model;α_AM,β_AM,α_PM,β_PM)
        else
            @error "Unsupported PA model"
        end
    else 
        # No PA, linear model 
        return initNonLinearPA(:Linear)
    end
end


function setup_nonLinearPA_control(dict,indexRadio)
    pourcentage =0.1
    with_nonLinearPA = @loadKey dict "with_nonLinearPA" false 
    if with_nonLinearPA
        # pa_models = @loadKey dict "nonLinearPA_models" :Linear
        if !haskey(dict,"nonLinearPA_models")
            # Specific error message as for non linear PA we don't want to infer linear 
            @error "Non linear PA model is set but no model has been provided. Be sure \"nonLinearPA_models\" key is provided in dict"
        end
        pa_models = @loadKey dict "nonLinearPA_models" # No fallback to be sure it crash as we should use indexRadio (covered by previous test)
        #FIXME Code will only works with Saleh
        model = Symbol(pa_models[1])###
        pa_random_range = @loadKey dict "nonLinearPA_random_range" (nothing,0)
        if  model == :Saleh
            pa_base_saleh = @loadKey dict "nonLinearPA_base_saleh" [0,0,0,0]
            # --- Base model for Saleh 
            range_pa = map( x-> x + pourcentage *x ,pa_base_saleh)
            α_AM,β_AM,α_PM,β_PM = range_pa
            return initNonLinearPA(model;α_AM,β_AM,α_PM,β_PM)
        else
            @error "Unsupported PA model"
        end
    else 
        # No PA, linear model 
        return initNonLinearPA(:Linear)
    end
end


""" Be sure we break phase continuity of the phase noise between bursts. PhaseNoiseModel is build to ensure phase continuit between call. As bursts will not necessary be continuous, we should start from a random phase point (between -π and π). This function update the input structure to ensure the next call to addPhaseNoise has no phase continuity.
"""
function randomize_phaseNoise!(s_noise::RFImpairmentsModels.PhaseNoiseModel)
    if s_noise isa RFImpairmentsModels.WienerPhaseNoise
        # Pure starting point for random phase 
        #if r==2
        #    s_noise.ϕ̄ =  mod(randn(Float64),π)-π/2
        #else 
            s_noise.ϕ̄ =  mod(randn(Float64),2π)-π
        #end 
    end
end

""" Apply power reduction to signal to emulate distance. Returns a scaling factor to multiply with the rx signal
"""
function setup_rx_power(dict,r,snr,k)
    
    with_rx_power = @loadKey dict "with_rx_power" false
    if !with_rx_power 
        scale = 1 
    else 
        awgn_snr_base = @loadKey dict "awgn_snr_base" 80
        δ = snr - awgn_snr_base
        scale = 10^(δ/10)
    end
    return scale 
end


""" Setup multipath channel model for radio index r
""" 
function setup_channel(dict,r,k)
    with_channel = @loadKey dict "with_channel" false 
    if !with_channel
        # Identity channel model 
        return [ComplexF32(1)] 
    else 
        # Multipath channel model
        channel_model = @loadKey dict "channel_model" :none   # By default no channel 
        channel_τ_m = @loadKey dict "channel_τ_m" 1           # Max delay spread  
        channel_nb_taps = @loadKey dict "channel_nb_taps"  1  # Number of active taps 
        # We use this to have a constant channel per radio 
        channel_fix_channel = @loadKey dict "channel_fix_channel"  false # By default channel change 
        channel_seed = @loadKey dict "channel_seed" -1                   # Seed used if constant channel 
        if channel_fix_channel && channel_seed != -1
            # Fix seed per radio 
            Random.seed!(channel_seed + r)
        else 
            # Fix seed per realization of channel 
            Random.seed!(channel_seed + r + k)
        end 
        return Augmentation.getChannel(channel_τ_m;model=Symbol(channel_model[r]),nbTaps=channel_nb_taps)
    end
end
     #=   
""" Get one element in the array, randomly
"""
@inline function choose(a::AbstractVector)
    return a[rand(1:end)]
end

"""" Get several (`numb`) elements in the array; all differents
""" 
function choose(a::AbstractVector{T},numb::Number) where T
    @assert numb < length(a) "Unable to select randomly more elements ($numb) than the length of input vector ($(length(a)))"
    # Everyday I shuffling 
    b = shuffle(a)
    # Keep only numb elements
    return b[1:numb]
end


""" Return a pure random CIR sequence with maximal delay spread τ
TODO: Sotchastic model ? 
"""
function getChannel(τ::Number;model=:none,nbTaps=1)
    if model == :none 
        # No channel, no attenuation, no phase => it is 1
        cir = [ComplexF32(1)]
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
    else 
        @error "Unknown channel model: $model is not supported"
    end
    return cir
end

 =#

        
