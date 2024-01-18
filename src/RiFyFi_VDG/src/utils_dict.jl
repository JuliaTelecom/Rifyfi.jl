
""" Try to load key from dict. Returns default if not found. If default is `missing` and key is not in dict, raised an error
"""
function getKey(dict,key,default=missing)
    if ismissing(default)
        # Try without fallback
        return dict[key]
    else 
        if haskey(dict,key)
            # Key is there, safely returns it 
            return dict[key]
        else
            # Key is not there, returns the default value
            return default
        end
    end
end
macro loadKey(dict,key,default=missing)
    str = :(key = getKey($(esc(dict)),$(esc(key)),$(esc(default))))
    return str
end

""" Convert a dictionnary of symbol with dictionnary whose keys are string. Convert a dictionnary with strings to dictonnary with symbols
"""
_f_helper(x,op) = x # For recursive call
_f_helper(d::AbstractDict,op) = Dict(op(k) => _f_helper(v,op) for (k, v) in d)
string_dict(d::AbstractDict) = _f_helper(d,String)
symbol_dict(d::AbstractDict) = _f_helper(d,Symbol)


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


""" Transforme la matrice des labels en un vecteur d'indice 0-(NbRadios-1) """
function create_bigMat_Labels_Tx(new_bigLabels)
    bigLabels   = zeros(Int,size(new_bigLabels)[2])
    for i in 1:size(new_bigLabels)[2]
        for j in 1:size(new_bigLabels)[1]
            if new_bigLabels[j,i] == 1
                bigLabels[i] = j-1;
            end
        end
    end

    return bigLabels
end
