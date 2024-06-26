




"function to add a dynamic CFO "
function add_dynamic_CFO!(y::AbstractVector{Complex{T}},x::AbstractVector,δ::Number,fs::Number,ϕ=0,dyn=0) where {T<:Real}
    # --- Basic array check 
    @assert length(x) == length(y) "Input and output should have same length (here input x is $(length(x)) and pre-allocated output has size $(length(y))"
    # --- CFO pulsation 

   # dynamic = rand(9500:10500)/10000   

#   ppm = 0.1
#   fr = ppm * 2.4 * 1000

   # dynamic = rand(9000:11000)/10000   
   δ = δ + dyn
    ω = δ / fs #* dynamic
    # --- Adding CFO 
    @inbounds @simd for n ∈ eachindex(x)
        y[n] = x[n] * Complex{T}(exp(2im*π*ω*( (n-1) + ϕ)))
    end
end


function add_dynamic_CFO!(x::AbstractVector,cfo::RFImpairmentsModels.CFO,dyn_value)
    add_dynamic_CFO!(x,x,cfo.f,cfo.fs,cfo.ϕ,dyn_value)
end



