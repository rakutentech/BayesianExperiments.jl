"""
    NormalInverseGamma{T<:Real} <: ContinuousUnivariateDistribution

Normal Inverse Gamma distribution. Adapted from the `NormalInverseGamma` in the
[ConjugatePriors.jl](https://github.com/JuliaStats/ConjugatePriors.jl/blob/master/src/normalinversegamma.jl)
package.
"""
struct NormalInverseGamma{T <: Real} <: ContinuousUnivariateDistribution
    μ::T
    v::T 
    α::T # shape 
    θ::T # scale
    function NormalInverseGamma{T}(μ::T, v::T, α::T, θ::T) where T <: Real
    	v > zero(v) && α > zero(α) && θ > zero(θ) || error("v, α and θ must be positive")
    	new{T}(T(μ), T(v), T(α), T(θ))
    end
end

function NormalInverseGamma(μ::Real, v::Real, α::Real, θ::Real)
    T = promote_type(typeof(μ), typeof(v), typeof(α), typeof(θ))
    return NormalInverseGamma{T}(T(μ), T(v), T(α), T(θ))
end

function rand(d::NormalInverseGamma, numsamples::Int)
    sig2_list = rand(InverseGamma(d.α, d.θ), numsamples)
    mu_list = [rand(Normal(d.μ, sqrt(sig2 * d.v))) for sig2 in sig2_list]
    return mu_list, sig2_list
end