abstract type ExperimentData end
struct BinaryData <: ExperimentData end

abstract type ModelStatistics end

"""
    BernoulliStatistics(numsuccesses, numtrials)

Sample statistics of data generated from Bernoulli distribution.
"""
struct BernoulliStatistics <: ModelStatistics
    s::Int
    n::Int
    function BernoulliStatistics(; s::Int, n::Int)
        # TODO: validate input data
        s <= n ||
            error("Number of trials should be equal or larger than number of successes.")
        return new(s, n)
    end

    function BernoulliStatistics(data::Vector{T}) where {T<:Real}
        return new(sum(data), length(data))
    end
end

struct ExponentialStatistics <: ModelStatistics
    n::Int
    x̄::Real

    function ExponentialStatistics(; n, x̄)
        # TODO: validate input data
        return new(n, x̄)
    end

    function ExponentialStatistics(data::Vector{T}) where {T<:Real}
        return new(length(data), mean(data))
    end
end

struct NormalStatistics <: ModelStatistics
    n::Int
    meanx::Real
    sdx::Real

    function NormalStatistics(; n, meanx, sdx)
        return new(n, meanx, sdx)
    end

    function NormalStatistics(data::Vector{T}) where {T<:Real}
        n = length(data)
        meanx = mean(data)
        sdx = std(data)
        return new(n, meanx, sdx)
    end
end

struct LogNormalStatistics <: ModelStatistics
    n::Int
    meanlogx::Real
    sdlogx::Real

    function LogNormalStatistics(; n, meanlogx, sdlogx)
        return new(n, meanlogx, sdlogx)
    end

    function LogNormalStatistics(data::Vector{T}) where {T<:Real}
        data > zero(data) || error("data must be positive for lognormal data")
        n = length(data)
        logdata = log.(data)
        meanlogx = mean(logdata)
        sdlogx = std(logdata)
        return new(n, meanlogx, sdlogx)
    end
end

getall(stats::NormalStatistics) = (stats.n, stats.meanx, stats.sdx)
getall(stats::LogNormalStatistics) = (stats.n, stats.meanlogx, stats.sdlogx)

"""
update!(stats_old, stats_new)

Batch update for Normal statistics.

## References

- [Batch updates for simple statistics](
https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html)
"""
function update!(stats1::T, stats2::T) where {T<:NormalStatistics}
    n1 = stats1.n
    m1 = stats1.meanx
    s1 = stats1.sdx
    n2 = stats2.n
    m2 = stats2.meanx
    s2 = stats2.sdx

    new_n = n1 + n2
    n1_ratio = n1 / (n1 + n2)
    n2_ratio = n2 / (n1 + n2)
    new_meanx = n1_ratio * m1 + n2_ratio * m2
    new_sdx = sqrt(n1_ratio * s1^2 + n2_ratio * s2^2 + n1 * n2 / (n1 + n2)^2 * (m1 - m2)^2)

    return NormalStatistics(; n=new_n, meanx=new_meanx, sdx=new_sdx)
end