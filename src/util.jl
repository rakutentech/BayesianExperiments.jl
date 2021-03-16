"""
    catbyrow(arr)

Convert an array of array to a 2-dimenional matrix, the inner vector 
is transposed as concatenated as rows.
"""
catbyrow(arr) = vcat(transpose(arr)...);

"""
    toparameter(parameters)

Convert a parameter vector of length 1 to a single parameter.
"""
function toparameter(parameters::Vector{Symbol}) 
    length(parameters) == 1 || error("Number of parameters should be 1")
    return parameters[1]
end


effsamplesize(n1, n2) = 1/(1/n1 + 1/n2)

dofpooled(n1, n2) = n1 + n2 - 2

function dofwelch(sd1, sd2, n1, n2)
    se1_sq = sd1^2/n1
    se2_sq = sd2^2/n2
    (se1_sq + se2_sq)^2 / (se1_sq^2/(n1-1) + se2_sq^2/(n2-1))
end

pooledsd(sd1, sd2, n1, n2) = sqrt(((n1-1)*sd1^2 + (n2-1)*sd2^2) / (n1 + n2 - 2))

zstat(x̄, μ0, σ, n) = sqrt(n)*(x̄ - μ0)/σ