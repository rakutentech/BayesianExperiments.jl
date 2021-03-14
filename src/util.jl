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