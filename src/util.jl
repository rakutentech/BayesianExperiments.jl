"""
    unnest(arr)

Convert an array of array to a 2-dimenional matrix, 
with the inner array as rows.
"""
unnest(arr) = vcat(transpose(arr)...);

"""
    converttoparameter(parameters)

Convert a parameter vector of length 1 to a single parameter.
"""
function converttoparameter(parameters::Vector{Symbol}) 
    length(parameters) == 1 || error("Number of parameters should be 1")
    parameters[1]
end