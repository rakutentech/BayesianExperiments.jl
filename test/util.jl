using Test

@testset "unnest(arr)" begin
    arr1 = [1, 2, 3]
    arr2 = [4, 5, 6]
    @tset unnset([arr1, arr2]) [1 2 3; 4 5 6]
end