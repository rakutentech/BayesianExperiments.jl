using Test

@testset "unnest(arr)" begin
    arr1 = [1, 2, 3]
    arr2 = [4, 5, 6]
    @test unnest([arr1, arr2]) == [1 2 3; 4 5 6]
end