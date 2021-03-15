@testset "Effect size" begin
    @testset "One Normal Sample" begin
        mu = 0
        sd = 2.5
        n = 100
        stats = NormalStatistics(meanx=mu, sdx=sd, n=n)
        @test effectsize(stats) == mu / sd 

        mu0 = 0.3
        @test effectsize(stats, μ0=mu0) == (mu-mu0)/sd
    end

    @testset "Two samples" begin
        # Example taken from "lsr" cohensD function
        gradesA = [55, 65, 65, 68, 70]
        gradesB = [56, 60, 62, 66]
        stats = TwoSampleStatistics(NormalStatistics(gradesA), NormalStatistics(gradesB))
        @test isapprox(effectsize(stats), 0.699892)
    end
end

@testset "Normal Statistics batch update" begin
    Random.seed!(12)
    data1 = rand(Normal(0.01, 1), 100_000)
    data2 = rand(Normal(0.03, 2), 100_000);
    s1 = NormalStatistics(data1)
    s2 = NormalStatistics(data2)
    stats1 = update!(s1, s2)
    stats2 = NormalStatistics([data1;data2])

    @test stats1.n == stats2.n
    @test isapprox(stats1.meanx, stats2.meanx, atol=0.0001)
    @test isapprox(stats1.sdx, stats2.sdx, atol=0.0001)
end

@testset "Merge two sample statistics" begin
    @testset "Two samples with equal sizes and variances" begin
        stats1= NormalStatistics(meanx=0.8, sdx=1.32, n=100)
        stats2= NormalStatistics(meanx=0.85, sdx=1.32, n=100)
        twostats = TwoSampleStatistics(stats1, stats2)
        stats = merge(twostats)
        @test stats.n == effsamplesize(stats1.n, stats2.n)
        @test stats.meanx == stats1.meanx - stats2.meanx
        @test stats.sdx == pooledsd(stats1.sdx, stats2.sdx, stats1.n, stats2.n)
    end
end