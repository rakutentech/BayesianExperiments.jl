using Base.Iterators: partition

@testset "ConjugateBernoulli" begin
    # underlying distribution
    n = 100000
    θ = 0.55
    
    Random.seed!(1234)
    data = rand(Bernoulli(θ), n)

    @testset "Model update and posteriros sampling" begin
        model = ConjugateBernoulli(1, 1)
        stats = BernoulliStatistics(s=sum(data), n=n)
        update!(model, stats)
        posterior_samples = samplepost(model, 100_000)
        @test mean(posterior_samples.θ) ≈ 0.55
    end

    @testset "BernoulliStatistics constructor" begin
        stats = BernoulliStatistics(data)
        @test stats.n == n
        @test stats.s / stats.n ≈ mean(data)
    end

    @testset "Online update vs. One shot update" begin
        numsteps = 10
        α = 10
        β = 30
        data_by_step = reshape(data, (10, div(n, numsteps)))
        model_online = ConjugateBernoulli(α, β)
        for i = 1:numsteps
            stats = BernoulliStatistics(data_by_step[i,:])
            update!(model_online, stats)
        end

        model_oneshot = ConjugateBernoulli(α, β)
        stats = BernoulliStatistics(data)
        update!(model_oneshot, stats)

        @test model_online.dist == model_oneshot.dist
    end

end

@testset "ConjugateExponential" begin
    # underlying distribution
    n = 100_000
    Random.seed!(1234)
    θ = 25.0 
    data = rand(Exponential(θ), n)
    
    @testset "Posteriors sampling" begin
        model = ConjugateExponential(0.001, 1000)
        stats = ExponentialStatistics(n=n, x̄=mean(data))
        update!(model, stats)
        posterior_samples = samplepost(model, 100_000)
        @test isapprox(mean(posterior_samples.θ), θ, atol=0.1) 
    end

    @testset "ExponentialStatistics constructor" begin
        stats = ExponentialStatistics(data)
        @test stats.n == n
        @test isapprox(stats.x̄, 25, atol=0.1)
    end

    @testset "Online update vs. One shot update" begin
        numsteps = 10
        α = 2 
        θ = 5 
        data_by_step = reshape(data, (10, div(n, numsteps)))
        model_online = ConjugateExponential(α, θ)
        for i = 1:numsteps
            stats = ExponentialStatistics(data_by_step[i, :]) 
            update!(model_online, stats)
        end

        model_oneshot = ConjugateExponential(α, θ)
        stats = ExponentialStatistics(data)
        update!(model_oneshot, stats)

        @test typeof(model_online.dist) === typeof(model_oneshot.dist)
        @test model_online.dist.α ≈ model_oneshot.dist.α
        @test model_online.dist.θ ≈ model_oneshot.dist.θ
    end
end

@testset "ConjugateNormal" begin
    # underlying distribution
    n = 100_000
    Random.seed!(1234)
    μ = 13.0 
    σ = 2.5
    data = rand(Normal(μ, σ), n)

    @testset "Posteriors sampling" begin
        model = ConjugateNormal(0.0, 1.0, 0.001, 0.001)
        stats = NormalStatistics(data)
        update!(model, stats)
        posterior_samples = samplepost(model, 100_000)
        @test mean(posterior_samples.μ) ≈ μ 
        @test std(posterior_samples.μ) ≈ σ / sqrt(n)
    end

    @testset "NormalStatistics constructor" begin
        stats = NormalStatistics(data)
        @test stats.n == n
        @test isapprox(stats.meanx, μ, atol=0.01)
        @test isapprox(stats.sdx, σ, atol=0.01)
    end

    @testset "Online update vs. One shot update" begin
        numsteps = 10
        data_by_step = reshape(data, (10, div(n, numsteps)))
        model_online = ConjugateNormal(0.0, 10.0, 10., 5.0) 
        for i = 1:numsteps
            stats = NormalStatistics(data_by_step[i, :]) 
            update!(model_online, stats)
        end

        model_oneshot = ConjugateNormal(0.0, 10.0, 10., 5.0) 
        stats = NormalStatistics(data)
        update!(model_oneshot, stats)

        @test typeof(model_online.dist) === typeof(model_oneshot.dist)
        @test model_online.dist.μ ≈ model_oneshot.dist.μ
        @test model_online.dist.v ≈ model_oneshot.dist.v
        @test model_online.dist.α ≈ model_oneshot.dist.α
        @test model_online.dist.θ ≈ model_oneshot.dist.θ
    end
end

@testset "ConjugateLogNormal" begin
    # underlying distribution
    meanlogx = 7.0
    sdlogx = 0.9
    varlogx = sdlogx * sdlogx
    n = 100000
    
    Random.seed!(1234)
    data  = rand(LogNormal(meanlogx, sdlogx), n)

    @testset "Posteriros sampling" begin
        model = ConjugateLogNormal(0.0, 1.0, 0.001, 0.001)
        stats = LogNormalStatistics(n=n, 
            meanlogx=mean(log.(data)), sdlogx=std(log.(data)))
        update!(model, stats)
        
        @test model.dist.μ ≈ 7.0 
        @test model.dist.v ≈ 0 
        @test isapprox(model.dist.α, 50000, rtol=0.01)
        @test isapprox(model.dist.θ, 40525, rtol=0.01)
        
        posterior_samples = samplepost(model, 100_000)
        @test mean(posterior_samples.μ_logx) ≈ meanlogx
        @test mean(posterior_samples.σ²_logx) ≈ varlogx
        @test isapprox(mean(posterior_samples.μ_x), 1645, rtol=0.01)
        @test isapprox(mean(posterior_samples.σ²_x), 3.4e6, rtol=0.01)
    end

    @testset "LogNormalStatistics constructor" begin
        stats = LogNormalStatistics(data)
        @test stats.n == n 
        @test stats.meanlogx ≈ meanlogx 
        @test stats.sdlogx ≈ sdlogx 
    end

    @testset "Online update vs. One shot update" begin
        numsteps = 10
        data_by_step = reshape(data, (10, div(n, numsteps)))
        model_online = ConjugateLogNormal(0.0, 10.0, 10., 5.0) 
        for i = 1:numsteps
            stats = LogNormalStatistics(data_by_step[i, :]) 
            update!(model_online, stats)
        end

        model_oneshot = ConjugateLogNormal(0.0, 10.0, 10., 5.0) 
        stats = LogNormalStatistics(data)
        update!(model_oneshot, stats)

        @test typeof(model_online.dist) === typeof(model_oneshot.dist)
        @test model_online.dist.μ ≈ model_oneshot.dist.μ
        @test model_online.dist.v ≈ model_oneshot.dist.v
        @test model_online.dist.α ≈ model_oneshot.dist.α
        @test model_online.dist.θ ≈ model_oneshot.dist.θ
    end
end