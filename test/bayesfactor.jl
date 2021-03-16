@testset "StudentTModel" begin
    @testset "One group" begin
        # This example is taken from 5.2 in the Gitbook "Bayesian Statistics":
        # https://statswithr.github.io/book/hypothesis-testing-with-normal-populations.html
        n = 10
        xbar = 0.0804
        s = 0.0523

        normalstat = NormalStatistics(n=n, meanx=xbar, sdx=0.0523)
        stats = StudentTStatistics(normalstat)
        model = StudentTModel(r=1.0)

        bf = bayesfactor(model, stats)
        @test isapprox(bf, 50.6, rtol=0.01)
    end
end