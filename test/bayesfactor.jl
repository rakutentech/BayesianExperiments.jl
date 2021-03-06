@testset "StudentTEffectSize" begin
    @testset "One group" begin
        # This example is taken from 5.2 in the Gitbook "Bayesian Statistics":
        # https://statswithr.github.io/book/hypothesis-testing-with-normal-populations.html
        # R code result:
        # ## Single numerical variable
        # n = 10, y-bar = 0.0804, s = 0.0523
        # (Using Zellner-Siow Cauchy prior:  mu ~ C(0, 1*sigma)
        # (Using Jeffreys prior: p(sigma^2) = 1/sigma^2
        # 
        # Hypotheses:
        # H1: mu = 0 versus H2: mu != 0
        # Priors:
        # P(H1) = 0.5 , P(H2) = 0.5
        # Results:
        # BF[H2:H1] = 50.7757
        # P(H1|data) = 0.0193  P(H2|data) = 0.9807 
        n = 10
        xbar = 0.0804
        s = 0.0523

        normalstat = NormalStatistics(n=n, meanx=xbar, sdx=0.0523)
        stats = StudentTStatistics(normalstat)
        model = StudentTEffectSize(r=1.0)

        bf = bayesfactor(model, stats)
        @test isapprox(bf, 50.6, rtol=0.01)
    end

    @testset "Two Samples" begin
        # Hypotheses:
        # H1: mu_mature mom  = mu_younger mom
        # H2: mu_mature mom != mu_younger mom
        # 
        # Priors: P(H1) = 0.5  P(H2) = 0.5 
        # 
        # Results:
        # BF[H1:H2] = 5.7162
        # P(H1|data) = 0.8511 
        # P(H2|data) = 0.1489 
        # 
        # Posterior summaries for under H2:
        # 95% Cred. Int.: (-4.3386 , 0.8633)

        # statistics from the nc dataset in R's `statsr` package
        # Group 1: mature mom
        # Group 2: young mom
        normalstats = TwoNormalStatistics(
            NormalStatistics(meanx=28.8, sdx=13.5, n=133),
            NormalStatistics(meanx=30.6, sdx=14.3, n=867)
        )
        model = StudentTEffectSize(r=1.0)

        # pooled case
        stats_pooled = StudentTStatistics(normalstats, pooled=true)
        bf12_pooled = 1/bayesfactor(model, stats_pooled)
        @test isapprox(bf12_pooled, 5.45, rtol=0.001)

        # welch case
        stats_welch = StudentTStatistics(normalstats, pooled=false)
        bf12_welch = 1/bayesfactor(model, stats_welch)
        @test isapprox(bf12_welch, 5.03,  rtol=0.001)
    end

    @testset "One sample: Sleep data" begin
        # This example uses the sleep data and "BayesFactor" package in R:
        # https://richarddmorey.github.io/BayesFactor/
        # Environment:
        # R: 4.0.4
        # BayesFactor:0.9.12-4.2
        # "sleep" data in R have records of 10 paired observations
        sleepdata = [-1.2, -2.4, -1.3, -1.3, 0.0, -1.0, -1.8, -0.8, -4.6, -1.4]

        normalstats = NormalStatistics(sleepdata)
        model = StudentTEffectSize()
        stats = StudentTStatistics(normalstats)

        # test the calculation for one-sample t-statistics
        @test isapprox(stats.t,  -4.0621, rtol=0.001)
        @test stats.n == length(sleepdata)
        @test stats.dof == length(sleepdata) - 1

        # test bayes factor calculation
        model = StudentTEffectSize(r=sqrt(2)/2)
        bf = bayesfactor(model, stats)

        @test isapprox(bf, 17.259, rtol=0.001)
    end

    @testset "Two samples: Chicken weights" begin
        # Chicken Weights by Feed Type in R:
        # Environment:
        # R: 4.0.4
        # BayesFactor:0.9.12-4.2
        #
        #> t.test(weight ~ feed, data=chickwts, var.eq=TRUE)
        # 	Two Sample t-test
        #
        # data:  weight by feed
        # t = -2.934, df = 20, p-value = 0.008205
        # alternative hypothesis: true difference in means is not equal to 0
        # 95 percent confidence interval:
        #  -100.17618  -16.92382
        # sample estimates:
        # mean in group horsebean   mean in group linseed 
        #                  160.20                  218.75 
        #
        # >bf = ttestBF(
        #     x=(chickwts%>%filter(feed=="horsebean"))$weight,
        #     y=(chickwts%>%filter(feed=="linseed"))$weight
        #     )
        #
        # Bayes factor analysis
        # --------------
        # [1] Alt., r=0.707 : 5.975741 ??0%
        #
        # Against denominator:
        # Null, mu1-mu2 = 0 
        # ---
        # Bayes factor type: BFindepSample, JZS
        horsebean=[179, 160, 136, 227, 217, 168, 108, 124, 143, 140]
        linseed=[309, 229, 181, 141, 260, 203, 148, 169, 213, 257, 244, 271]

        stats1 = NormalStatistics(horsebean)
        stats2 = NormalStatistics(linseed)
        # Here we assume equal variance
        tstat = StudentTStatistics(TwoNormalStatistics(stats1, stats2))

        @test isapprox(tstat.t, -2.934, rtol=0.001)
        @test tstat.dof == 20 

        model = StudentTEffectSize(r=sqrt(2)/2)
        bf = bayesfactor(model, tstat)
        @test isapprox(bf, 5.9757,  rtol=0.001)
    end
   
end