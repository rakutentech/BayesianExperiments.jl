DataGeneratingDistibutions = Union{Vector{UnivariateDistribution},Vector{Vector{UnivariateDistribution}}}
"""
    convert(::DataGeneratingDistibutions, v)

Converting DataGeneratingDistibutions to specific vector.
"""
function Base.convert(::Type{DataGeneratingDistibutions}, v::Vector{T}) where T <: UnivariateDistribution
    convert(Vector{UnivariateDistribution}, v)
end
function Base.convert(::Type{DataGeneratingDistibutions}, v::Vector{Vector{T}}) where T <: UnivariateDistribution 
    convert(Vector{Vector{UnivariateDistribution}}, v)
end

"""
    Simulation(experiment, parameters, datagendists, maxsteps, onestepsizes, minsteps)

A simulation setup includes the experiment, data generating distributions, max number of steps
and minimum number of steps.
"""
struct Simulation
    experiment::Experiment
    parameters::Vector{Symbol}
    datagendists::DataGeneratingDistibutions
    maxsteps::Integer
    onestepsizes::Union{Vector{Int},Vector{Vector{Int}}}
    minsteps::Integer
    function Simulation(;experiment, datagendists, 
        maxsteps, onestepsizes, minsteps=0, parameters::Union{Vector{Symbol},Nothing}=nothing)
        minsteps <= maxsteps || error("maxsteps should be equal or larger than minsteps")
        if parameters === nothing
            return new(experiment, getdefaultparams(experiment), datagendists, maxsteps, onestepsizes, minsteps)
        else
            return new(experiment, parameters, datagendists, maxsteps, onestepsizes, minsteps)
        end
    end
end

function Base.show(io::IO, simulation::Simulation)
    info = """
    $(simulation.experiment)
    maxsteps  : $(simulation.maxsteps)
    minstpes  : $(simulation.minsteps)
    """
    print(io, typeof(simulation), "\n", info)
end

function updateonce!(experiment::Experiment, datagendists, onestepsizes)
    length(experiment.models) == length(datagendists) || 
        error("Number of models should be equal to number of data gen distributions.")
    length(datagendists) == length(onestepsizes) ||
        error("Number of one step sizes should be equal to number of data gen distributions.")
    for (modelname, datagendist, onestepsize) in zip(experiment.modelnames, datagendists, onestepsizes)
        model = experiment.models[modelname]
        stats = sample_stats(model, datagendist, onestepsize)
        update!(experiment, modelname, stats)
    end
    nothing
end

updateonce!(simulation::Simulation) = updateonce!(simulation.experiment, simulation.datagendists, simulation.onestepsizes)

function runonce(simulation::Simulation, numsamples::Integer)
    experiment = simulation.experiment
    runnum = 0
    metricvals = Vector{Vector{Float64}}()
    winner = nothing
    while experiment.winner === nothing && runnum < simulation.maxsteps
        runnum += 1
        updateonce!(simulation)
        _, metricval = calculatemetrics(experiment, simulation.parameters, numsamples=numsamples)
        push!(metricvals, metricval)
        if runnum < simulation.minsteps
            continue
        else
            winner = decide!(experiment, simulation.parameters, numsamples=numsamples)
        end
    end
    metricvals, winner
end

function record!(winners, winner)
    if winner === nothing
        push!(winners, "nothing")
    else
        push!(winners, winner)
    end
    nothing
end

function runsequential(simulation::Simulation; numsamples=10_000, numsims=50)        
    numsteps = Vector{Int}()
    winners = Vector{String}()
    metricvals = Vector{Vector{Vector{Float64}}}()
    for _ = 1:numsims
        simcopy = deepcopy(simulation)
        metrival, winner = runonce(simcopy, numsamples)
        push!(metricvals, metrival)
        push!(numsteps, length(metrival))
        record!(winners, winner)
    end
    numsteps, winners, metricvals
end

# TODO: speed up using StaticArrays and multiple threads.