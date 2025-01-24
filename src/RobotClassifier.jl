module RobotClassifier

using Flux, Images, FileIO, BSON, Plots, Luxor, MLDataUtils
using .Threads: @threads
#using CUDA  # Remove if not using GPU

export Config, CONFIG, generate_dataset, train_model, visualize_results

include("config.jl")
include("data/generator.jl")
include("data/loader.jl")
include("model.jl")
include("train.jl")
include("visualization.jl")

end # module