module RobotClassifier

using Flux, OneHotArrays, Images, FileIO, BSON, Plots, Luxor, MLDataUtils, Cairo, Colors, ImageMagick, ImageIO, CUDA
using .Threads: @threads
#using CUDA  # Remove if not using GPU

export Config, CONFIG, generate_dataset, train_model, visualize_results, test_generation

include("config.jl")
include("dataset/generator.jl")
include("dataset/loader.jl")
include("model.jl")
include("train.jl")
include("visualization.jl")

end # module