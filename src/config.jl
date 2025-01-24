struct Config
    image_size::Tuple{Int,Int}
    batch_size::Int
    learning_rate::Float64
    epochs::Int
    train_split::Float64
    n_samples::Int
end

const CONFIG = Config(
    (128, 128),
    32,
    1e-3,
    30,
    0.8,
    500
)