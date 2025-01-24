using Flux

function create_model()
    return Chain(
        # Input: 128x128 RGB
        Conv((3, 3), 3 => 32, relu, pad=(1,1)),
        BatchNorm(32),
        MaxPool((2,2)),
        
        Conv((3, 3), 32 => 64, relu, pad=(1,1)),
        BatchNorm(64),
        MaxPool((2,2)),
        
        Conv((3, 3), 64 => 128, relu, pad=(1,1)),
        BatchNorm(128),
        MaxPool((2,2)),
        
        Flux.flatten,
        Dense(128*16*16, 256, relu),
        Dropout(0.5),
        Dense(256, 2),
        softmax
    )
end