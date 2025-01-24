using Flux: onehotbatch, crossentropy, params, ADAM, DataLoader
using BSON
using ProgressBars

mutable struct TrainingHistory
    train_loss::Vector{Float32}
    val_loss::Vector{Float32}
    train_acc::Vector{Float32}
    val_acc::Vector{Float32}
end

function evaluate_model(model, val_loader)
    total_loss = 0.0f0
    total_correct = 0
    total_samples = 0
    
    for (x, y) in val_loader
        ŷ = model(x)
        total_loss += crossentropy(ŷ, y)
        total_correct += sum(onecold(ŷ) .== onecold(y))
        total_samples += size(y, 2)
    end
    
    (loss=total_loss / length(val_loader), acc=total_correct / total_samples)
end

function train_model(config::Config)
    model = create_model() |> gpu
    opt = ADAM(config.learning_rate)
    history = TrainingHistory([], [], [], [])
    
    # Load data
    X, y = load_dataset("data/train", config)
    X = gpu(X)
    y = gpu(y)
    
    # Split data
    (train_idx, val_idx) = splitobs(1:size(X, 4), at=config.train_split)
    
    train_loader = DataLoader((X[:,:,:,train_idx], y[:,train_idx]), 
                            batchsize=config.batch_size, shuffle=true)
    val_loader = DataLoader((X[:,:,:,val_idx], y[:,val_idx]), 
                          batchsize=config.batch_size)
    
    best_val_acc = 0.0f0
    
    @info "Starting training..."
    for epoch in 1:config.epochs  # Regular for loop instead of @epochs macro
        progress = ProgressBar(train_loader)
        epoch_loss = 0.0f0
        correct = 0
        total = 0
        
        for (x, y) in progress
            x = gpu(x)
            y = gpu(y)
            
            # Augmentation
            x = x .* (0.8f0 .+ 0.4f0 .* rand(Float32, 1, 1, 1, size(x, 4)))
            x = circshift(x, (0, rand(-5:5), 0, 0))
            
            grads = gradient(params(model)) do
                ŷ = model(x)
                loss = crossentropy(ŷ, y)
                epoch_loss += loss
                correct += sum(onecold(ŷ) .== onecold(y))
                total += size(y, 2)
                return loss
            end
            
            Flux.update!(opt, params(model), grads)
        end
        
        # Validation
        val = evaluate_model(model, val_loader)
        
        # Record history
        push!(history.train_loss, epoch_loss / length(train_loader))
        push!(history.train_acc, correct / total)
        push!(history.val_loss, val.loss)
        push!(history.val_acc, val.acc)
        
        # Save best model
        if val.acc > best_val_acc
            best_val_acc = val.acc
            BSON.@save "models/best_model.bson" model
        end
        
        @info "Epoch $epoch" train_loss=history.train_loss[end] train_acc=history.train_acc[end] val_loss=val.loss val_acc=val.acc
    end
    
    model, history
end