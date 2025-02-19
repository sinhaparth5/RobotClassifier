using Flux: onehotbatch, crossentropy, trainable, onecold, ADAM, DataLoader
using Flux
using OneHotArrays
using BSON
using ProgressBars
using CUDA

mutable struct TrainingHistory
    train_loss::Vector{Float32}
    val_loss::Vector{Float32}
    train_acc::Vector{Float32}
    val_acc::Vector{Float32}
end

function evaluate_model(model, val_loader, device)
    model.eval()  # Set to evaluation mode
    total_loss = 0.0f0
    total_correct = 0
    total_samples = 0
    
    for (x, y) in val_loader
        x, y = x |> device, y |> device
        ŷ = model(x)
        total_loss += crossentropy(ŷ, y)
        total_correct += sum(onecold(ŷ) .== onecold(y))
        total_samples += size(y, 2)
    end
    
    model.train()  # Set back to training mode
    return (loss=total_loss / length(val_loader), acc=total_correct / total_samples)
end

function train_model(config::Config)
    # Check GPU availability
    device = CUDA.functional() ? gpu : cpu
    @info "Using device: $(CUDA.functional() ? "GPU" : "CPU")"
    
    try
        # Create model and move to appropriate device
        model = create_model() |> device
        opt = ADAM(config.learning_rate)
        history = TrainingHistory([], [], [], [])
        
        # Create directories
        mkpath("models")
        
        # Load and prepare data
        @info "Loading dataset..."
        X, y = load_dataset("data/train", config)
        X, y = X |> device, y |> device
        
        # Split data
        @info "Preparing data loaders..."
        (train_idx, val_idx) = splitobs(1:size(X, 4), at=config.train_split)
        
        train_loader = DataLoader((X[:,:,:,train_idx], y[:,train_idx]), 
                                batchsize=config.batch_size, shuffle=true)
        val_loader = DataLoader((X[:,:,:,val_idx], y[:,val_idx]), 
                              batchsize=config.batch_size)
        
        best_val_acc = 0.0f0
        patience = 5  # Early stopping patience
        patience_counter = 0
        
        @info "Starting training for $(config.epochs) epochs..."
        for epoch in 1:config.epochs
            # Training phase
            progress = ProgressBar(train_loader)
            epoch_loss = 0.0f0
            correct = 0
            total = 0
            
            for (x, y) in progress
                x, y = x |> device, y |> device
                
                # Augmentation
                x = x .* (0.8f0 .+ 0.4f0 .* rand(Float32, 1, 1, 1, size(x, 4)))
                x = circshift(x, (0, rand(-5:5), 0, 0))
                
                # Use explicit gradient computation
                loss = 0.0f0
                gs = gradient(model) do m
                    ŷ = m(x)
                    loss = crossentropy(ŷ, y)
                    epoch_loss += loss
                    correct += sum(Flux.onecold(ŷ) .== Flux.onecold(y))
                    total += size(y, 2)
                    loss
                end
                
                # Update parameters
                Flux.update!(opt, trainable(model), gs)
                
                # Update progress bar
                set_description(progress, "Epoch $epoch")
            end
            
            # Validation phase
            val_loss = 0.0f0
            val_correct = 0
            val_total = 0
            
            for (x, y) in val_loader
                x, y = x |> device, y |> device
                ŷ = model(x)
                val_loss += crossentropy(ŷ, y)
                val_correct += sum(Flux.onecold(ŷ) .== Flux.onecold(y))
                val_total += size(y, 2)
            end
            
            # Calculate metrics
            train_loss = epoch_loss / length(train_loader)
            train_acc = correct / total
            val_loss = val_loss / length(val_loader)
            val_acc = val_correct / val_total
            
            # Record history
            push!(history.train_loss, train_loss)
            push!(history.train_acc, train_acc)
            push!(history.val_loss, val_loss)
            push!(history.val_acc, val_acc)
            
            # Save best model and check early stopping
            if val_acc > best_val_acc
                best_val_acc = val_acc
                BSON.@save "models/best_model.bson" model
                @info "Saved new best model with validation accuracy: $(round(val_acc * 100, digits=2))%"
                patience_counter = 0
            else
                patience_counter += 1
                if patience_counter >= patience
                    @info "Early stopping triggered after $epoch epochs"
                    break
                end
            end
            
            @info "Epoch $epoch/$((config.epochs))" train_loss=round(train_loss, digits=4) train_acc=round(train_acc, digits=4) val_loss=round(val_loss, digits=4) val_acc=round(val_acc, digits=4)
        end
        
        return model, history
        
    catch e
        @error "Training failed" exception=(e, catch_backtrace())
        rethrow(e)
    end
end