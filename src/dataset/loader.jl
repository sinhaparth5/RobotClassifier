using Flux, Images, FileIO, MLDataUtils

function preprocess_image(img, size)
    img = imresize(img, size)
    img = Float32.(channelview(RGB.(img)))
    img = permutedims(img, (3, 2, 1))  # Correct dimension order
end

function load_dataset(path::String, config::Config)
    classes = readdir(path)
    images = []
    labels = []
    
    for (label_idx, class) in enumerate(classes)
        class_path = joinpath(path, class)
        @info "Loading images from $class_path"
        
        for img_file in readdir(class_path)
            full_path = joinpath(class_path, img_file)
            try
                # Use Images.jl directly with format hint
                img = load(full_path; format=PNG)
                img = preprocess_image(img, config.image_size)
                push!(images, img)
                push!(labels, label_idx)
            catch e
                @warn "Removing corrupted file $full_path: $e"
                rm(full_path; force=true)
            end
        end
    end
    
    @info "Loaded $(length(images)) valid images"
    X = cat(images..., dims=4)
    y = Flux.onehotbatch(labels, 1:length(classes))
    (X, y)
end