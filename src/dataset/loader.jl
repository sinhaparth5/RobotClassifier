using Flux, Images, FileIO, Base.Iterators, ImageMagick, ImageIO

function preprocess_image(img, size)
    # Resize image to target size
    img = imresize(img, size)
    # Convert to RGB and normalize to Float32
    img = Float32.(channelview(RGB.(img)))
    # Rearrange dimensions to match Flux's expected format (width, height, channels, batch)
    img = permutedims(img, (3, 2, 1))
    return img
end

function load_dataset(path::String, config::Config)
    classes = readdir(path)
    images = []
    labels = []
    
    for (label_idx, class) in enumerate(classes)
        class_path = joinpath(path, class)
        @info "Loading images from $class_path"
        
        files = readdir(class_path)
        if isempty(files)
            @warn "No images found in $class_path"
            continue
        end
        
        for img_file in files
            full_path = joinpath(class_path, img_file)
            try
                # Use FileIO with explicit format
                img = load(full_path)
                
                # Preprocess image
                processed_img = preprocess_image(img, config.image_size)
                
                push!(images, processed_img)
                push!(labels, label_idx)
            catch e
                @warn "Error loading file $full_path: $e"
            end
        end
    end
    
    if isempty(images)
        error("No valid images found in $path. Please check if dataset was generated correctly.")
    end
    
    @info "Loaded $(length(images)) valid images"
    
    # Convert to batched format
    X = cat(images..., dims=4)
    y = Flux.onehotbatch(labels, 1:length(classes))
    
    return X, y
end