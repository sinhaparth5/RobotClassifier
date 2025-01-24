using Luxor, FileIO

function generate_robot(size::Tuple{Int,Int}, filename::String)
    # Create parent directory if needed
    mkpath(dirname(filename))
    
    # Explicit PNG format specification
    drawing = Drawing(size[1], size[2], filename)
    origin()
    background("white")
    
    # Scale coordinates to image size
    setopacity(1.0)
    
    # Body (centered and scaled)
    setcolor("silver")
    box(O, 60, 60, :fill)  # Centered box
    
    # Eyes (relative to center)
    setcolor("blue")
    circle(Point(-15, -10), 8, :fill)
    circle(Point(15, -10), 8, :fill)
    
    # Neck/Head (centered)
    setcolor("gray")
    line(Point(0, -30), Point(0, -50), :stroke)
    circle(Point(0, -50), 5, :fill)
    
    # Mouth (centered)
    setcolor("black")
    box(Point(0, 10), 40, 5, :fill)
    
    # Explicit finish with validation
    if !finish()
        @error "Failed to save robot image: $filename"
        return false
    end
    return true
end

function generate_non_robot(size::Tuple{Int,Int}, filename::String)
    mkpath(dirname(filename))
    drawing = Drawing(size[1], size[2], filename)
    origin()
    background("white")
    
    # Use normalized coordinates
    setopacity(1.0)
    translate(-size[1]/2, -size[2]/2)  # Work in pixel coordinates
    
    for _ in 1:5
        setcolor(rand(["red", "green", "blue", "yellow"]))
        x = rand(10:size[1]-10)
        y = rand(10:size[2]-10)
        sz = rand(10:20)
        
        if rand() > 0.5
            circle(Point(x, y), sz, :fill)
        else
            rect(x, y, sz, sz, :fill)
        end
    end
    
    if !finish()
        @error "Failed to save non-robot image: $filename"
        return false
    end
    return true
end

function generate_dataset(config::Config)
    dirs = [
        "data/train/robot", "data/train/not_robot",
        "data/val/robot", "data/val/not_robot"
    ]
    mkpath.(dirs)
    
    n_train = floor(Int, config.train_split * config.n_samples)
    n_val = config.n_samples - n_train

    # Generate with progress tracking
    @info "Generating training data..."
    for i in 1:n_train
        success_robot = generate_robot(config.image_size, "data/train/robot/robot_$i.png")
        success_non = generate_non_robot(config.image_size, "data/train/not_robot/not_robot_$i.png")
        if !(success_robot && success_non)
            @error "Failed to generate training sample $i"
        end
    end

    @info "Generating validation data..."
    for i in 1:n_val
        success_robot = generate_robot(config.image_size, "data/val/robot/robot_$i.png")
        success_non = generate_non_robot(config.image_size, "data/val/not_robot/not_robot_$i.png")
        if !(success_robot && success_non)
            @error "Failed to generate validation sample $i"
        end
    end
end