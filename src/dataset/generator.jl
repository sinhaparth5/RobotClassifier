using Cairo, Colors, Random

const ROBOT_PALETTE = [
    RGB(0.8, 0.8, 0.8),   # Silver body
    RGB(0.0, 0.4, 0.9),   # Blue eyes
    RGB(0.3, 0.3, 0.3),   # Dark accents
    RGB(0.9, 0.2, 0.2)    # Red details
]

const NON_ROBOT_PALETTE = [
    RGB(0.98, 0.5, 0.45),  # Coral
    RGB(0.45, 0.8, 0.75),   # Mint
    RGB(0.95, 0.9, 0.3),    # Yellow
    RGB(0.6, 0.4, 0.8)      # Purple
]

function create_canvas(width, height, filename)
    mkpath(dirname(filename))
    surface = Cairo.CairoImageSurface(width, height, Cairo.FORMAT_ARGB32)
    cr = Cairo.CairoContext(surface)
    Cairo.set_source_rgb(cr, 1, 1, 1)  # White background
    Cairo.paint(cr)
    (surface, cr)
end

function save_canvas(surface, cr, filename)
    Cairo.write_to_png(surface, filename)
    Cairo.destroy(cr)
    Cairo.destroy(surface)
end

function generate_robot_image(config::Config, filename::String)
    try
        width, height = config.image_size
        surface, cr = create_canvas(width, height, filename)
        
        # Body 
        body_radius = min(width, height) ÷ 4
        
        # Draw the main body circle
        Cairo.set_source_rgb(cr, red(ROBOT_PALETTE[1]), green(ROBOT_PALETTE[1]), blue(ROBOT_PALETTE[1]))
        Cairo.arc(cr, width/2, height/2, body_radius, 0, 2π)
        Cairo.fill_preserve(cr)
        
        # Add shading effect
        Cairo.set_source_rgba(cr, 0.7, 0.7, 0.7, 0.3)
        Cairo.arc(cr, width/2 - body_radius/3, height/2 - body_radius/3, body_radius, 0, 2π)
        Cairo.fill(cr)

        # Add highlight
        Cairo.set_source_rgba(cr, 1, 1, 1, 0.2)
        Cairo.arc(cr, width/2 - body_radius/4, height/2 - body_radius/4, body_radius/2, 0, 2π)
        Cairo.fill(cr)

        # Eyes with reflection
        eye_radius = body_radius ÷ 4
        eye_positions = [
            (width/2 - body_radius/1.5, height/2 - eye_radius),
            (width/2 + body_radius/1.5, height/2 - eye_radius)
        ]
        for (x, y) in eye_positions
            # Eye base
            Cairo.set_source_rgb(cr, red(ROBOT_PALETTE[2]), green(ROBOT_PALETTE[2]), blue(ROBOT_PALETTE[2]))
            Cairo.arc(cr, x, y, eye_radius, 0, 2π)
            Cairo.fill(cr)
            
            # Eye highlight
            Cairo.set_source_rgba(cr, 1, 1, 1, 0.8)
            Cairo.arc(cr, x + eye_radius/3, y - eye_radius/3, eye_radius/3, 0, 2π)
            Cairo.fill(cr)
        end

        # Mouth
        Cairo.set_source_rgb(cr, red(ROBOT_PALETTE[4]), green(ROBOT_PALETTE[4]), blue(ROBOT_PALETTE[4]))
        mouth_width = body_radius * 1.2
        Cairo.rectangle(cr, width/2 - mouth_width/2, height/2 + body_radius/2, 
                       mouth_width, body_radius/8)
        Cairo.fill(cr)

        # Decorative elements
        Cairo.set_source_rgb(cr, red(ROBOT_PALETTE[3]), green(ROBOT_PALETTE[3]), blue(ROBOT_PALETTE[3]))
        Cairo.set_line_width(cr, body_radius/20)
        
        # Antennae
        Cairo.move_to(cr, width/2 - body_radius/3, height/2 - body_radius)
        Cairo.line_to(cr, width/2 - body_radius/4, height/2 - body_radius*1.4)
        Cairo.move_to(cr, width/2 + body_radius/3, height/2 - body_radius)
        Cairo.line_to(cr, width/2 + body_radius/4, height/2 - body_radius*1.4)
        Cairo.stroke(cr)

        # Add some circuit-like details
        Cairo.set_line_width(cr, body_radius/40)
        for i in 1:3
            angle = 2π * i / 3
            x = width/2 + cos(angle) * body_radius/2
            y = height/2 + sin(angle) * body_radius/2
            Cairo.move_to(cr, x, y)
            Cairo.line_to(cr, x + cos(angle) * body_radius/4, y + sin(angle) * body_radius/4)
        end
        Cairo.stroke(cr)

        save_canvas(surface, cr, filename)
        return true
    catch e
        @error "Robot generation failed" exception=(e, catch_backtrace())
        return false
    end
end

function generate_non_robot_image(config::Config, filename::String)
    try
        width, height = config.image_size
        surface, cr = create_canvas(width, height, filename)
        rng = MersenneTwister(hash(filename))  # Deterministic randomness

        # Base shape
        base_color = rand(rng, NON_ROBOT_PALETTE)
        Cairo.set_source_rgba(cr, base_color.r, base_color.g, base_color.b, 0.9)
        Cairo.arc(cr, width/2, height/2, min(width, height)/3, 0, 2π)
        Cairo.fill(cr)

        # Organic shapes overlay
        for _ in 1:rand(rng, 3:6)
            color = rand(rng, NON_ROBOT_PALETTE)
            Cairo.set_source_rgba(cr, color.r, color.g, color.b, 0.6)
            
            # Random polygon
            Cairo.move_to(cr, rand(rng, width÷4:3width÷4), rand(rng, height÷4:3height÷4))
            for _ in 1:rand(rng, 4:7)
                Cairo.line_to(cr, rand(rng, 0:width), rand(rng, 0:height))
            end
            Cairo.close_path(cr)
            Cairo.fill(cr)
        end

        # Texture pattern
        Cairo.set_source_rgba(cr, 0, 0, 0, 0.05)
        for _ in 1:100
            x = rand(rng, 0:width)
            y = rand(rng, 0:height)
            Cairo.arc(cr, x, y, rand(rng, 1:3), 0, 2π)
            Cairo.fill(cr)
        end

        save_canvas(surface, cr, filename)
        return true
    catch e
        @error "Non-robot generation failed" exception=(e, catch_backtrace())
        return false
    end
end

function generate_dataset(config::Config)
    # Create directory structure
    dirs = [
        "data/train/robot", "data/train/not_robot",
        "data/val/robot", "data/val/not_robot"
    ]
    mkpath.(dirs)

    # Calculate splits
    n_train = floor(Int, config.train_split * config.n_samples)
    n_val = config.n_samples - n_train

    # Generate training data
    @info "Generating training data..."
    for i in 1:n_train
        generate_robot_image(config, "data/train/robot/robot_$i.png")
        generate_non_robot_image(config, "data/train/not_robot/not_robot_$i.png")
    end

    # Generate validation data
    @info "Generating validation data..."
    for i in 1:n_val
        generate_robot_image(config, "data/val/robot/robot_$i.png")
        generate_non_robot_image(config, "data/val/not_robot/not_robot_$i.png")
    end
    
    @info "Dataset generation completed: $(2*config.n_samples) total images"
end

function test_generation(config=Config((256, 256), 32, 1e-3, 10, 0.8, 100))
    mkpath("test_samples")
    generate_robot_image(config, "test_samples/robot_sample.png")
    generate_non_robot_image(config, "test_samples/non_robot_sample.png")
    @info "Test images generated in test_samples/"
end