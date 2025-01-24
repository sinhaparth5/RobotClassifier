using Plots

function plot_sample_images(path::String, n_samples=3)
    classes = readdir(path)
    p = plot(layout=(length(classes), n_samples), size=(900, 300))
    
    for (i, class) in enumerate(classes)
        class_path = joinpath(path, class)
        files = readdir(class_path)[1:n_samples]
        
        for (j, file) in enumerate(files)
            img = load(joinpath(class_path, file))
            plot!(p[i,j], heatmap(channelview(img)), axis=false, title=j==1 ? class : "")
        end
    end
    
    mkpath("visualizations")
    savefig(p, "visualizations/samples.png")
end

function plot_training_curves(history::TrainingHistory)
    p1 = plot(history.train_loss, label="Train", xlabel="Epoch", ylabel="Loss", title="Training/Validation Loss")
    plot!(p1, history.val_loss, label="Validation")
    
    p2 = plot(history.train_acc, label="Train", xlabel="Epoch", ylabel="Accuracy", title="Training/Validation Accuracy")
    plot!(p2, history.val_acc, label="Validation")
    
    p = plot(p1, p2, layout=(2,1), size=(800,600))
    savefig(p, "visualizations/training_curves.png")
end

function visualize_results(model, history::TrainingHistory)
    mkpath("visualizations")
    plot_sample_images("data/train")
    plot_training_curves(history)
end