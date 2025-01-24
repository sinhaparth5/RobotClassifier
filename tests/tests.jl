using Test
using RobotClassifier

@testset "RobotClassifier.jl" begin
    config = Config((64, 64), 16, 1f-3, 2, 0.8, 10)
    
    @testset "Dataset Generation" begin
        generate_dataset(config)
        @test isdir("data/train/robot")
        @test length(readdir("data/train/robot")) == 8
    end
    
    @testset "Model Training" begin
        model, history = train_model(config)
        @test length(history.train_loss) == config.epochs
        @test all(0 .≤ history.train_acc .≤ 1)
    end
end