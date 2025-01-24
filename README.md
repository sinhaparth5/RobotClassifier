# Robot Classifier

## Run the code
```bash
using Pkg
Pkg.activate(".")

include("src/RobotClassifier.jl")

# Generate dataset
RobotClassifier.generate_dataset(RobotClassifier.CONFIG)

# Train model
model, history = RobotClassifier.train_model(RobotClassifier.CONFIG)

# Visualize results
RobotClassifier.visualize_results(model, history)
```