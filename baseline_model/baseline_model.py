import torch
import torch.nn as nn
import torch.nn.functional as F
import training
import dataloader


class BaselineModel(nn.Module):
    """
        A very basic baseline model
    """
    def __init__(self) -> None:
        super(BaselineModel, self).__init__()
        self.name = "baseline"
        self.fc1 = nn.Linear(57, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 10) # Classify between 10 classes

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    # Fixed PyTorch random seed for reproducible result
    torch.manual_seed(123)

    # classes = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    baseline_model = BaselineModel()

    # Train the model
    path = "../Data/features_3_sec.csv"
    batch_size=64
    lr=0.0003
    num_epochs=70

    training.train(
        net=baseline_model,
        path=path,
        batch_size=batch_size,
        learning_rate=lr,
        num_epochs=num_epochs
    )

    # Plot the training curves
    model_path = training.get_model_name(baseline_model.name, batch_size, lr, num_epochs - 1)
    training.plot_training_curve(model_path)
