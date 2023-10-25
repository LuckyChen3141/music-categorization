import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler


class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.label_idx = {label: idx for idx, label in enumerate(sorted(set(self.data["label"])))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input = torch.tensor(self.data.iloc[idx, 2:-1], dtype=torch.float32)
        label = F.one_hot(torch.tensor(self.label_idx[self.data.iloc[idx, -1]]), num_classes=len(self.label_idx))
        
        if self.transform:
            input = self.transform(input)

        return input, label
    

class BaselineModel(nn.Module):
    """
        A very basic baseline model
    """
    def __init__(self) -> None:
        super(BaselineModel, self).__init__()
        self.name = "baseline"
        self.fc1 = nn.Linear(57, 100)
        self.fc2 = nn.Linear(100, 10) # Classify between 10 classes

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def train(self, train_loader, val_loader, batch_size=1, learning_rate=0.01, num_epochs=10):
        ########################################################################
        # Define the Loss function and optimizer
        # The loss function will be Cross Entropy.
        # Optimizer will be SGD with Momentum.
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)
        ########################################################################
        # Set up some numpy arrays to store the training/test loss/erruracy
        train_err = np.zeros(num_epochs)
        train_loss = np.zeros(num_epochs)
        val_err = np.zeros(num_epochs)
        val_loss = np.zeros(num_epochs)
        ########################################################################
        # Train the network
        # Loop over the data iterator and sample a new batch of training data
        # Get the output from the network, and optimize our loss function.
        start_time = time.time()
        for epoch in range(num_epochs):  # loop over the dataset multiple times
            total_train_loss = 0.0
            total_train_err = 0.0
            total_epoch = 0
            for i, data in enumerate(train_loader, 0):
                # Get the inputs
                inputs, labels = data
                # Zero the parameter gradients
                optimizer.zero_grad()
                # Forward pass, backward pass, and optimize
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # Calculate the statistics
                corr = (torch.argmax(outputs,1) == labels)
                total_train_err += int(corr.sum())
                total_train_loss += loss.item()
                total_epoch += len(labels)
            train_err[epoch] = float(total_train_err) / total_epoch
            train_loss[epoch] = float(total_train_loss) / (i+1)
            val_err[epoch], val_loss[epoch] = self.evaluate(val_loader, criterion)
            print(("Epoch {}: Train err: {}, Train loss: {} |"+
                "Validation err: {}, Validation loss: {}").format(
                    epoch + 1,
                    train_err[epoch],
                    train_loss[epoch],
                    val_err[epoch],
                    val_loss[epoch]))
        # Save the trained model to a file
        model_path = get_model_name(self.name, batch_size, learning_rate, epoch)
        torch.save(self.state_dict(), model_path)
        print('Finished Training')
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Total time elapsed: {:.2f} seconds".format(elapsed_time))
        # Write the train/test loss/err into CSV file for plotting later
        np.savetxt("{}_train_err.csv".format(model_path), train_err)
        np.savetxt("{}_train_loss.csv".format(model_path), train_loss)
        np.savetxt("{}_val_err.csv".format(model_path), val_err)
        np.savetxt("{}_val_loss.csv".format(model_path), val_loss)

    def evaluate(self, loader, criterion):
        """ Evaluate the network on the validation set.

        Args:
            loader: PyTorch data loader for the validation set
            criterion: The loss function
        Returns:
            err: A scalar for the avg classification error over the validation set
            loss: A scalar for the average loss function over the validation set
        """
        total_loss = 0.0
        total_err = 0.0
        total_epoch = 0
        for i, data in enumerate(loader, 0):
            inputs, labels = data
            outputs = self(inputs)
            loss = criterion(outputs, labels)
            corr = (torch.argmax(outputs,1) == labels)
            total_err += int(corr.sum())
            total_loss += loss.item()
            total_epoch += len(labels)
        err = float(total_err) / total_epoch
        loss = float(total_loss) / (i + 1)
        return err, loss


def plot_training_curve(path):
    """ Plots the training curve for a model run, given the csv files
    containing the train/validation error/loss.

    Args:
        path: The base path of the csv files produced during training
    """
    train_err = np.loadtxt("{}_train_err.csv".format(path))
    val_err = np.loadtxt("{}_val_err.csv".format(path))
    train_loss = np.loadtxt("{}_train_loss.csv".format(path))
    val_loss = np.loadtxt("{}_val_loss.csv".format(path))
    plt.title("Train vs Validation Error")
    n = len(train_err) # number of epochs
    plt.plot(range(1,n+1), train_err, label="Train")
    plt.plot(range(1,n+1), val_err, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend(loc='best')
    plt.show()
    plt.title("Train vs Validation Loss")
    plt.plot(range(1,n+1), train_loss, label="Train")
    plt.plot(range(1,n+1), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()


def get_model_name(name, batch_size, learning_rate, epoch):
    """ Generate a name for the model consisting of all the hyperparameter values
    Args:
        name: name of the model
        batch_size: size of the batch used during training
        learning_rate: learning rate value used during training
        epoch: epoch value
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    path = "model_{0}_bs{1}_lr{2}_epoch{3}".format( name,
                                                    batch_size,
                                                    learning_rate,
                                                    epoch)
    return path


# def set_device():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     if device != "cuda":
#         print("CPU is enabled in this notebook.")
#     else:
#         print("GPU is enabled in this notebook.")

#     return device

def get_dataloader(path):
    trainset = CustomDataset(path)

    # Split into train and validation
    indices = list(range(len(trainset)))
    np.random.seed(1000) # Fixed numpy random seed for reproducible shuffling
    np.random.shuffle(indices)
    split = int(len(indices) * 0.8) #split at 80%

    # split into training and validation indices
    train_indices, val_indices = indices[:split], indices[split:]  
    train_sampler = SubsetRandomSampler(train_indices)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, num_workers=1, sampler=train_sampler)
    val_sampler = SubsetRandomSampler(val_indices)
    val_loader = torch.utils.data.DataLoader(trainset, batch_size=1, num_workers=1, sampler=val_sampler)
    return train_loader, val_loader


if __name__ == "__main__":
    # Fixed PyTorch random seed for reproducible result
    torch.manual_seed(123)

    # classes = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    path = "./Data/features_30_sec.csv"
    train_loader, val_loader = get_dataloader(path)

    baseline_model = BaselineModel()

    # Train the model
    batch_size = 1
    lr = 0.01
    num_epochs = 10
    baseline_model.train(
        train_loader=train_loader,
        val_loader=val_loader,
        batch_size=batch_size,
        learning_rate=lr,
        num_epochs=num_epochs
    )

    # Plot the training curves
    model_path = get_model_name(baseline_model.name, batch_size, lr, num_epochs - 1)
    plot_training_curve(model_path)
