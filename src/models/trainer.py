from typing import Dict, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd


class Trainer:
    """
    A class used to train an arbitrary model

    ...

    Attributes
    ----------
    model : nn.Module
        the module to use
    dataloaders : Dict[str, DataLoader]
        A dictionary containing training, validation, and test dataloaders
    """

    model: nn.Module
    dataloaders: Dict[str, DataLoader]

    """Initialize the trainer

    The model is initialized as a default model, and the dataloaders are empty
    """

    def __init__(self):
        self.model = nn.Module()
        self.dataloaders = {"train": None, "val": None, "test": None}

    """Create the model
    
    Set the model to the model in use
    """

    def create_model(self) -> None:
        pass

    """Set dataloaders
    
    Parameters
    ----------
    train_loader : DataLoader
        Dataloader that contain the training data
    
    val_loader : DataLoader
        Dataloader that contain the validation dataset
    
    test_loader : DataLoader
        Dataloader that contain the testing data
    """

    def set_dataloaders(
        self, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader
    ) -> None:
        self.dataloaders["train"] = train_loader
        self.dataloaders["val"] = val_loader
        self.dataloaders["test"] = test_loader

    """Calculate accuray for a given output and target

    The accuracy is calculated using cos similiraty
    
    Parameters
    ----------
    output : torch.Tensor
        A torch tensor contain the output from model

    target : torch.Tensor
        A troch tensor contain the target output

    Return
    ----------
    Accuracy of the output
    """

    def get_accuracy(self, output: torch.Tensor, target: torch.Tensor) -> float:
        # Calculate accuracy by finding cos similiraty
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        return torch.mean(torch.abs(cos(output, target)))

    """Train the model

    Parameters
    ----------
    weight_decay : float
        weight decay used for optimizer

    learning_rate : float
        learning rate used for optimizer

    num_epochs : int
        number of training iterations

    device : bool
        If the model should be trained using CPU or GPU
        default is torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    Return
    ---------
    A pd.DataFrame containing four columns: train_loss  val_loss  train_acc   val_acc
    """

    def train(
        self,
        weight_decay: float,
        learning_rate: float,
        num_epochs: int,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ) -> pd.DataFrame:

        model = self.model.to(device)  # move model to GPU if applicable
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        history = []

        for e in range(num_epochs):
            train_loss = 0.0
            val_loss = 0.0

            train_acc = 0
            val_acc = 0

            model.train()

            for data, target in self.dataloaders["train"]:

                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()

                # Clear gradients
                optimizer.zero_grad()
                # Predicted outputs
                output = model(data)

                # Loss and backpropagation of gradients
                loss = criterion(output, target)
                loss.backward()

                # Update the parameters
                optimizer.step()

                # Track train loss by multiplying average loss by number of examples in batch
                train_loss += loss.item() * data.size(0)
                # check target have same shape as output
                target = target.data.view_as(output)
                accuracy = self.get_accuracy(output, target)
                # Multiply average accuracy times the number of examples in batch
                train_acc += accuracy.item() * data.size(0)

            # Don't need to keep track of gradients
            with torch.no_grad():
                # Set to evaluation mode
                model.eval()

                # Validation loop
                for data, target in self.dataloaders["val"]:
                    # Tensors to gpu
                    data, target = data.cuda(), target.cuda()

                    # Forward pass
                    output = model(data)

                    # Validation loss
                    loss = criterion(output, target)
                    # Multiply average loss times the number of examples in batch
                    val_loss += loss.item() * data.size(0)

                    # check target have same shape as output
                    target = target.data.view_as(output)
                    accuracy = self.get_accuracy(output, target)
                    # Multiply average accuracy times the number of examples
                    val_acc += accuracy.item() * data.size(0)

                # Calculate average losses
                train_loss = train_loss / len(self.dataloaders["train"].dataset)
                val_loss = val_loss / len(self.dataloaders["val"].dataset)

                # Calculate average accuracy
                train_acc = train_acc / len(self.dataloaders["train"].dataset)
                val_acc = val_acc / len(self.dataloaders["val"].dataset)

                print(
                    f"\nEpoch: {e} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {val_loss:.4f}"
                )
                print(
                    f"\t \tTraining Accuracy: {100 * train_acc:.2f}% \tValidation Accuracy: {100 * val_acc:.2f}%"
                )
                history.append([train_loss, val_loss, train_acc, val_acc])

        return pd.DataFrame(
            history, columns=["train_loss", "val_loss", "train_acc", "val_acc"]
        )

    """Test the performance of the model

    TODO: implement this after we know how to test the model
    """

    def test(self):
        pass
