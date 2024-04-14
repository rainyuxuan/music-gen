import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import math
from models.trainer import Trainer

def conv_size(org_size: int, kernel_size: int, stride: int, padding: int) -> int:
    return int(math.floor((org_size + 2.0 * padding - kernel_size) / stride) + 1)

class CNNTrainer(Trainer):
    def create_model(self, token_size, seq_size, n_layer, out_size) -> None:
        self.model = CNN(token_size, seq_size, n_layer, out_size)
    
    def train(
        self,
        weight_decay: float,
        learning_rate: float,
        num_epochs: int,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ) -> pd.DataFrame:

        self.model = self.model.to(device)  # move model to GPU if applicable
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        history = []

        for e in range(num_epochs):
            train_loss = 0.0
            val_loss = 0.0

            train_acc = 0
            val_acc = 0

            self.model.train()

            for data, target, fname in self.dataloaders["train"]:

                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()

                # Clear gradients
                optimizer.zero_grad()

                output = torch.empty((target.shape[0], target.shape[1], target.shape[2]), device=device)
                batch_loss = 0.0
                for i in range(target.shape[-1]):
                    # Predicted outputs
                    cur_output = self.model(data)

                    # Loss and backpropagation of gradients
                    loss = criterion(cur_output, target[:, :, i]) / target.shape[1]
                    loss.backward()
                    batch_loss += loss.item()

                    output[:, :, i] = cur_output
                    data = torch.cat((data[:, :, 1:], torch.unsqueeze(target[:, :, i], dim= -1)), dim=-1)

                # Update the parameters
                optimizer.step()

                # Track train loss by multiplying average loss by number of examples in batch
                train_loss += batch_loss * data.size(0)
                # check target have same shape as output
                target = target.data.view_as(output)
                accuracy = self.get_accuracy(output, target)
                # Multiply average accuracy times the number of examples in batch
                train_acc += accuracy.item() * data.size(0)

            # Don't need to keep track of gradients
            with torch.no_grad():
                # Set to evaluation mode
                self.model.eval()

                # Validation loop
                for data, target, fname in self.dataloaders["val"]:
                    # Tensors to gpu
                    if torch.cuda.is_available():
                        data, target = data.cuda(), target.cuda()

                    # Forward pass
                    output = self.model.generate(data)

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

class ConvBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(1)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x

class CNN(nn.Module):
    def __init__(self, token_size: int, seq_size: int, n_layer: int, out_size: int):
        super().__init__()
        self.token_size = token_size
        self.seq_size = seq_size
        self.n_layer = n_layer
        self.out_size = out_size
        cur_size = [self.token_size, self.seq_size]
        self.in_layer = nn.Conv2d(1, 1, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.bn = nn.BatchNorm2d(1)

        cur_size[0] = conv_size(conv_size(cur_size[0], 7, 2, 3), 3, 2, 1)
        cur_size[1] =  conv_size(conv_size(cur_size[1], 7, 2, 3), 3, 2, 1)

        self.blocks = nn.ModuleList([ConvBlock() for _ in range(self.n_layer)])

        for _ in range(self.n_layer):
            cur_size[0] = conv_size(cur_size[0], 3, 2, 1)
            cur_size[1] = conv_size(cur_size[1], 3, 2, 1)
        
        linear_in = cur_size[0] * cur_size[1]
        self.linear1 = nn.Linear(linear_in, 10)
        self.linear2 = nn.Linear(10, token_size)
    
    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        cur_token = self.in_layer(x)
        cur_token = self.bn(cur_token)
        cur_token = torch.max_pool2d(cur_token, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        for block in self.blocks:
            cur_token = block(cur_token)
            cur_token = torch.max_pool2d(cur_token, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        
        cur_token = torch.flatten(cur_token, start_dim=1)
        cur_token = self.linear1(cur_token)
        cur_token = torch.relu(cur_token)
        cur_token = self.linear2(cur_token)
        cur_token = torch.relu(cur_token)

        return cur_token
    
    def generate(self, x):
        device = x.device
        out = torch.empty((x.shape[0], self.token_size, self.out_size), device=device)

        for i in range(self.out_size):
            cur_token = self.forward(x)
            out[:, :, i] = cur_token
            x = torch.cat((x[:, :, 1:], torch.unsqueeze(cur_token, dim= -1)), dim=-1)
            

        return out

