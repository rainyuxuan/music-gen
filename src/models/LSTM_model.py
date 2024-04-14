import os
import sys
wdir = os.getcwd().replace("\\", "/") + "/src"
sys.path.insert(1, wdir)

import torch
import torch.nn as nn

from features import *
from models.trainer import *
from data.dataset import SpectrogramDataset
from torch.utils.data import random_split
from tqdm import tqdm

@dataclass
class LSTMConfig:
    emb_size: int = 100
    hidden_size: int = 10
    output_size: int = 249
    token_size: int = 201
    dropout: float = 0.2
    batch_size: int = 2
    bias: bool = True


class MyLSTM(nn.Module):
    def __init__(self, config):
        """
        Develop a model designed to be trained using music data.
        It contains two single fully connected layers and one LSTM layer.

        : param input_size: input data size
        : param hidden_size: hidden size
        : param output_size: output data size
        : param dropout: dropout rate for fully connected layer
        : param batch_size: number of samples in a batch
        """
        super(MyLSTM, self).__init__()
        self.emb_size = config.emb_size
        self.hidden_size = config.hidden_size
        self.output_size = config.output_size
        self.token_size = config.token_size
        self.dropout = config.dropout
        self.batch_size = config.batch_size
        self.bias = config.bias
        # Call init_hidden to initialize for safety reasons
        self.hidden = None

        self.fc1 = nn.Linear(self.token_size, self.emb_size, bias=self.bias)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout)
        self.lstm = nn.LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.fc2 = nn.Linear(self.hidden_size, self.token_size, bias=self.bias)

    def forward(self, x):
        x = torch.transpose(x, -1, -2)
        self.init_hidden(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        out, self.hidden = self.lstm(x, self.hidden)
        logits = self.fc2(out[:, -1, :])
        return logits

    def init_hidden(self, input: torch.Tensor):
        self.hidden = (torch.zeros(1, input.shape[0], self.hidden_size).to(input.device),
                       torch.zeros(1, input.shape[0], self.hidden_size).to(input.device))
        
    def generate(self, input: torch.Tensor):
        device = input.device
        out = torch.empty((input.shape[0], self.token_size, self.output_size), device=device)

        for i in range(self.output_size):
            cur_token = self.forward(input)
            out[:, :, i] = cur_token
            input = torch.cat((input[:, :, 1:], torch.unsqueeze(cur_token, dim= -1)), dim=-1)

        return out


class LSTMTrainer(Trainer):
    def create_model(self, 
                     emb_size: int = 100,
                     hidden_size: int = 10,
                     output_size: int = 249,
                     token_size: int = 201,
                     dropout: float = 0.2,
                     batch_size: int = 2,
                     bias: bool = True) -> None:
        config = LSTMConfig(emb_size, 
                            hidden_size, 
                            output_size,
                            token_size,
                            dropout,
                            batch_size,
                            bias)
        self.model = MyLSTM(config)

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

            train_data = self.dataloaders["train"]
            train_bar = tqdm(train_data, total=len(train_data), desc=f'Train epoch {e}')
            for data, target, _ in train_bar:

                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()

                # Clear gradients
                optimizer.zero_grad()

                # Predicted outputs
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
                val_data = self.dataloaders["val"]
                val_bar = tqdm(val_data, total=len(val_data), desc=f'Validation epoch {e}')
                for data, target, _ in val_bar:
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


if __name__ == "__main__":
    wdir = os.getcwd()
    path = f"{wdir}/data/processed/musicnet"
    spec_dataset = SpectrogramDataset(
        f"{path}/train_data", label_dir=f"{path}/train_labels"
    )

    mini_dataset = [spec_dataset[i] for i in range(30)]

    train_set, valid_set = random_split(mini_dataset, [0.8, 0.2])
    train_dl = DataLoader(train_set, batch_size=2, shuffle=True)
    valid_dl = DataLoader(valid_set, batch_size=2, shuffle=True)
    trainer = LSTMTrainer()
    trainer.create_model()
    trainer.set_dataloaders(train_loader=train_dl, val_loader=valid_dl, test_loader=None)
    trainer.train(1e-4,1e-4, 10)
