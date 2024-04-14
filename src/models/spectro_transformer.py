import torch
from transformers import ASTConfig, ASTModel, ASTForAudioClassification
from torch import nn, Tensor
from data.dataset import *
from features.preprocessor import *
from models.trainer import *
from tqdm import tqdm

# n = 25

class AST(nn.Module):
    def __init__(self, config):
        super(AST, self).__init__()
        self.config = config
        self.output_layer = nn.Linear(in_features=config.hidden_size, out_features=config.num_mel_bins)
        self.transformer = ASTModel(self.config)

    def forward(self, x):
        # print(x.shape)
        x = torch.transpose(x,1,2)
        x = self.transformer(x)
        x = x.pooler_output
        x = self.output_layer(x)
        return x

class AST_trainer(Trainer):
    def __init__(self):
        super(AST_trainer, self).__init__()

    def create_model(self, config) -> None:
        self.model =  AST(config)

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

                # i = data.shape[2]
                # data = data[:, :, i-n:]
                # target = target[:, :, :5]
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                optimizer.zero_grad()
                batch_loss = 0.0
                output = torch.empty((target.shape[0], target.shape[1], target.shape[2]), device=device)
                for i in range(target.shape[-1]):
                    cur_output = self.model(data)
                    loss = criterion(cur_output, target[:, :, i]) / target.shape[1]
                    loss.backward()
                    batch_loss += loss.item()
                    output[:, :, i] = cur_output
                    data = torch.cat((data[:, :, 1:], torch.unsqueeze(target[:, :, i], dim=-1)), dim=-1)
                optimizer.step()
                # Track train loss by multiplying average loss by number of examples in batch
                train_loss += batch_loss * data.size(0)
                # check target have same shape as output
                target = target.data.view_as(output)
                accuracy = self.get_accuracy(output, target)
                # Multiply average accuracy times the number of examples in batch
                train_acc += accuracy * data.size(0)

            # Don't need to keep track of gradients
            with torch.no_grad():
                # Set to evaluation mode
                self.model.eval()

                # Validation loop
                val_data = self.dataloaders["val"]
                val_bar = tqdm(val_data, total=len(val_data), desc=f'Validation epoch {e}')
                for data, target, _ in val_bar:
                    # Tensors to gpu
                    # i = data.shape[2]
                    # data = data[:, :, i - n:]
                    # target = target[:, :, :5]
                    if torch.cuda.is_available():
                        data, target = data.cuda(), target.cuda()
                    # Forward pass
                    batch_loss = 0.0
                    output = torch.empty((target.shape[0], target.shape[1], target.shape[2]), device=device)
                    for i in range(target.shape[-1]):
                        cur_output = self.model(data)

                        loss = criterion(cur_output, target[:, :, i]) / target.shape[1]
                        batch_loss += loss.item()
                        output[:, :, i] = cur_output
                        data = torch.cat((data[:, :, 1:], torch.unsqueeze(target[:, :, i], dim=-1)), dim=-1)
                    # Validation loss

                    # Multiply average loss times the number of examples in batch
                    val_loss += batch_loss * data.size(0)

                    # check target have same shape as output
                    target = target.data.view_as(output)
                    accuracy = self.get_accuracy(output, target)
                    # Multiply average accuracy times the number of examples
                    val_acc += accuracy * data.size(0)

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
    train_data = SpectrogramDataset(data_dir="/Users/xunuo/Desktop/a2a-music-gen/data/processed/train",
                                    label_dir="/Users/xunuo/Desktop/a2a-music-gen/data/processed/train_label")
    valid_data = SpectrogramDataset(data_dir="/Users/xunuo/Desktop/a2a-music-gen/data/processed/test",
                                    label_dir="/Users/xunuo/Desktop/a2a-music-gen/data/processed/test_label")
    train_loader = DataLoader(train_data, batch_size=2)
    val_loader = DataLoader(valid_data, batch_size=2)
    configuration = ASTConfig()
    configuration.num_mel_bins = 201  # dimension size of frequency
    configuration.max_length = 662  # dimension size of input time length
    configuration.hidden_dropout_prob = 0.2
    model = AST(configuration)
    Trainer = AST_trainer()
    Trainer.set_dataloaders(train_loader=train_loader, val_loader=val_loader, test_loader=None)
    Trainer.model = model
    Trainer.train(1e-4, 1e-2, 10)





