from trainer import *
import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset

class Dummy_Trainer(Trainer):
    def create_model(self):
        self.model = nn.Sequential(nn.Linear(64, 32),
                                   nn.ReLU(),
                                   nn.Linear(32, 7)
                                   )
    
dummy_trainer = Dummy_Trainer()
dummy_trainer.create_model()

data = []
label = []

for i in range(30):
    data.append(np.random.rand(64))
    label.append(np.random.rand(7))

dataset = TensorDataset(torch.Tensor(data), torch.Tensor(label))

train_ds, val_ds, test_ds = random_split(dataset, [10, 10, 10])

train_loader = DataLoader(train_ds, batch_size=2)
val_loader = DataLoader(val_ds, batch_size=2)
test_loader = DataLoader(test_ds, batch_size=2)

dummy_trainer.set_dataloaders(train_loader, val_loader, test_loader)

history = dummy_trainer.train(0.0, 0.001, 5)

print(history)