import torch
import torch.nn as nn
import math

def conv_size(org_size: int, kernel_size: int, stride: int, padding: int) -> int:
    return int(math.floor((org_size + 2.0 * padding - kernel_size) / stride) + 1)


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
        hidden_size = int((linear_in + token_size) / 2)
        self.linear1 = nn.Linear(linear_in, hidden_size)
        self.linear2 = nn.Linear(hidden_size, token_size)
    
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

