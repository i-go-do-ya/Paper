import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchsummary import summary


class VGG16(nn.Module):
    def __init__(self, base_dim, num_classes=10):
        super(VGG16, self).__init__()
        self.feature = nn.Sequential(
            conv_2_block(3,base_dim), # 3x224x224 -> 64x112x112
            conv_2_block(base_dim,2*base_dim), # 64x112x112 -> 128x56x56
            conv_3_block(2*base_dim,4*base_dim), # 128x56x56 -> 256x28x28
            conv_3_block(4*base_dim,8*base_dim), # 256x28x28 -> 512x14x14
            conv_3_block(8*base_dim,8*base_dim), # 512x14x14 -> 512x7x7   
        )
        self.fc_layer = nn.Sequential(
            # IMAGENET : 224x224이므로
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.feature(x)
        #print(x.shape)
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1) # 1차원으로 펴줌
        #print(x.shape)
        x = self.fc_layer(x)
        return x
    

def conv_2_block(in_dim,out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
    return model


def conv_3_block(in_dim,out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
    return model


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(device)
    model = VGG16(base_dim=64).to(device)
    # print(model)
    print(summary(model, input_size=(3, 224, 224)))