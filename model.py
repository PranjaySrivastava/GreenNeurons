import torch
import torch.nn as nn

T_OUT = 16

class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hidden_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch + hidden_ch, 4*hidden_ch, 3, padding=1)

    def forward(self, x, h, c):
        gates = self.conv(torch.cat([x,h], dim=1))
        i,f,o,g = torch.chunk(gates, 4, dim=1)

        i,f,o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)

        c = f*c + i*g
        h = o * torch.tanh(c)
        return h,c


class Model(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, 64, 1)

        self.lstm1 = ConvLSTMCell(64,64)
        self.lstm2 = ConvLSTMCell(64,64)

        self.head = nn.Sequential(
            nn.Conv2d(64,64,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,32,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,T_OUT,1)
        )

    def forward(self,x):
        B,C,T,H,W = x.shape

        h1 = torch.zeros(B,64,H,W,device=x.device)
        c1 = torch.zeros(B,64,H,W,device=x.device)
        h2 = torch.zeros(B,64,H,W,device=x.device)
        c2 = torch.zeros(B,64,H,W,device=x.device)

        for t in range(T):
            xt = self.proj(x[:,:,t])
            h1,c1 = self.lstm1(xt,h1,c1)
            h2,c2 = self.lstm2(h1,h2,c2)

        return self.head(h2)


class CNNModel(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(in_ch,32,3,padding=1),
            nn.ReLU(),
            nn.Conv3d(32,64,3,padding=1),
            nn.ReLU(),
            nn.Conv3d(64,64,3,padding=1),
            nn.ReLU(),
        )

        self.head = nn.Conv3d(64,T_OUT,1)

    def forward(self,x):
        x = self.encoder(x)
        x = self.head(x)
        return x[:,:,-1]
