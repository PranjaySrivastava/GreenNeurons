import os, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SEED = 42
BATCH_SIZE = 2
EPOCHS = 8          # 🔥 reduced (major speedup)
EPOCHS_MODEL2 = 5   # 🔥 CNN shorter training
LR = 2e-4
T_IN = 10
T_OUT = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# MODELS
# -------------------------
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
            nn.Conv3d(in_ch, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3, padding=1),
            nn.ReLU(),
        )

        # 🔥 VERY IMPORTANT: maps to 16 output timesteps
        self.head = nn.Conv3d(64, T_OUT, 1)

    def forward(self, x):
        x = self.encoder(x)      # (B,64,T,H,W)
        x = self.head(x)         # (B,16,T,H,W)
        x = x[:, :, -1]          # (B,16,H,W) ✅ correct
        return x

model = Model(len(FEATS)).to(DEVICE)
model2 = CNNModel(len(FEATS)).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=LR)

# -------------------------
# LOSS
# -------------------------
def loss_fn(pred,target):
    weight = 1 + 2*(target > 0.6).float()
    smape = torch.abs(pred-target)/(torch.abs(pred)+torch.abs(target)+1e-6)
    mse = (pred-target)**2
    return (0.5*(smape*weight).mean() + 0.5*(mse*weight).mean())

# -------------------------
# TRAIN MODEL 1
# -------------------------
for epoch in range(EPOCHS):
    model.train()
    total = 0

    for x,y in train_loader:
        x,y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

        last = x[:,:, -1]
        pred = model(x) + last[:,0:1]

        loss = loss_fn(pred,y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()

        total += loss.item()

    print(f"[Model1] Epoch {epoch+1} | Loss: {total/len(train_loader):.4f}")

# -------------------------
# TRAIN MODEL 2
# -------------------------
print("\nTraining Model 2...")

for epoch in range(EPOCHS_MODEL2):
    model2.train()
    total = 0

    for x,y in train_loader:
        x,y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

        last = x[:,:, -1]
        pred = model2(x) + last[:,0:1]

        loss = loss_fn(pred,y)

        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()

        total += loss.item()

    print(f"[Model2] Epoch {epoch+1} | Loss: {total/len(train_loader):.4f}")

