import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from model import Model, CNNModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# LOAD DATA
# -------------------------
FEATS = [
    "cpm25","q2","t2","u10","v10","swdown","pblh","psfc","rain",
    "PM25","NH3","SO2","NOx","NMVOC_e","NMVOC_finn","bio"
]

TEST_ROOT = "/kaggle/input/.../test_in"  # UPDATE PATH

test_data = {f: np.load(f"{TEST_ROOT}/{f}.npy").astype(np.float32) for f in FEATS}

# -------------------------
# DATASET
# -------------------------
class TestDataset(Dataset):
    def __init__(self):
        self.x = np.stack([test_data[f] for f in FEATS], axis=1)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return torch.tensor(self.x[i]).float()

test_loader = DataLoader(TestDataset(), batch_size=2, shuffle=False)

# -------------------------
# LOAD MODEL
# -------------------------
checkpoint = torch.load("/kaggle/working/model.pth", map_location=DEVICE)

model = Model(len(FEATS)).to(DEVICE)
model.load_state_dict(checkpoint["model1"])

model2 = CNNModel(len(FEATS)).to(DEVICE)
model2.load_state_dict(checkpoint["model2"])

model.eval()
model2.eval()

# -------------------------
# TTA
# -------------------------
def tta(x,model):
    p1 = model(x)
    p2 = torch.flip(model(torch.flip(x,[3])),[2])
    p3 = torch.flip(model(torch.flip(x,[4])),[3])
    return (p1+p2+p3)/3

# -------------------------
# INFERENCE
# -------------------------
preds = []

with torch.no_grad():
    for x in test_loader:
        x = x.to(DEVICE)

        last = x[:,:, -1]

        p1 = tta(x, model)
        p2 = model2(x)

        p = 0.75*p1 + 0.25*p2
        p = p + last[:,0:1]

        preds.append(p.cpu().numpy())

preds = np.concatenate(preds,axis=0)

# -------------------------
# SAVE
# -------------------------
preds = preds.transpose(0,2,3,1).astype(np.float32)
np.save("/kaggle/working/preds.npy", preds)

print("Inference complete")
