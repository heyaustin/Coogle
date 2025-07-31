# import warnings
# import math
from typing import Optional
import torch.nn.functional as F
import torchaudio  # just for convenience
from torch import Tensor
import torch.nn as nn
import os
import glob
import random
import torch
import torchaudio
import torchaudio.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
# from speechbrain.lobes.features import MFCC
from my_mfcc import MFCCFrontend

import torch
import torch.nn as nn
import torchaudio.functional as F
from speechbrain.lobes.features import MFCC

# Reproducibility
torch.manual_seed(42)
random.seed(42)

DATA_DIR = "datasets"       # 10 sub‑dirs (one per label)
SAMPLE_RATE = 16_000  # Hz
N_MFCC = 40     # learnable filterbank channels
BATCH_SIZE = 32
EPOCHS = 10
VAL_RATIO = 0.1
TEST_RATIO = 0.2
CLIP_LEN = SAMPLE_RATE

# STFT‑style params (optional, kept for reference)
N_FFT = 400  # 25 ms @ 16 kHz
HOP_LENGTH = 160  # 10 ms @ 16 kHz
N_MELS = 40   # mel filters (same as N_MFCC here)

mfcc_transform = MFCCFrontend(
    sr=SAMPLE_RATE,
    n_mfcc=N_MFCC,
    n_mels=N_MELS,
    win_len=N_FFT,
    hop_len=HOP_LENGTH
)


class SpeechCommands10(Dataset):
    # 2. Custom Dataset
    def __init__(self, root, mfcc_transform):
        self.files, self.labels = [], []
        for label in sorted(os.listdir(root)):
            path = os.path.join(root, label)
            if not os.path.isdir(path):
                continue
            for wav in glob.glob(os.path.join(path, "*.wav")):
                self.files.append(wav)
                self.labels.append(label)

        # Encode string labels → integers
        self.le = LabelEncoder()
        self.targets = self.le.fit_transform(self.labels)
        self.mfcc_transform = mfcc_transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        wav_path = self.files[idx]
        waveform, sr = torchaudio.load(wav_path)
        # Resample if necessary
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

        # Mono
        waveform = waveform.mean(dim=0, keepdim=True)

        # Pad / truncate to exactly 1 second
        if waveform.shape[1] < CLIP_LEN:
            pad = CLIP_LEN - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            waveform = waveform[:, :CLIP_LEN]

        # print shape of waveform
        # print("Waveform shape:", waveform.shape)
        mfcc = self.mfcc_transform(waveform)      # (n_mfcc, time)
        return mfcc, self.targets[idx]


class CNNKeyword(nn.Module):
    # 4. Simple CNN classifier
    #    Input shape: (batch, 1, n_mfcc, time)
    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            # nn.LazyLinear(128),
            # nn.Linear(9240, 128),  # Explicitly define input size
            nn.Linear(4040, 128),  # Explicitly define input size
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    # 3. Prepare data loaders
    dataset = SpeechCommands10(DATA_DIR, mfcc_transform)

    # Train/val/test split (stratified via indices per class)
    labels = dataset.targets
    indices = np.arange(len(dataset))
    test_size = int(TEST_RATIO * len(dataset))
    train_size = len(dataset) - test_size
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, stratify=labels, random_state=42)

    train_subset = torch.utils.data.Subset(dataset, train_idx)
    test_subset = torch.utils.data.Subset(dataset, test_idx)

    # Further split train → train/val
    val_size = int(VAL_RATIO * len(train_subset))
    train_size = len(train_subset) - val_size
    train_subset, val_subset = random_split(train_subset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNKeyword(n_classes=len(dataset.le.classes_)).to(device)

    # 5. Optimizer / loss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 6. Training loop
    best_val_acc, patience, wait = 0.0, 5, 0

    for epoch in tqdm(range(1, EPOCHS + 1)):
        model.train()
        for mfcc, target in train_loader:
            mfcc, target = mfcc.to(device), target.to(device)
            # print("MFCC shape:", mfcc.unsqueeze(1).shape)  # Debugging line
            logits = model(mfcc.unsqueeze(1))          # add channel dim
            loss = criterion(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ─ validation ─
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for mfcc, target in val_loader:
                mfcc, target = mfcc.to(device), target.to(device)
                logits = model(mfcc.unsqueeze(1))
                preds = logits.argmax(dim=1)
                correct += (preds == target).sum().item()
                total += target.size(0)

        val_acc = correct / total
        print(f"Epoch {epoch:02d} | val_acc = {val_acc:.4f}")

        # Early stopping
        # if val_acc > best_val_acc:
        #     best_val_acc, wait = val_acc, 0
        #     torch.save(model.state_dict(), "model.pt")
        # else:
        #     wait += 1
        #     if wait >= patience:
        #         print("Early stopping …")
        #         break

    torch.save(model.state_dict(), "model.pt")

    # 7. Testing
    model.load_state_dict(torch.load("model.pt"))
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for mfcc, target in test_loader:
            mfcc = mfcc.to(device)
            logits = model(mfcc.unsqueeze(1))
            preds = logits.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(target.numpy())

    print("Test accuracy:", accuracy_score(y_true, y_pred))
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=dataset.le.classes_,
            digits=3,
        )
    )
