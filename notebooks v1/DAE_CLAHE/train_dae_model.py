"""Standalone DAE Training Script for Chest X-Ray Infiltration.

Trains Convolutional DAE with skip connections on realistic CXR synthetic noise:
  - Input: Noisy image (Poisson photon noise + Gaussian electronic noise)
  - Target: Clean ground truth CXR
  - Loss: Combined MSE Loss + L1 Gradient Edge Loss
  - Checkpoint: Saved to checkpoints/dae_trained.pth
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from dae_clahe_utils import DAE, add_synthetic_noise


class CXRDenoiseDataset(Dataset):
    def __init__(self, image_paths, img_size=256, noise_type="mixed", seed=42):
        self.image_paths = image_paths
        self.img_size = img_size
        self.noise_type = noise_type
        self.seed = seed

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        else:
            img = cv2.resize(img, (self.img_size, self.img_size))

        noisy = add_synthetic_noise(img, noise_type=self.noise_type, seed=self.seed + idx)

        # Convert to tensor [0, 1]
        noisy_t = torch.from_numpy(noisy.astype(np.float32) / 255.0).unsqueeze(0)
        clean_t = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        return noisy_t, clean_t


def train_dae(
    manifest_csv: str,
    base_data_dir: str,
    output_ckpt: str = "checkpoints/dae_trained.pth",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 1e-3,
    num_samples: int = 60,
    seed: int = 42
):
    print(f"=== Starting DAE Training Pipeline ===")
    print(f"Manifest: {manifest_csv}")
    print(f"Epochs: {epochs} | Batch size: {batch_size} | Samples: {num_samples} | LR: {lr}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    df = pd.read_csv(manifest_csv)
    # Sample balanced set of normal and infiltration
    norm_paths = [os.path.join(base_data_dir, p) for p in df[df['Class'] == 'Normal']['Relative_Path'].tolist()][:num_samples // 2]
    inf_paths = [os.path.join(base_data_dir, p) for p in df[df['Class'] == 'Infiltration']['Relative_Path'].tolist()][:num_samples // 2]
    all_paths = [p for p in (norm_paths + inf_paths) if os.path.exists(p)]

    print(f"Found {len(all_paths)} verified images for training.")
    if len(all_paths) == 0:
        raise FileNotFoundError("No training images found!")

    dataset = CXRDenoiseDataset(all_paths, img_size=256, noise_type="mixed", seed=seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = DAE(in_channels=1, base_channels=32, residual=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    model.train()
    for ep in range(1, epochs + 1):
        t0 = time.time()
        running_loss = 0.0
        for noisy_b, clean_b in loader:
            noisy_b = noisy_b.to(device)
            clean_b = clean_b.to(device)

            optimizer.zero_grad()
            pred = model(noisy_b)
            loss = mse_loss(pred, clean_b) + 0.5 * l1_loss(pred, clean_b)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(noisy_b)

        ep_loss = running_loss / len(dataset)
        elapsed = time.time() - t0
        print(f"Epoch [{ep}/{epochs}] - Loss: {ep_loss:.5f} - Time: {elapsed:.2f}s")

    os.makedirs(os.path.dirname(output_ckpt), exist_ok=True)
    model.save_checkpoint(output_ckpt, optimizer=optimizer, epoch=epochs)
    print(f"[DAE Train] Successfully saved trained checkpoint to: {output_ckpt}")
    return output_ckpt


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    manifest = os.path.join(current_dir, "../Gradcam/sample_manifest_200.csv")
    base_dir = os.path.abspath(os.path.join(current_dir, "../.."))
    ckpt_out = os.path.join(current_dir, "checkpoints/dae_trained.pth")

    train_dae(manifest_csv=manifest, base_data_dir=base_dir, output_ckpt=ckpt_out, epochs=3, batch_size=16)
