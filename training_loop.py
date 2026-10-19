"""
Simple training loop for DiffusionPointCloudModel.

Features:
- Optionally uses the repo RF3DPoseDataset via the SimpleM4HumanDataset adapter, or a synthetic toy dataset.
- Implements a simple DDPM-style MSE loss where the model predicts the noise added to x0.
- Linear beta schedule and vectorized training step.
- Checkpoint saving and basic logging.

Usage examples:
  python training_loop.py --use_real_dataset False --num_points 128 --temporal_window 4 --batch_size 8 --epochs 10

If using the real dataset, provide --use_real_dataset True and ensure the dataset cache has been prepared. The script will attempt to import dataset utilities from the repository and will fall back to a synthetic dataset if unavailable.
"""

import os
import argparse
import math
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


# Try to import repository dataset adapter and model
try:
    from dataset.simple_dataset import SimpleM4HumanDataset
    from dataset.m4human_dataset import RF3DPoseDataset
    REAL_DS_AVAILABLE = True
except Exception:
    SimpleM4HumanDataset = None
    RF3DPoseDataset = None
    REAL_DS_AVAILABLE = False

try:
    from my_models.base_diffusion import DiffusionPointCloudModel
except Exception:
    # If import fails, raise since model should exist from prior step
    raise

gender_info = {
    "P1": 0, "P3": 0, "P4": 0, "P7": 0, "P8": 0, "P10": 0, "P13": 0, "P15": 0,
    "P2": 1, "P5": 1, "P6": 1, "P9": 1, "P11": 1, "P12": 1, "P14": 1, "P16": 1,
    "P17": 1, "P18": 1, "P19": 1, "P20": 1
}
root_dir = "/home/koponko/M4Human/rf3dpose_all"


class SyntheticPointCloudDataset(Dataset):
    """Small synthetic dataset for quick testing.

    Produces random point-cloud sequences x0 in a bounded cube so training loop can run without the full dataset.
    """

    def __init__(self, length=1000, temporal_window=4, num_points=128):
        self.length = length
        self.T = temporal_window
        self.P = num_points

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # x0 in range [-1, 1]
        x0 = (torch.rand(self.T, self.P, 3) - 0.5) * 2.0
        return {"radar_points": x0, "joints": torch.zeros(1)}


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


def train(
    model,
    optimizer,
    dataloader,
    device,
    timesteps=1000,
    betas=None,
    max_steps=None,
    save_every=500,
    out_dir="checkpoints",
):
    os.makedirs(out_dir, exist_ok=True)

    if betas is None:
        betas = linear_beta_schedule(timesteps).to(device)
    else:
        betas = betas.to(device)

    alphas = 1.0 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)

    mse = nn.MSELoss()

    step = 0
    model.train()

    for epoch in range(1000000):  # stop by max_steps
        epoch_loss = 0.0
        batches = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            # batch["radar_points"] expected shape (B, T, P, C) or (T,P,C) for single
            x0 = batch["radar_points"]
            if x0.dim() == 3:
                x0 = x0.unsqueeze(0)
            # The real radar cache stores xyz plus an auxiliary channel; the
            # diffusion model is configured for xyz input only.
            x0 = x0[..., :3].to(device).float()
            B = x0.shape[0]

            # sample timesteps uniformly for each sample in the batch
            t = torch.randint(0, timesteps, (B,), device=device)

            # sample noise
            eps = torch.randn_like(x0, device=device)

            # gather alpha_cumprod values for t and reshape for broadcasting
            a_t = alpha_cumprod[t].view(B, 1, 1, 1)
            sqrt_a_t = torch.sqrt(a_t)
            sqrt_one_minus_a_t = torch.sqrt(1 - a_t)

            # produce noisy x_t
            x_t = sqrt_a_t * x0 + sqrt_one_minus_a_t * eps

            # predict eps
            optimizer.zero_grad()
            pred = model(x_t, t)

            loss = mse(pred, eps)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1
            step += 1

            if step % save_every == 0:
                ckpt = {
                    "step": step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }
                ckpt_path = os.path.join(out_dir, f"ckpt_step_{step}.pt")
                torch.save(ckpt, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

            if max_steps is not None and step >= max_steps:
                print("Reached max_steps, finishing training loop.")
                return

        avg_loss = epoch_loss / max(1, batches)
        print(f"Epoch {epoch} completed — avg loss: {avg_loss:.6f} — steps: {step}")


def main():
    parser = argparse.ArgumentParser(description="Simple training loop for diffusion point-cloud model")
    parser.add_argument("--use_real_dataset", type=lambda s: s.lower() in ("true", "1"), default=False,
                        help="Whether to attempt to use the repository RF3DPoseDataset (True/False) — default False")
    parser.add_argument("--num_points", type=int, default=128, help="Number of points per frame (P)")
    parser.add_argument("--temporal_window", type=int, default=4, help="Temporal window length (T)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--timesteps", type=int, default=1000, help="Diffusion timesteps (T_diff)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=root_dir,
        help="Directory containing the RF3DPoseDataset cache parent (it contains rf3dpose_all/)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if args.use_real_dataset and REAL_DS_AVAILABLE:
        print("Attempting to create RF3DPoseDataset from repo (this may be slow and require prepared cache).")
        try:
            # user must provide file_pairs or adjust dataset creation in external code; here we attempt a minimal default
            # The RF3DPoseDataset requires a file_pairs list; try to find helper get_all_file_pairs
            from dataset.m4human_utils import get_all_file_pairs
            

            file_pairs = get_all_file_pairs(root_dir=root_dir, gender_info=gender_info)
            rf3d = RF3DPoseDataset(
                file_pairs,
                transform=None,
                load_save=True,
                use_image=False,
                cache_dir=args.cache_dir,
            )
            dataset = rf3d

            # infer num_points from first sample
            sample = dataset[0]
            inferred_P = sample["radar_points"].shape[1]
            print(f"Inferred num_points from dataset: {inferred_P}")
            num_points = inferred_P
        except Exception as e:
            print("Failed to create RF3DPoseDataset:", e)
            print("Falling back to synthetic dataset.")
            dataset = SyntheticPointCloudDataset(length=2000, temporal_window=args.temporal_window, num_points=args.num_points)
            num_points = args.num_points
    else:
        if args.use_real_dataset and not REAL_DS_AVAILABLE:
            print("RF3DPoseDataset adapter not available in this environment. Using synthetic dataset.")
        dataset = SyntheticPointCloudDataset(length=2000, temporal_window=args.temporal_window, num_points=args.num_points)
        num_points = args.num_points

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    # instantiate model, using inferred num_points if possible
    model = DiffusionPointCloudModel(in_channels=3, temporal_window=args.temporal_window, num_points=num_points, hidden_dim=256, time_emb_dim=128)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # we train for args.epochs over dataset epochs but allow early stop via max_steps
    max_steps = args.max_steps
    if max_steps is None:
        # set a default based on epochs and dataset size
        max_steps = args.epochs * math.ceil(len(dataloader))

    print(f"Starting training: epochs={args.epochs}, batch_size={args.batch_size}, dataset_len={len(dataset)}, max_steps={max_steps}")

    train(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        device=device,
        timesteps=args.timesteps,
        betas=None,
        max_steps=max_steps,
        save_every=args.save_every,
        out_dir=args.out_dir,
    )

    # final save
    final_path = os.path.join(args.out_dir, "final_ckpt.pt")
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}, final_path)
    print(f"Training complete. Final checkpoint saved to {final_path}")


if __name__ == "__main__":
    main()
