import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import json




def get_model_device(model):
    return next(iter(model.parameters())).device


class CausalConv1d(nn.Module):
    

    def __init__(self, hidden_size, kernel_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            hidden_size, hidden_size, kernel_size, groups=hidden_size, bias=True
        )

    def init_state(self, batch_size: int, device: torch.device | None = None):
        if device is None:
            device = get_model_device(self)
        return torch.zeros(
            batch_size, self.hidden_size, self.kernel_size - 1, device=device
        )

    def forward(self, x: torch.Tensor, state: torch.Tensor):
        x_with_state = torch.concat([state, x[:, :, None]], dim=-1)
        out = self.conv(x_with_state)
        new_state = x_with_state[:, :, 1:]
        return out.squeeze(-1), new_state


class Mamba2(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        inner_size: int | None = None,
        head_size: int = 64,
        bc_head_size: int = 128,
        conv_kernel_size: int = 4,
    ):
        super().__init__()

        self.head_size = head_size
        self.bc_head_size = bc_head_size
        if inner_size is None:
            inner_size = 2 * hidden_size
        assert inner_size % head_size == 0
        self.inner_size = inner_size
        self.num_heads = inner_size // head_size

        # Projections
        self.input_proj = nn.Linear(hidden_size, inner_size, bias=False)
        self.z_proj = nn.Linear(hidden_size, inner_size, bias=False)
        self.b_proj = nn.Linear(hidden_size, bc_head_size, bias=False)
        self.c_proj = nn.Linear(hidden_size, bc_head_size, bias=False)
        self.dt_proj = nn.Linear(hidden_size, self.num_heads, bias=True)

        # Convs
        self.input_conv = CausalConv1d(inner_size, conv_kernel_size)
        self.b_conv = CausalConv1d(bc_head_size, conv_kernel_size)
        self.c_conv = CausalConv1d(bc_head_size, conv_kernel_size)

        # Other parameters
        self.a = nn.Parameter(-torch.empty(self.num_heads).uniform_(1, 16))
        self.d = nn.Parameter(torch.ones(self.num_heads))

        # Output
        self.norm = nn.RMSNorm(inner_size, eps=1e-5)
        self.out_proj = nn.Linear(inner_size, hidden_size, bias=False)

    def init_state(self, batch_size: int, device: torch.device | None = None):
        if device is None:
            device = get_model_device(self)
        conv_states = [
            conv.init_state(batch_size, device)
            for conv in [self.input_conv, self.b_conv, self.c_conv]
        ]
        ssm_state = torch.zeros(
            batch_size, self.num_heads, self.head_size, self.bc_head_size, device=device
        )
        return conv_states + [ssm_state]

    def forward(self, t, state):
        batch_size = t.shape[0]

        x = self.input_proj(t)
        z = self.z_proj(t)
        b = self.b_proj(t)
        c = self.c_proj(t)
        dt = self.dt_proj(t)

        x_conv_state, b_conv_state, c_conv_state, ssm_state = state
        x, x_conv_state = self.input_conv(x, x_conv_state)
        b, b_conv_state = self.b_conv(b, b_conv_state)
        c, c_conv_state = self.c_conv(c, c_conv_state)
        x = F.silu(x)
        b = F.silu(b)
        c = F.silu(c)

        x = x.view(batch_size, self.num_heads, self.head_size)
        dt = F.softplus(dt)

        # SSM computation, this implements the discretized state space model.
        # new_state computation: h[t] = exp(A*dt) * h[t-1] + dt * B * x[t]
        # [batch_size, num_heads]
        decay = torch.exp(self.a[None] * dt)
        # Broadcasting everything to the right shapes:
        #  dt is [batch_size, num_heads]
        #  b  is [batch_size, bc_head_size]
        #  x  is [batch_size, head_size]
        # The new contribution (and ssm_state) is [batch_size, num_heads, head_size, bc_head_size]
        new_state_contrib = dt[:, :, None, None] * b[:, None, None] * x[:, :, :, None]
        ssm_state = decay[:, :, None, None] * ssm_state + new_state_contrib

        # output computation: y[t] = C @ h[t] + D * x[t]
        # The accumulation in the product of C and h[t] is on the bc_head_size dimension
        state_contrib = torch.einsum("bc,bnhc->bnh", c, ssm_state)
        # d has shape [num_heads], broadcasting it to the shape of x.
        y = state_contrib + self.d[None, :, None] * x

        # Combine heads
        y = y.view(batch_size, self.inner_size)
        # Gate, normalization and out
        y = y * F.silu(z)
        y = self.norm(y)
        output = self.out_proj(y)

        new_state = [x_conv_state, b_conv_state, c_conv_state, ssm_state]
        return output, new_state


class Mamba2Predictor(nn.Module):
    """Full model with input projection and output head"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        inner_size: int | None = None,
        head_size: int = 64,
        bc_head_size: int = 128,
        conv_kernel_size: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)

        # Mamba2 layers
        self.mamba_layers = nn.ModuleList(
            [
                Mamba2(
                    hidden_size,
                    inner_size=inner_size,
                    head_size=head_size,
                    bc_head_size=bc_head_size,
                    conv_kernel_size=conv_kernel_size,
                )
                for _ in range(num_layers)
            ]
        )

        # Layer norms
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(num_layers)]
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor, states=None):
        
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Initialize states if needed
        if states is None:
            states = [
                layer.init_state(batch_size, device) for layer in self.mamba_layers
            ]

        # Input projection
        x = self.input_proj(x)  # (batch, seq, hidden)
        x = self.input_norm(x)

        outputs = []
        final_states = []

        # Process sequence
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, hidden)

            # Pass through Mamba2 layers
            new_states = []
            for i, (mamba_layer, layer_norm) in enumerate(
                zip(self.mamba_layers, self.layer_norms)
            ):
                residual = x_t
                x_t, state = mamba_layer(x_t, states[i])
                x_t = layer_norm(x_t + residual)
                x_t = self.dropout(x_t)
                new_states.append(state)

            states = new_states
            outputs.append(x_t)

        # Stack outputs
        outputs = torch.stack(outputs, dim=1)  # (batch, seq, hidden)

        # Generate predictions
        predictions = self.output_head(outputs)  # (batch, seq, 1)

        return predictions, states




class TimeSeriesDataset(Dataset):
    def __init__(self, features, targets, seq_length=20):
        self.features = features
        self.targets = targets
        self.seq_length = seq_length

    def __len__(self):
        return len(self.features) - self.seq_length

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_length]
        y = self.targets[idx : idx + self.seq_length]
        return torch.FloatTensor(x), torch.FloatTensor(y).squeeze(-1)




class MetricsLogger:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "train_mse": [],
            "val_mse": [],
            "train_mae": [],
            "val_mae": [],
            "learning_rates": [],
        }

    def update(self, epoch_metrics):
        for key, value in epoch_metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)

    def save(self):
        with open(os.path.join(self.save_dir, "metrics.json"), "w") as f:
            json.dump(self.metrics, f, indent=4)

    def plot_metrics(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Training Metrics", fontsize=16)

        # Loss
        ax = axes[0, 0]
        ax.plot(self.metrics["train_loss"], label="Train Loss", marker="o")
        ax.plot(self.metrics["val_loss"], label="Val Loss", marker="s")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training and Validation Loss")
        ax.legend()
        ax.grid(True)

        # MSE
        ax = axes[0, 1]
        ax.plot(self.metrics["train_mse"], label="Train MSE", marker="o")
        ax.plot(self.metrics["val_mse"], label="Val MSE", marker="s")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.set_title("Mean Squared Error")
        ax.legend()
        ax.grid(True)

        # MAE
        ax = axes[1, 0]
        ax.plot(self.metrics["train_mae"], label="Train MAE", marker="o")
        ax.plot(self.metrics["val_mae"], label="Val MAE", marker="s")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MAE")
        ax.set_title("Mean Absolute Error")
        ax.legend()
        ax.grid(True)

        # Learning Rate
        ax = axes[1, 1]
        ax.plot(self.metrics["learning_rates"], marker="o", color="purple")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.grid(True)
        ax.set_yscale("log")

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "training_metrics.png"), dpi=300)
        plt.close()


def calculate_metrics(predictions, targets):
    """Calculate MSE and MAE"""
    mse = F.mse_loss(predictions, targets).item()
    mae = F.l1_loss(predictions, targets).item()
    return mse, mae


def save_checkpoint(
    model, optimizer, scheduler, epoch, metrics, save_dir, is_best=False
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
    }

    # Save regular checkpoint
    checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint, checkpoint_path)

    # Save best model
    if is_best:
        best_path = os.path.join(save_dir, "best_model.pt")
        torch.save(checkpoint, best_path)
        print(f"✓ Saved best model at epoch {epoch}")

    # Keep only last 5 checkpoints
    checkpoints = sorted(
        [f for f in os.listdir(save_dir) if f.startswith("checkpoint_epoch_")]
    )
    if len(checkpoints) > 5:
        for old_ckpt in checkpoints[:-5]:
            os.remove(os.path.join(save_dir, old_ckpt))




def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []

    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        # Forward pass
        predictions, _ = model(x)
        predictions = predictions.squeeze(-1)

        # Calculate loss
        loss = criterion(predictions, y)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        all_predictions.append(predictions.detach())
        all_targets.append(y.detach())

    avg_loss = total_loss / len(train_loader)
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    mse, mae = calculate_metrics(all_predictions, all_targets)

    return avg_loss, mse, mae


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            predictions, _ = model(x)
            predictions = predictions.squeeze(-1)

            loss = criterion(predictions, y)

            total_loss += loss.item()
            all_predictions.append(predictions)
            all_targets.append(y)

    avg_loss = total_loss / len(val_loader)
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    mse, mae = calculate_metrics(all_predictions, all_targets)

    return avg_loss, mse, mae


def train_model(model, train_loader, val_loader, config):
    """Main training loop"""
    device = config["device"]
    model = model.to(device)

    # Setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(config["save_dir"], f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # Save config
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # Initialize logger
    logger = MetricsLogger(save_dir)
    best_val_loss = float("inf")

    print(f"{'='*60}")
    print(f"Training started at {timestamp}")
    print(f"Model: {config['model_name']}")
    print(f"Device: {device}")
    print(f"Save directory: {save_dir}")
    print(f"{'='*60}\n")

    # Training loop
    for epoch in range(1, config["num_epochs"] + 1):
        # Train
        train_loss, train_mse, train_mae = train_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Validate
        val_loss, val_mse, val_mae = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Log metrics
        epoch_metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "train_mae": train_mae,
            "val_mae": val_mae,
            "learning_rates": current_lr,
        }
        logger.update(epoch_metrics)

        # Print progress
        print(f"Epoch {epoch}/{config['num_epochs']}")
        print(
            f"  Train - Loss: {train_loss:.6f}, MSE: {train_mse:.6f}, MAE: {train_mae:.6f}"
        )
        print(f"  Val   - Loss: {val_loss:.6f}, MSE: {val_mse:.6f}, MAE: {val_mae:.6f}")
        print(f"  LR: {current_lr:.2e}")

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        if epoch % config["save_every"] == 0 or is_best:
            save_checkpoint(
                model, optimizer, scheduler, epoch, epoch_metrics, save_dir, is_best
            )

        # Plot metrics every 10 epochs
        if epoch % 10 == 0:
            logger.plot_metrics()

        print()

    # Final save
    logger.save()
    logger.plot_metrics()

    print(f"{'='*60}")
    print(f"Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Results saved to: {save_dir}")
    print(f"{'='*60}")

    return model, logger



if __name__ == "__main__":
    from data_prep.data_clean import clean_indicator
    from data_prep.data_load import prepare_data

    torch.autograd.set_detect_anomaly(True)

    # Configuration
    config = {
        "model_name": "Mamba2Predictor",
        "seq_length": 20,
        "hidden_size": 128,
        "num_layers": 3,
        "inner_size": None,  # Will be 2 * hidden_size by default
        "head_size": 64,
        "bc_head_size": 128,
        "conv_kernel_size": 4,
        "dropout": 0.2,
        "batch_size": 64,
        "num_epochs": 100,
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "train_split": 0.8,
        "save_every": 5,
        "save_dir": "./checkpoints_mamba2",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    print("Loading data...")
    test_dir = "/home/aman/code/ml_fr/ml_stocks/data/NIFTY_5_years.csv"

    load_df = prepare_data(test_dir)
    df = clean_indicator(load_df)

    # Prepare features and target
    target_col = "Low"
    feature_cols = [col for col in df.columns if col != target_col]

    train_size = int(len(df) * config["train_split"])
    train_df = df[:train_size]
    val_df = df[train_size:]

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_df[feature_cols].values)
    val_features = scaler.transform(val_df[feature_cols].values)

    train_targets = train_df[target_col].values.reshape(-1, 1)
    val_targets = val_df[target_col].values.reshape(-1, 1)

    # Create datasets
    train_dataset = TimeSeriesDataset(
        train_features, train_targets, config["seq_length"]
    )
    val_dataset = TimeSeriesDataset(val_features, val_targets, config["seq_length"])

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Input features: {len(feature_cols)}")

    # Initialize model
    model = Mamba2Predictor(
        input_size=len(feature_cols),
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        inner_size=config["inner_size"],
        head_size=config["head_size"],
        bc_head_size=config["bc_head_size"],
        conv_kernel_size=config["conv_kernel_size"],
        dropout=config["dropout"],
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train model
    trained_model, metrics_logger = train_model(model, train_loader, val_loader, config)

    print(
        "\nTraining complete! Check the checkpoints_mamba2 directory for saved models and metrics."
    )
