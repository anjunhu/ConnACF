"""GNN training pipeline for G-safeguard / BlindGuard defense.

Trains a :class:`MyGAT` model on graph data collected by
:class:`TrainingDataGenerator`.  The pipeline uses BCEWithLogitsLoss,
Adam optimizer, and CosineAnnealingLR scheduling.  The best checkpoint
(by validation accuracy) is saved to disk.

Usage (CLI)::

    python -m connacf.defense.train_gnn --dataset_path data/dataset.pkl \\
        --save_dir checkpoints/ --epochs 20

Usage (programmatic)::

    from connacf.defense.train_gnn import train_gnn
    train_gnn("data/dataset.pkl", "checkpoints/")
"""

import argparse
import logging
import os
import pickle
from datetime import datetime
from typing import Literal, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

from connacf.defense.gnn_model import MyGAT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AgentGraphDataset(Dataset):
    """PyTorch Geometric dataset that loads graph snapshots from a pickle file.

    The pickle file is expected to contain a list of dicts, each with keys
    ``features``, ``labels``, ``edge_index``, ``edge_attr``, and
    ``adj_matrix`` (all numpy arrays).  See
    :class:`~connacf.defense.training_data_generator.TrainingDataGenerator`
    for the serialization format.

    The dataset is split 80/20 into train and validation sets based on the
    ``phase`` parameter.

    Args:
        root: Path to the pickle file containing graph data.
        transform: Optional PyTorch Geometric transform.
        phase: ``"train"`` for the first 80% of samples, ``"val"`` for the
            remaining 20%.
    """

    def __init__(
        self,
        root: str,
        transform=None,
        phase: Literal["train", "val"] = "train",
    ):
        super().__init__()
        with open(root, "rb") as f:
            origin_dataset = pickle.load(f)

        n = len(origin_dataset)
        split = int(n * 0.8)

        if phase == "train":
            self.dataset = origin_dataset[:split]
        elif phase == "val":
            self.dataset = origin_dataset[split:]
        else:
            raise ValueError(f"Unknown phase '{phase}', expected 'train' or 'val'.")

    def len(self) -> int:  # noqa: D102 – PyG override
        return len(self.dataset)

    def get(self, idx: int) -> Data:  # noqa: D102 – PyG override
        raw = self.dataset[idx]
        x = torch.tensor(raw["features"], dtype=torch.float)
        y = torch.tensor(raw["labels"], dtype=torch.long)
        edge_index = torch.tensor(raw["edge_index"], dtype=torch.long)
        edge_attr = torch.tensor(raw["edge_attr"], dtype=torch.float)

        data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
        data.num_nodes = len(x)
        return data


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_epoch(
    model: MyGAT,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Run one training epoch.

    For each batch the node features ``x`` are recomputed from edge
    attributes via ``scatter(reduce='mean')`` following the reference
    implementation.

    Returns:
        ``(avg_loss, accuracy_percent)``
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data in loader:
        x = data.x.to(device)
        y = data.y.to(device)
        edge_index = data.edge_index.to(device)
        edge_attr = data.edge_attr.to(device)

        # Recompute node features from edge attributes (reference pattern).
        first_window = edge_attr[:, 0, :]
        x = scatter(first_window, edge_index[0], dim=0, dim_size=len(data.x), reduce="mean")

        optimizer.zero_grad()
        outputs = model(x, edge_index=edge_index, edge_attr=edge_attr)
        loss = criterion(outputs, y.float().unsqueeze(-1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = (torch.sigmoid(outputs) >= 0.5).squeeze()
        total += y.size(0)
        correct += (predicted == y).sum().item()

    avg_loss = running_loss / max(len(loader), 1)
    accuracy = 100.0 * correct / max(total, 1)
    return avg_loss, accuracy


def evaluate_epoch(
    model: MyGAT,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Run one evaluation epoch (no gradient computation).

    Returns:
        ``(avg_loss, accuracy_percent)``
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in loader:
            x = data.x.to(device)
            y = data.y.to(device)
            edge_index = data.edge_index.to(device)
            edge_attr = data.edge_attr.to(device)

            first_window = edge_attr[:, 0, :]
            x = scatter(first_window, edge_index[1], dim=0, dim_size=len(data.x), reduce="mean")

            outputs = model(x, edge_index, edge_attr)
            loss = criterion(outputs, y.float().unsqueeze(-1))
            running_loss += loss.item()

            predicted = (torch.sigmoid(outputs) >= 0.5).squeeze()
            total += y.size(0)
            correct += (predicted == y).sum().item()

    avg_loss = running_loss / max(len(loader), 1)
    accuracy = 100.0 * correct / max(total, 1)
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Main training function (programmatic API)
# ---------------------------------------------------------------------------

def train_gnn(
    dataset_path: str,
    save_dir: str,
    hidden_dim: int = 1024,
    num_heads: int = 8,
    num_layers: int = 2,
    epochs: int = 20,
    lr: float = 0.001,
    weight_decay: float = 0.0002,
    batch_size: int = 32,
    dropout: float = 0.2,
    device: str = None,
) -> str:
    """Train a :class:`MyGAT` model on collected ConnaCF graph data.

    Args:
        dataset_path: Path to the pickle file produced by
            :class:`~connacf.defense.training_data_generator.TrainingDataGenerator`.
        save_dir: Directory where the best checkpoint will be saved.
        hidden_dim: Total hidden dimensionality for the GAT layers.
        num_heads: Number of attention heads.
        num_layers: Number of GATwithEdgeConv layers.
        epochs: Number of training epochs.
        lr: Learning rate for Adam.
        weight_decay: Weight decay for Adam.
        batch_size: Mini-batch size for the training DataLoader.
        dropout: Dropout rate between GAT layers.
        device: Torch device string (e.g. ``"cuda:0"`` or ``"cpu"``).
            Defaults to the lowest-index GPU with ≥300 MiB free, or CPU.

    Returns:
        Path to the saved best-model checkpoint file.
    """
    # Resolve device — auto-select free GPU when not specified.
    if device is None:
        from connacf.utils.gpu_utils import free_device
        device = free_device()
    elif "cuda" in device and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU.")
        device = "cpu"
    device_obj = torch.device(device)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_dataset = AgentGraphDataset(dataset_path, phase="train")
    val_dataset = AgentGraphDataset(dataset_path, phase="val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Infer model dimensions from the first training example.
    example = train_dataset[0]
    in_channels = example.x.size(1)
    edge_dim: Tuple[int, ...] = tuple(example.edge_attr.size()[1:])

    logger.info(
        "Dataset loaded: %d train, %d val | in_channels=%d, edge_dim=%s",
        len(train_dataset),
        len(val_dataset),
        in_channels,
        edge_dim,
    )

    # ------------------------------------------------------------------
    # Model / optimiser / scheduler
    # ------------------------------------------------------------------
    model = MyGAT(
        in_channels,
        hidden_dim,
        out_channels=1,
        heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        edge_dim=edge_dim,
    )
    model.to(device_obj)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

    # ------------------------------------------------------------------
    # Checkpoint path
    # ------------------------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = (
        f"{timestamp}-hiddim_{hidden_dim}-heads_{num_heads}"
        f"-layers_{num_layers}-epochs_{epochs}-lr_{lr}"
        f"-dropout_{dropout}-wd_{weight_decay}.pth"
    )
    save_path = os.path.join(save_dir, filename)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_acc = 0.0

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device_obj
        )
        val_loss, val_acc = evaluate_epoch(
            model, val_loader, criterion, device_obj
        )
        scheduler.step()

        saved_tag = ""
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            saved_tag = " || Save!"

        logger.info(
            "Epoch %d/%d || Train Loss: %.4f, Acc: %.2f%% "
            "|| Val Loss: %.4f, Acc: %.2f%%%s",
            epoch,
            epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            saved_tag,
        )
        # Also print so CLI users see progress without configuring logging.
        print(
            f"Epoch {epoch}/{epochs} || "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% || "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%"
            f"{saved_tag}"
        )

    logger.info("Training complete. Best val accuracy: %.2f%%", best_acc)
    return save_path


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry-point for GNN training."""
    parser = argparse.ArgumentParser(
        description="Train MyGAT model on ConnaCF graph data for defense."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the pickle dataset file.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoint",
        help="Directory to save the best model checkpoint.",
    )
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0002)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (e.g. 'cuda:0' or 'cpu'). Defaults to cuda:<CUDA_VISIBLE_DEVICES> or cpu.",
    )

    args = parser.parse_args()
    if args.device is None:
        from connacf.utils.gpu_utils import free_device
        args.device = free_device()

    # Configure root logger for CLI usage.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    train_gnn(
        dataset_path=args.dataset_path,
        save_dir=args.save_dir,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        dropout=args.dropout,
        device=args.device,
    )


if __name__ == "__main__":
    main()
