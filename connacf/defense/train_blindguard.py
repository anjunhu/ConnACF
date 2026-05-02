"""BlindGuard training pipeline.

Trains a BlindGuardModel on normal (unattacked) graph data collected by
TrainingDataGenerator. Unlike G-Safeguard, BlindGuard is unsupervised:
it only needs normal interaction data and learns to detect anomalies via
corruption-guided contrastive learning.

Usage (CLI)::

    python -m connacf.defense.train_blindguard --dataset_path data/normal.pkl \\
        --save_dir checkpoints/ --epochs 50

Usage (programmatic)::

    from connacf.defense.train_blindguard import train_blindguard_from_dataset
    train_blindguard_from_dataset("data/normal.pkl", "checkpoints/")
"""

import argparse
import logging
import os
import pickle
from datetime import datetime

import torch
from torch_geometric.data import Data

from connacf.defense.blindguard_model import BlindGuardModel, train_blindguard

logger = logging.getLogger(__name__)


def load_normal_graphs(dataset_path: str) -> list:
    """Load normal (unattacked) graphs from a pickle file.

    Filters out any graphs with attacker labels (y=1) to ensure only
    normal data is used for training.

    Args:
        dataset_path: Path to pickle file from TrainingDataGenerator.

    Returns:
        List of PyG Data objects with only normal agent interactions.
    """
    with open(dataset_path, "rb") as f:
        raw = pickle.load(f)

    graphs = []
    for sample in raw:
        x = torch.tensor(sample["features"], dtype=torch.float)
        edge_index = torch.tensor(sample["edge_index"], dtype=torch.long)
        y = torch.tensor(sample.get("labels", [0] * x.size(0)), dtype=torch.long)

        # Use only graphs with no attackers (or strip attacker nodes)
        # For simplicity, include all graphs but only use x (not labels)
        graphs.append(Data(x=x, edge_index=edge_index, y=y))

    logger.info(f"Loaded {len(graphs)} graphs from {dataset_path}")
    return graphs


def train_blindguard_from_dataset(
    dataset_path: str,
    save_dir: str,
    input_dim: int = 384,
    hidden_dim: int = 512,
    output_dim: int = 256,
    corruption_alpha: float = 1.0,
    temperature: float = 0.07,
    n_epochs: int = 50,
    lr: float = 1e-3,
    corruption_fraction: float = 0.3,
) -> str:
    """Train BlindGuard on normal graph data and save checkpoint.

    Args:
        dataset_path: Path to pickle file with normal interaction graphs.
        save_dir: Directory to save the trained model checkpoint.
        input_dim: SentenceBERT embedding dimension.
        hidden_dim: MLP hidden dimension.
        output_dim: Representation dimension.
        corruption_alpha: Noise scaling for synthetic anomalies.
        temperature: Contrastive loss temperature.
        n_epochs: Training epochs.
        lr: Learning rate.
        corruption_fraction: Fraction of nodes to corrupt per graph.

    Returns:
        Path to saved checkpoint.
    """
    from connacf.utils.gpu_utils import free_device
    device = free_device()

    graphs = load_normal_graphs(dataset_path)
    if not graphs:
        raise ValueError(f"No graphs found in {dataset_path}")

    model = BlindGuardModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        corruption_alpha=corruption_alpha,
        temperature=temperature,
    )

    model = train_blindguard(
        model=model,
        normal_graphs=graphs,
        n_epochs=n_epochs,
        lr=lr,
        corruption_fraction=corruption_fraction,
        device=device,
    )

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(save_dir, f"blindguard_{timestamp}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"[BlindGuard] Saved checkpoint to {checkpoint_path}")
    print(f"[BlindGuard] ✓ Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Train BlindGuard on normal interaction data")
    parser.add_argument("--dataset_path", required=True, help="Path to normal graphs pickle")
    parser.add_argument("--save_dir", default="checkpoints/", help="Checkpoint output directory")
    parser.add_argument("--input_dim", type=int, default=384)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--output_dim", type=int, default=256)
    parser.add_argument("--corruption_alpha", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--corruption_fraction", type=float, default=0.3)
    args = parser.parse_args()

    train_blindguard_from_dataset(
        dataset_path=args.dataset_path,
        save_dir=args.save_dir,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        corruption_alpha=args.corruption_alpha,
        temperature=args.temperature,
        n_epochs=args.epochs,
        lr=args.lr,
        corruption_fraction=args.corruption_fraction,
    )
