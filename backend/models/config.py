from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

@dataclass
class Config:
    features_path: Path
    out_dir: Path

    seed: int = 42
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-3
    epochs: int = 40
    patience: int = 6

    hidden_sizes: Tuple[int,int] = (256,128)
    dropout: float = 0.2

    test_actor_frac: float = 0.2
    val_actor_frac: float = 0.2