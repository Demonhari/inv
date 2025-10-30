from __future__ import annotations
import os, random, json
from pathlib import Path
from datetime import datetime
import yaml
import numpy as np
import torch

def load_yaml(path: str | os.PathLike):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def dump_yaml(obj, path: str | os.PathLike):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

def dump_json(obj, path: str | os.PathLike):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def ensure_dir(path: str | os.PathLike):
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
