from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any
from rich.console import Console
from rich.table import Table
from ultralytics import YOLO

from .utils import (
    load_yaml, dump_yaml, dump_json,
    ensure_dir, timestamp, seed_everything
)

console = Console()

def _flat_aug(aug: Dict[str, Any]) -> Dict[str, Any]:
    # Map config keys to ultralytics train args
    keymap = {
        "hsv_h": "hsv_h", "hsv_s": "hsv_s", "hsv_v": "hsv_v",
        "degrees": "degrees", "translate": "translate", "scale": "scale",
        "shear": "shear", "perspective": "perspective",
        "flipud": "flipud", "fliplr": "fliplr",
        "mosaic": "mosaic", "mixup": "mixup", "copy_paste": "copy_paste",
    }
    return { keymap[k]: v for k, v in (aug or {}).items() if k in keymap }

def train(config_path: str) -> str:
    cfg = load_yaml(config_path)
    seed_everything(int(cfg.get("seed", 42)))

    project = cfg.get("project_name", "defect_detection")
    run_name = cfg.get("run_name", f"run_{timestamp()}")
    save_dir = ensure_dir(cfg.get("save_dir", "outputs"))
    model_name = cfg.get("model", "yolov8s.pt")
    data_yaml = cfg["data_config"]

    # Collect YOLO train params
    train_args = {
        "data": data_yaml,
        "epochs": int(cfg.get("epochs", 100)),
        "imgsz": int(cfg.get("imgsz", 640)),
        "batch": int(cfg.get("batch", 16)),
        "device": cfg.get("device", "auto"),
        "workers": int(cfg.get("workers", 8)),
        "optimizer": cfg.get("optimizer", "AdamW"),
        "lr0": float(cfg.get("lr0", 0.001)),
        "lrf": float(cfg.get("lrf", 0.01)),
        "weight_decay": float(cfg.get("weight_decay", 0.0005)),
        "patience": int(cfg.get("patience", 50)),
        "cos_lr": bool(cfg.get("cos_lr", True)),

        # Where to store
        "project": save_dir,
        "name": f"{project}_{run_name}",
        "exist_ok": True,

        # Save more artifacts
        "save": True,
        "save_period": 10,
        "cache": "ram",
    }
    # augmentations
    train_args.update(_flat_aug(cfg.get("augmentations", {})))

    console.rule("[bold green]Training configuration")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Param")
    table.add_column("Value")
    for k, v in train_args.items():
        table.add_row(k, str(v))
    console.print(table)

    model = YOLO(model_name)
    results = model.train(**train_args)

    # Save a frozen copy of the config next to results
    run_dir = Path(results.save_dir)  # ultralytics exposes run dir here
    dump_yaml(cfg, run_dir / "config_used.yaml")

    console.print(f"[bold green]Training complete.[/bold green] Run dir: [cyan]{run_dir}[/cyan]")
    return str(run_dir)

def predict(
    weights: str,
    source: str,
    save_dir: str = "outputs",
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str = "auto",
    half: bool = False,
    visualize: bool = False,
) -> str:
    model = YOLO(weights)
    name = f"pred_{timestamp()}"
    args = dict(
        source=source, imgsz=imgsz, conf=conf, iou=iou, device=device,
        half=half, visualize=visualize, save=True, save_txt=True, save_conf=True,
        project=save_dir, name=name, exist_ok=True
    )
    console.print(f"[bold]Predicting on[/bold] {source}")
    res = model.predict(**args)
    out_dir = Path(res[0].save_dir) if res else Path(save_dir) / name
    console.print(f"[bold green]Predictions saved to[/bold green] {out_dir}")
    return str(out_dir)

def validate(
    weights: str,
    data_yaml: str,
    conf: float = 0.25,
    iou: float = 0.50,
    device: str = "auto",
    plots: bool = True,
    save_json: bool = True,
) -> Dict[str, Any]:
    model = YOLO(weights)
    console.print(f"[bold]Validating[/bold] weights: {weights}")
    metrics = model.val(
        data=data_yaml, conf=conf, iou=iou, device=device,
        plots=plots, save_json=save_json
    )
    # Extract common metrics into a nice dict
    out = dict(
        metrics=metrics.results_dict if hasattr(metrics, "results_dict") else {},
        speed=getattr(metrics, "speed", {}),
        confusion_matrix_path=str(getattr(metrics, "confusion_matrix", None)),
        pr_curve_path=str(getattr(metrics, "save_dir", "")),
    )
    # Save beside val outputs
    save_dir = Path(getattr(metrics, "save_dir", "outputs")) / f"eval_{timestamp()}"
    ensure_dir(save_dir)
    dump_json(out, save_dir / "metrics_summary.json")
    console.print(f"[bold green]Validation summary saved to[/bold green] {save_dir / 'metrics_summary.json'}")
    return out

def export(weights: str, fmt: str = "onnx") -> str:
    """
    Export formats include: onnx, openvino, engine(TensorRT), coreml, saved_model, tflite, pb, torchscript
    """
    model = YOLO(weights)
    console.print(f"[bold]Exporting[/bold] {weights} -> format={fmt}")
    export_path = model.export(format=fmt)
    console.print(f"[bold green]Exported model[/bold green]: {export_path}")
    return str(export_path)
