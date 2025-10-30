from __future__ import annotations
import typer
from rich import print
from pathlib import Path

from .engine import train as _train, predict as _predict, validate as _validate, export as _export
from .utils import ensure_dir

app = typer.Typer(help="Defect Detection CLI (YOLO-powered)")

@app.command()
def init_dirs():
    """
    Create a nice dataset skeleton with .gitkeep files so Git tracks empty dirs.
    """
    base = Path("data/defects")
    for sub in ["images/train", "images/val", "images/test", "labels/train", "labels/val", "labels/test"]:
        p = base / sub
        p.mkdir(parents=True, exist_ok=True)
        (p / ".gitkeep").touch()
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    (out / ".gitkeep").touch()
    print("[green]Created dataset skeleton under data/defects and outputs/.gitkeep[/green]")

@app.command()
def train(config: str = typer.Option("configs/train.yaml", "--config", "-c", help="Path to training YAML")):
    run_dir = _train(config)
    print(f"[bold green]Run complete:[/bold green] {run_dir}")

@app.command()
def predict(
    weights: str = typer.Argument(..., help="Path to .pt weights (e.g., best.pt)"),
    source: str = typer.Argument(..., help="Image/dir/video path or glob (e.g., data/defects/images/test)"),
    save_dir: str = typer.Option("outputs", help="Save directory for predictions"),
    imgsz: int = typer.Option(640, help="Inference image size"),
    conf: float = typer.Option(0.25, help="Confidence threshold"),
    iou: float = typer.Option(0.45, help="NMS IoU threshold"),
    device: str = typer.Option("auto", help="Device: auto, cpu, 0, 0,1 ..."),
    half: bool = typer.Option(False, help="Use half precision (FP16) if supported"),
    visualize: bool = typer.Option(False, help="Enable feature map visualization"),
):
    _predict(weights, source, save_dir, imgsz, conf, iou, device, half, visualize)

@app.command("val")
def validate(
    weights: str = typer.Argument(..., help="Path to .pt weights"),
    data_yaml: str = typer.Option("data/defect_data.yaml", help="YOLO data yaml"),
    conf: float = typer.Option(0.25, help="Conf threshold"),
    iou: float = typer.Option(0.50, help="IoU threshold"),
    device: str = typer.Option("auto", help="Device"),
    plots: bool = typer.Option(True, help="Save PR/Curves/Confusion plots"),
    save_json: bool = typer.Option(True, help="Save COCO-style json metrics where applicable"),
):
    _validate(weights, data_yaml, conf, iou, device, plots, save_json)

@app.command()
def export(
    weights: str = typer.Argument(..., help="Path to .pt weights"),
    fmt: str = typer.Option("onnx", help="Export format (onnx/openvino/engine/coreml/...)"),
):
    _export(weights, fmt)

if __name__ == "__main__":
    app()
