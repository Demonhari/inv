from edge_detector.model_zoo.yolo import YOLOv8Module
import torch
def test_forward():
    m = YOLOv8Module("yolov8n.pt"); out = m(torch.rand(1,3,640,640)/255)
    assert len(out)==1
