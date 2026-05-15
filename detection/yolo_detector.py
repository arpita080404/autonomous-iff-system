import cv2
import numpy as np
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.class_names = self.model.names

    def detect(self, frame):
        results = self.model(frame, verbose=False)
        result = results[0]
        annotated_frame = result.plot()
        return annotated_frame

    def get_detections(self, frame):
        results = self.model(frame, verbose=False)
        return results[0]

    def get_class_names(self):
        return self.class_names
