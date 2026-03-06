from ultralytics import YOLO
import pandas as pd


class TreeDetector:

    _model = None

    def __init__(self):

        if TreeDetector._model is None:

            TreeDetector._model = YOLO("models/tree_detector.pt")

        self.model = TreeDetector._model


    def detect(self, image_path):

        results = self.model(
            image_path,
            conf=0.25,
            imgsz=1024
        )[0]

        boxes = results.boxes.xyxy.cpu().numpy()

        data = []

        for box in boxes:

            xmin, ymin, xmax, ymax = box

            data.append({
                "xmin": float(xmin),
                "ymin": float(ymin),
                "xmax": float(xmax),
                "ymax": float(ymax)
            })

        predictions = pd.DataFrame(data)

        return predictions