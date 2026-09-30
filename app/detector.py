from ultralytics import YOLO


class ObjectDetector:

    def __init__(self, model_path):

        self.model = YOLO(model_path)


    def detect(
        self,
        frame,
        classes=None,
    ):

        return self.model.predict(
            frame,
            verbose=False,
            classes=classes,
        )
