import cv2

from PySide6.QtCore import QThread, Signal

from app.detector import ObjectDetector
from app.tracker import ObjectTracker


class VideoWorker(QThread):

    frame_ready = Signal(object)
    status_changed = Signal(str)
    finished_processing = Signal()

    def __init__(
        self,
        video_path,
        model_path="yolo11n.pt",
        parent=None,
    ):

        super().__init__(parent)

        self.video_path = video_path
        self.model_path = model_path

        self.running = True

        self.detector = ObjectDetector(
            self.model_path
        )

        self.tracker = ObjectTracker(
            self.detector.model
        )

    def run(self):

        print("WORKER: run() started")

        self.status_changed.emit(
            "Opening video..."
        )

        video = cv2.VideoCapture(
            self.video_path
        )

        if not video.isOpened():

            print("WORKER: Could not open video")

            self.status_changed.emit(
                "ERROR: Could not open video"
            )

            self.finished_processing.emit()

            return

        print("WORKER: Video opened")

        self.status_changed.emit(
            "Video processing started"
        )

        target_classes = [
            0,  # Person
            1,  # Bicycle
            2,  # Car
            3,  # Motorcycle
            5,  # Bus
            7,  # Truck
        ]

        frame_number = 0

        while self.running:

            success, frame = video.read()

            if not success:
                print("WORKER: Video ended")
                break

            frame_number += 1

            results = self.tracker.track(
                frame,
                classes=target_classes,
            )

            annotated_frame = (
                results[0].plot()
            )

            if frame_number == 1:
                print(
                    "WORKER: First frame processed:",
                    annotated_frame.shape,
                )

            self.frame_ready.emit(
                annotated_frame
            )

        video.release()

        print(
            "WORKER: run() finished after",
            frame_number,
            "frames",
        )

        self.status_changed.emit(
            "Video processing stopped"
        )

        self.finished_processing.emit()

    def stop(self):

        self.running = False
