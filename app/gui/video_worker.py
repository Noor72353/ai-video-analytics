import cv2

from PySide6.QtCore import QThread, Signal

from app.analytics.counter import ObjectCounter
from app.analytics.line_crossing import LineCrossingAnalyzer
from app.detector import ObjectDetector
from app.tracker import ObjectTracker


class VideoWorker(QThread):

    frame_ready = Signal(object)
    object_counts_ready = Signal(object)
    line_crossing_ready = Signal(object, object)
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

        self.detector = ObjectDetector(self.model_path)
        self.tracker = ObjectTracker(self.detector.model)

        self.target_classes = [
            0,  # Person
            1,  # Bicycle
            2,  # Car
            3,  # Motorcycle
            5,  # Bus
            7,  # Truck
        ]

        # Current visible-object counter
        self.counter = ObjectCounter(
            self.target_classes
        )

        # Horizontal line will be placed at 50%
        # of the video height.
        self.line_crossing = None

    def run(self):

        print("WORKER: run() started")

        self.status_changed.emit(
            "Opening video..."
        )

        video = cv2.VideoCapture(
            self.video_path
        )

        if not video.isOpened():

            print(
                "WORKER: Could not open video"
            )

            self.status_changed.emit(
                "ERROR: Could not open video"
            )

            self.finished_processing.emit()
            return

        print("WORKER: Video opened")

        self.status_changed.emit(
            "Video processing started"
        )

        frame_number = 0

        while self.running:

            success, frame = video.read()

            if not success:

                print(
                    "WORKER: Video ended"
                )

                break

            frame_number += 1

            frame_height = frame.shape[0]

            # Horizontal line at the middle
            # of the video frame.
            line_y = frame_height // 2

            # Create LineCrossingAnalyzer once
            # using the current video line.
            if self.line_crossing is None:

                self.line_crossing = (
                    LineCrossingAnalyzer(
                        self.target_classes,
                        line_y,
                    )
                )

            results = self.tracker.track(
                frame,
                classes=self.target_classes,
            )

            result = results[0]

            # -------------------------------------------------
            # OBJECT COUNTING
            # -------------------------------------------------

            detected_class_ids = []

            if result.boxes is not None:

                detected_class_ids = [
                    int(class_id)
                    for class_id in result.boxes.cls.tolist()
                ]

            self.counter.reset()

            self.counter.update(
                detected_class_ids
            )

            counts = self.counter.get_counts()

            self.object_counts_ready.emit(
                counts
            )

            # -------------------------------------------------
            # LINE CROSSING
            # -------------------------------------------------

            track_ids = []
            class_ids = []
            center_ys = []

            if (
                result.boxes is not None
                and result.boxes.id is not None
            ):

                track_ids = [
                    int(track_id)
                    for track_id
                    in result.boxes.id.tolist()
                ]

                class_ids = [
                    int(class_id)
                    for class_id
                    in result.boxes.cls.tolist()
                ]

                boxes = result.boxes.xyxy.tolist()

                center_ys = [
                    int(
                        (box[1] + box[3]) / 2
                    )
                    for box in boxes
                ]

                self.line_crossing.update(
                    track_ids,
                    class_ids,
                    center_ys,
                    line_y,
                )

            entry_counts = (
                self.line_crossing.get_entry_counts()
            )

            exit_counts = (
                self.line_crossing.get_exit_counts()
            )

            self.line_crossing_ready.emit(
                entry_counts,
                exit_counts,
            )

            # -------------------------------------------------
            # DRAW LINE ON VIDEO
            # -------------------------------------------------

            cv2.line(
                frame,
                (0, line_y),
                (frame.shape[1], line_y),
                (0, 255, 255),
                4,
            )

            # Label the line
            cv2.putText(
                frame,
                "ENTRY / EXIT LINE",
                (30, line_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
            )

            # -------------------------------------------------
            # YOLO ANNOTATION
            # -------------------------------------------------

            annotated_frame = result.plot()

            # Redraw the line after YOLO annotation
            # so it remains clearly visible.
            cv2.line(
                annotated_frame,
                (0, line_y),
                (
                    annotated_frame.shape[1],
                    line_y,
                ),
                (0, 255, 255),
                4,
            )

            cv2.putText(
                annotated_frame,
                "ENTRY / EXIT LINE",
                (30, line_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
            )

            if frame_number == 1:

                print(
                    "WORKER: First frame processed:",
                    annotated_frame.shape,
                )

                print(
                    "WORKER: Line position:",
                    line_y,
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
