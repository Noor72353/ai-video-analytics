import cv2

from PySide6.QtCore import QThread, Signal

from app.analytics.counter import ObjectCounter
from app.analytics.line_crossing import LineCrossingAnalyzer
from app.analytics.zone import ZoneAnalyzer
from app.analytics.dwell_time import DwellTimeAnalyzer
from app.detector import ObjectDetector
from app.tracker import ObjectTracker


class VideoWorker(QThread):

    frame_ready = Signal(object)
    object_counts_ready = Signal(object)
    line_crossing_ready = Signal(object, object)
    zone_analytics_ready = Signal(object, object, object)
    dwell_time_ready = Signal(object)
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

        self.target_classes = [
            0,  # Person
            1,  # Bicycle
            2,  # Car
            3,  # Motorcycle
            5,  # Bus
            7,  # Truck
        ]

        # Object counting
        self.counter = ObjectCounter(
            self.target_classes
        )

        # Line crossing
        self.line_crossing = None

        # Zone analytics
        self.zone_analyzer = ZoneAnalyzer(
            self.target_classes
        )

        # Dwell time analytics
        self.dwell_time_analyzer = (
            DwellTimeAnalyzer(
                self.target_classes
            )
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
            frame_width = frame.shape[1]

            # -------------------------------------------------
            # LINE POSITION
            # -------------------------------------------------

            line_y = frame_height // 2

            # -------------------------------------------------
            # ZONE
            # -------------------------------------------------

            # Large central rectangular analytics zone.
            zone_width = int(
                frame_width * 0.70
            )

            zone_height = int(
                frame_height * 0.65
            )

            zone_x1 = int(
                (frame_width - zone_width) / 2
            )

            zone_y1 = int(
                (frame_height - zone_height) / 2
            )

            zone_x2 = zone_x1 + zone_width
            zone_y2 = zone_y1 + zone_height

            zone = (
                zone_x1,
                zone_y1,
                zone_x2,
                zone_y2,
            )

            # Create line analyzer once.
            if self.line_crossing is None:

                self.line_crossing = (
                    LineCrossingAnalyzer(
                        self.target_classes,
                        line_y,
                    )
                )

            # -------------------------------------------------
            # YOLO TRACKING
            # -------------------------------------------------

            results = self.tracker.track(
                frame,
                classes=self.target_classes,
            )

            result = results[0]

            # -------------------------------------------------
            # OBJECT DATA
            # -------------------------------------------------

            detected_class_ids = []

            track_ids = []
            class_ids = []
            centers = []
            center_ys = []

            if result.boxes is not None:

                detected_class_ids = [
                    int(class_id)
                    for class_id in result.boxes.cls.tolist()
                ]

                if result.boxes.id is not None:

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

                    boxes = (
                        result.boxes.xyxy.tolist()
                    )

                    for box in boxes:

                        x1, y1, x2, y2 = box

                        center_x = int(
                            (x1 + x2) / 2
                        )

                        center_y = int(
                            (y1 + y2) / 2
                        )

                        centers.append(
                            (
                                center_x,
                                center_y,
                            )
                        )

                        center_ys.append(
                            center_y
                        )

            # -------------------------------------------------
            # OBJECT COUNTING
            # -------------------------------------------------

            self.counter.reset()

            self.counter.update(
                detected_class_ids
            )

            counts = (
                self.counter.get_counts()
            )

            self.object_counts_ready.emit(
                counts
            )

            # -------------------------------------------------
            # LINE CROSSING
            # -------------------------------------------------

            if self.line_crossing is not None:

                self.line_crossing.update(
                    track_ids,
                    class_ids,
                    center_ys,
                    line_y,
                )

            entry_counts = (
                self.line_crossing
                .get_entry_counts()
            )

            exit_counts = (
                self.line_crossing
                .get_exit_counts()
            )

            self.line_crossing_ready.emit(
                entry_counts,
                exit_counts,
            )

            # -------------------------------------------------
            # ZONE ANALYTICS
            # -------------------------------------------------

            zone_counts = (
                self.zone_analyzer.update(
                    track_ids,
                    class_ids,
                    centers,
                    zone,
                )
            )

            zone_entry_counts = (
                self.zone_analyzer
                .get_entry_counts()
            )

            zone_exit_counts = (
                self.zone_analyzer
                .get_exit_counts()
            )

            self.zone_analytics_ready.emit(
                zone_counts,
                zone_entry_counts,
                zone_exit_counts,
            )

            # -------------------------------------------------
            # DWELL TIME ANALYTICS
            # -------------------------------------------------

            dwell_times = (
                self.dwell_time_analyzer.update(
                    track_ids,
                    class_ids,
                    centers,
                )
            )

            self.dwell_time_ready.emit(
                dwell_times
            )

            # -------------------------------------------------
            # YOLO ANNOTATION
            # -------------------------------------------------

            annotated_frame = result.plot()

            # -------------------------------------------------
            # DRAW ENTRY / EXIT LINE
            # -------------------------------------------------

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
                (
                    30,
                    max(
                        line_y - 30,
                        50,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.8,
                (0, 255, 255),
                4,
            )

            # -------------------------------------------------
            # DRAW ANALYTICS ZONE
            # -------------------------------------------------

            cv2.rectangle(
                annotated_frame,
                (zone_x1, zone_y1),
                (zone_x2, zone_y2),
                (255, 0, 255),
                4,
            )

            cv2.putText(
                annotated_frame,
                "ANALYTICS ZONE",
                (
                    zone_x1,
                    max(
                        zone_y1 - 30,
                        50,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.8,
                (255, 0, 255),
                4,
            )

            # -------------------------------------------------
            # DEBUG INFORMATION
            # -------------------------------------------------

            if frame_number == 1:

                print(
                    "WORKER: First frame processed:",
                    annotated_frame.shape,
                )

                print(
                    "WORKER: Line position:",
                    line_y,
                )

                print(
                    "WORKER: Zone:",
                    zone,
                )

            # -------------------------------------------------
            # SEND FRAME TO GUI
            # -------------------------------------------------

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
