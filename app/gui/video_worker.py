import cv2

from PySide6.QtCore import QThread, Signal

from app.analytics.counter import ObjectCounter
from app.analytics.line_crossing import LineCrossingAnalyzer
from app.analytics.zone import ZoneAnalyzer
from app.analytics.dwell_time import DwellTimeAnalyzer
from app.analytics.trajectory import TrajectoryAnalyzer
from app.detector import ObjectDetector
from app.tracker import ObjectTracker


class VideoWorker(QThread):

    frame_ready = Signal(object)
    object_counts_ready = Signal(object)
    line_crossing_ready = Signal(object, object)
    zone_analytics_ready = Signal(object, object, object)
    dwell_time_ready = Signal(object)
    trajectory_ready = Signal(object)

    status_changed = Signal(str)

    duration_ready = Signal(float)
    position_ready = Signal(float)

    finished_processing = Signal()

    def __init__(
        self,
        source,
        model_path="yolo11n.pt",
        parent=None,
    ):
        super().__init__(parent)

        self.source = source
        self.model_path = model_path

        self.running = True
        self.paused = False

        self.seek_requested = False
        self.seek_position = 0.0

        self.video_duration = 0.0

        self.detector = ObjectDetector(
            self.model_path
        )

        self.tracker = ObjectTracker(
            self.detector.model
        )

        self.target_classes = [
            0,
            1,
            2,
            3,
            5,
            7,
        ]

        self.counter = ObjectCounter(
            self.target_classes
        )

        self.line_crossing = None

        self.zone_analyzer = ZoneAnalyzer(
            self.target_classes
        )

        self.dwell_time_analyzer = (
            DwellTimeAnalyzer(
                self.target_classes
            )
        )

        self.trajectory_analyzer = (
            TrajectoryAnalyzer(
                max_length=30
            )
        )

    def run(self):

        print("WORKER: run() started")

        self.status_changed.emit(
            "Opening video..."
        )

        video = cv2.VideoCapture(
            self.source
        )

        if not video.isOpened():

            print(
                "WORKER: Could not open source"
            )

            self.status_changed.emit(
                "ERROR: Could not open source"
            )

            self.finished_processing.emit()

            return

        print("WORKER: Source opened")

        fps = video.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0 or fps > 240:
            fps = 30.0

        frame_count = video.get(
            cv2.CAP_PROP_FRAME_COUNT
        )

        is_file = (
            self.source != 0
        )

        if (
            is_file
            and frame_count > 0
        ):

            self.video_duration = (
                frame_count / fps
            )

        else:

            self.video_duration = 0.0

        self.duration_ready.emit(
            self.video_duration
        )

        self.status_changed.emit(
            "Video processing started"
        )

        frame_number = 0

        frame_delay_ms = max(
            1,
            int(
                1000 / fps
            ),
        )

        while self.running:

            # --------------------------------------------------
            # PAUSE
            # --------------------------------------------------

            if self.paused:

                self.msleep(30)

                continue

            # --------------------------------------------------
            # SEEK
            # --------------------------------------------------

            if self.seek_requested:

                video.set(
                    cv2.CAP_PROP_POS_MSEC,
                    self.seek_position * 1000,
                )

                frame_number = int(
                    self.seek_position * fps
                )

                self.seek_requested = False

                self.line_crossing = None

                self.zone_analyzer.reset()

                self.dwell_time_analyzer.reset()

                self.trajectory_analyzer.reset()

                continue

            # --------------------------------------------------
            # READ FRAME
            # --------------------------------------------------

            success, frame = video.read()

            if not success:

                print(
                    "WORKER: Video ended"
                )

                break

            frame_number += 1

            if is_file:

                current_position = (
                    frame_number / fps
                )

                current_position = min(
                    current_position,
                    self.video_duration,
                )

                self.position_ready.emit(
                    current_position
                )

            # --------------------------------------------------
            # FRAME GEOMETRY
            # --------------------------------------------------

            frame_height = frame.shape[0]
            frame_width = frame.shape[1]

            line_y = (
                frame_height // 2
            )

            zone_width = int(
                frame_width * 0.70
            )

            zone_height = int(
                frame_height * 0.65
            )

            zone_x1 = int(
                (
                    frame_width
                    - zone_width
                ) / 2
            )

            zone_y1 = int(
                (
                    frame_height
                    - zone_height
                ) / 2
            )

            zone_x2 = (
                zone_x1
                + zone_width
            )

            zone_y2 = (
                zone_y1
                + zone_height
            )

            zone = (
                zone_x1,
                zone_y1,
                zone_x2,
                zone_y2,
            )

            if self.line_crossing is None:

                self.line_crossing = (
                    LineCrossingAnalyzer(
                        self.target_classes,
                        line_y,
                    )
                )

            # --------------------------------------------------
            # TRACKING
            # --------------------------------------------------

            results = self.tracker.track(
                frame,
                classes=self.target_classes,
            )

            result = results[0]

            detected_class_ids = []

            track_ids = []
            class_ids = []
            centers = []
            center_ys = []

            if result.boxes is not None:

                detected_class_ids = [
                    int(class_id)
                    for class_id
                    in result.boxes.cls.tolist()
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

            # --------------------------------------------------
            # OBJECT COUNTS
            # --------------------------------------------------

            self.counter.reset()

            self.counter.update(
                detected_class_ids
            )

            self.object_counts_ready.emit(
                self.counter.get_counts()
            )

            # --------------------------------------------------
            # LINE CROSSING
            # --------------------------------------------------

            self.line_crossing.update(
                track_ids,
                class_ids,
                center_ys,
                line_y,
            )

            self.line_crossing_ready.emit(
                self.line_crossing.get_entry_counts(),
                self.line_crossing.get_exit_counts(),
            )

            # --------------------------------------------------
            # ZONE
            # --------------------------------------------------

            zone_counts = (
                self.zone_analyzer.update(
                    track_ids,
                    class_ids,
                    centers,
                    zone,
                )
            )

            self.zone_analytics_ready.emit(
                zone_counts,
                self.zone_analyzer.get_entry_counts(),
                self.zone_analyzer.get_exit_counts(),
            )

            # --------------------------------------------------
            # DWELL TIME
            # --------------------------------------------------

            self.dwell_time_ready.emit(
                self.dwell_time_analyzer.update(
                    track_ids,
                    class_ids,
                    centers,
                )
            )

            # --------------------------------------------------
            # TRAJECTORIES
            # --------------------------------------------------

            self.trajectory_analyzer.update(
                track_ids,
                centers,
            )

            trajectories = (
                self.trajectory_analyzer
                .get_all_trajectories()
            )

            self.trajectory_ready.emit(
                trajectories
            )

            # --------------------------------------------------
            # ANNOTATION
            # --------------------------------------------------

            annotated_frame = (
                result.plot()
            )

            for trajectory in (
                trajectories.values()
            ):

                if len(trajectory) < 2:
                    continue

                for index in range(
                    1,
                    len(trajectory),
                ):

                    previous_point = (
                        trajectory[
                            index - 1
                        ]
                    )

                    current_point = (
                        trajectory[index]
                    )

                    cv2.line(
                        annotated_frame,
                        previous_point,
                        current_point,
                        (0, 255, 255),
                        3,
                    )

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
                1.2,
                (0, 255, 255),
                3,
            )

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
                1.2,
                (255, 0, 255),
                3,
            )

            # --------------------------------------------------
            # SEND FRAME TO GUI
            # --------------------------------------------------

            self.frame_ready.emit(
                annotated_frame
            )

            # --------------------------------------------------
            # PLAYBACK PACING
            # --------------------------------------------------

            if is_file:

                self.msleep(
                    frame_delay_ms
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

    # ==========================================================
    # PLAYBACK CONTROL
    # ==========================================================

    def pause(self):

        self.paused = True

        self.status_changed.emit(
            "Playback paused"
        )

    def resume(self):

        self.paused = False

        self.status_changed.emit(
            "Video processing"
        )

    def seek(
        self,
        position,
    ):

        if self.source == 0:
            return

        position = max(
            0.0,
            float(position),
        )

        if self.video_duration > 0:

            position = min(
                position,
                self.video_duration,
            )

        self.seek_position = position

        self.seek_requested = True

    def stop(self):

        # IMPORTANT:
        # Resume first so a paused worker can
        # leave its pause loop and terminate.
        self.paused = False

        self.running = False
