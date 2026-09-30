import sys

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.gui.video_worker import VideoWorker


class MainWindow(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "AI Video Analytics System"
        )

        self.resize(
            1400,
            800,
        )

        self.video_worker = None

        # -------------------------------------------------
        # MAIN WIDGET
        # -------------------------------------------------

        main_widget = QWidget()

        main_layout = QHBoxLayout(
            main_widget
        )

        main_layout.setContentsMargins(
            10,
            10,
            10,
            10,
        )

        main_layout.setSpacing(
            10
        )

        # -------------------------------------------------
        # VIDEO CONTAINER
        # -------------------------------------------------

        video_container = QFrame()

        video_container.setStyleSheet(
            """
            QFrame {
                background-color: #1e1e1e;
                border: 2px solid #444444;
            }
            """
        )

        video_layout = QVBoxLayout(
            video_container
        )

        video_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        self.video_label = QLabel(
            "Video Feed"
        )

        self.video_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.video_label.setStyleSheet(
            """
            QLabel {
                background-color: #1e1e1e;
                color: white;
                font-size: 28px;
                border: none;
            }
            """
        )

        video_layout.addWidget(
            self.video_label
        )

        # -------------------------------------------------
        # ANALYTICS CONTAINER
        # -------------------------------------------------

        analytics_container = QFrame()

        analytics_container.setStyleSheet(
            """
            QFrame {
                background-color: #f4f4f4;
                border: 1px solid #cccccc;
            }
            """
        )

        analytics_container.setMinimumWidth(
            360
        )

        analytics_layout = QVBoxLayout(
            analytics_container
        )

        analytics_layout.setContentsMargins(
            15,
            15,
            15,
            15,
        )

        # -------------------------------------------------
        # SCROLL AREA
        # -------------------------------------------------

        scroll_area = QScrollArea()

        scroll_area.setWidgetResizable(
            True
        )

        scroll_area.setFrameShape(
            QFrame.Shape.NoFrame
        )

        analytics_content = QWidget()

        content_layout = QVBoxLayout(
            analytics_content
        )

        # -------------------------------------------------
        # TITLE
        # -------------------------------------------------

        title = QLabel(
            "AI Video Analytics"
        )

        title.setStyleSheet(
            """
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #222222;
            }
            """
        )

        content_layout.addWidget(
            title
        )

        # -------------------------------------------------
        # VIDEO CONTROLS
        # -------------------------------------------------

        controls_title = QLabel(
            "Video Controls"
        )

        controls_title.setStyleSheet(
            """
            QLabel {
                font-size: 17px;
                font-weight: bold;
                color: #333333;
            }
            """
        )

        content_layout.addWidget(
            controls_title
        )

        self.open_video_button = QPushButton(
            "Open Video"
        )

        self.open_video_button.clicked.connect(
            self.open_video
        )

        content_layout.addWidget(
            self.open_video_button
        )

        self.stop_video_button = QPushButton(
            "Stop Video"
        )

        self.stop_video_button.clicked.connect(
            self.stop_video
        )

        self.stop_video_button.setEnabled(
            False
        )

        content_layout.addWidget(
            self.stop_video_button
        )

        # -------------------------------------------------
        # PROCESSING STATUS
        # -------------------------------------------------

        content_layout.addSpacing(
            15
        )

        status_title = QLabel(
            "Processing Status"
        )

        status_title.setStyleSheet(
            """
            QLabel {
                font-size: 17px;
                font-weight: bold;
                color: #333333;
            }
            """
        )

        content_layout.addWidget(
            status_title
        )

        self.processing_status = QLabel(
            "Ready"
        )

        content_layout.addWidget(
            self.processing_status
        )

        # -------------------------------------------------
        # TRACKING STATUS
        # -------------------------------------------------

        content_layout.addSpacing(
            15
        )

        tracking_title = QLabel(
            "Tracking Status"
        )

        tracking_title.setStyleSheet(
            """
            QLabel {
                font-size: 17px;
                font-weight: bold;
                color: #333333;
            }
            """
        )

        content_layout.addWidget(
            tracking_title
        )

        self.tracking_status = QLabel(
            "Tracking: Ready"
        )

        content_layout.addWidget(
            self.tracking_status
        )

        # -------------------------------------------------
        # OBJECT COUNTS
        # -------------------------------------------------

        content_layout.addSpacing(
            15
        )

        object_title = QLabel(
            "Object Counts"
        )

        object_title.setStyleSheet(
            """
            QLabel {
                font-size: 17px;
                font-weight: bold;
                color: #333333;
            }
            """
        )

        content_layout.addWidget(
            object_title
        )

        self.person_count = QLabel(
            "Person: 0"
        )

        self.bicycle_count = QLabel(
            "Bicycle: 0"
        )

        self.car_count = QLabel(
            "Car: 0"
        )

        self.motorcycle_count = QLabel(
            "Motorcycle: 0"
        )

        self.bus_count = QLabel(
            "Bus: 0"
        )

        self.truck_count = QLabel(
            "Truck: 0"
        )

        content_layout.addWidget(
            self.person_count
        )

        content_layout.addWidget(
            self.bicycle_count
        )

        content_layout.addWidget(
            self.car_count
        )

        content_layout.addWidget(
            self.motorcycle_count
        )

        content_layout.addWidget(
            self.bus_count
        )

        content_layout.addWidget(
            self.truck_count
        )

        # -------------------------------------------------
        # LINE CROSSING
        # -------------------------------------------------

        content_layout.addSpacing(
            15
        )

        line_title = QLabel(
            "Line Crossing"
        )

        line_title.setStyleSheet(
            """
            QLabel {
                font-size: 17px;
                font-weight: bold;
                color: #333333;
            }
            """
        )

        content_layout.addWidget(
            line_title
        )

        self.line_person = QLabel(
            "Person: Entry 0 | Exit 0"
        )

        self.line_bicycle = QLabel(
            "Bicycle: Entry 0 | Exit 0"
        )

        self.line_car = QLabel(
            "Car: Entry 0 | Exit 0"
        )

        self.line_motorcycle = QLabel(
            "Motorcycle: Entry 0 | Exit 0"
        )

        self.line_bus = QLabel(
            "Bus: Entry 0 | Exit 0"
        )

        self.line_truck = QLabel(
            "Truck: Entry 0 | Exit 0"
        )

        content_layout.addWidget(
            self.line_person
        )

        content_layout.addWidget(
            self.line_bicycle
        )

        content_layout.addWidget(
            self.line_car
        )

        content_layout.addWidget(
            self.line_motorcycle
        )

        content_layout.addWidget(
            self.line_bus
        )

        content_layout.addWidget(
            self.line_truck
        )

        # -------------------------------------------------
        # ZONE ANALYTICS
        # -------------------------------------------------

        content_layout.addSpacing(
            15
        )

        zone_title = QLabel(
            "Zone Analytics"
        )

        zone_title.setStyleSheet(
            """
            QLabel {
                font-size: 17px;
                font-weight: bold;
                color: #333333;
            }
            """
        )

        content_layout.addWidget(
            zone_title
        )

        self.zone_total = QLabel(
            "Objects in Zone: 0"
        )

        self.zone_person = QLabel(
            "Person: 0"
        )

        self.zone_bicycle = QLabel(
            "Bicycle: 0"
        )

        self.zone_car = QLabel(
            "Car: 0"
        )

        self.zone_motorcycle = QLabel(
            "Motorcycle: 0"
        )

        self.zone_bus = QLabel(
            "Bus: 0"
        )

        self.zone_truck = QLabel(
            "Truck: 0"
        )

        content_layout.addWidget(
            self.zone_total
        )

        content_layout.addWidget(
            self.zone_person
        )

        content_layout.addWidget(
            self.zone_bicycle
        )

        content_layout.addWidget(
            self.zone_car
        )

        content_layout.addWidget(
            self.zone_motorcycle
        )

        content_layout.addWidget(
            self.zone_bus
        )

        content_layout.addWidget(
            self.zone_truck
        )

        # -------------------------------------------------
        # ZONE ENTRY / EXIT
        # -------------------------------------------------

        content_layout.addSpacing(
            10
        )

        zone_movement_title = QLabel(
            "Zone Entry / Exit"
        )

        zone_movement_title.setStyleSheet(
            """
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #333333;
            }
            """
        )

        content_layout.addWidget(
            zone_movement_title
        )

        self.zone_person_movement = QLabel(
            "Person: In 0 | Out 0"
        )

        self.zone_bicycle_movement = QLabel(
            "Bicycle: In 0 | Out 0"
        )

        self.zone_car_movement = QLabel(
            "Car: In 0 | Out 0"
        )

        self.zone_motorcycle_movement = QLabel(
            "Motorcycle: In 0 | Out 0"
        )

        self.zone_bus_movement = QLabel(
            "Bus: In 0 | Out 0"
        )

        self.zone_truck_movement = QLabel(
            "Truck: In 0 | Out 0"
        )

        content_layout.addWidget(
            self.zone_person_movement
        )

        content_layout.addWidget(
            self.zone_bicycle_movement
        )

        content_layout.addWidget(
            self.zone_car_movement
        )

        content_layout.addWidget(
            self.zone_motorcycle_movement
        )

        content_layout.addWidget(
            self.zone_bus_movement
        )

        content_layout.addWidget(
            self.zone_truck_movement
        )

        # -------------------------------------------------
        # DWELL TIME
        # -------------------------------------------------

        content_layout.addSpacing(
            15
        )

        dwell_title = QLabel(
            "Dwell Time"
        )

        dwell_title.setStyleSheet(
            """
            QLabel {
                font-size: 17px;
                font-weight: bold;
                color: #333333;
            }
            """
        )

        content_layout.addWidget(
            dwell_title
        )

        self.dwell_status = QLabel(
            "Tracked Classes: 0"
        )

        self.dwell_time = QLabel(
            "Maximum Dwell Time: 0.0 s"
        )

        content_layout.addWidget(
            self.dwell_status
        )

        content_layout.addWidget(
            self.dwell_time
        )

        # -------------------------------------------------
        # TRAJECTORY
        # -------------------------------------------------

        content_layout.addSpacing(
            15
        )

        trajectory_title = QLabel(
            "Trajectory Tracking"
        )

        trajectory_title.setStyleSheet(
            """
            QLabel {
                font-size: 17px;
                font-weight: bold;
                color: #333333;
            }
            """
        )

        content_layout.addWidget(
            trajectory_title
        )

        self.trajectory_status = QLabel(
            "Tracked Trajectories: 0"
        )

        self.trajectory_length = QLabel(
            "Trail Length: 30 points"
        )

        content_layout.addWidget(
            self.trajectory_status
        )

        content_layout.addWidget(
            self.trajectory_length
        )

        # -------------------------------------------------
        # ANALYTICS ENGINE
        # -------------------------------------------------

        content_layout.addSpacing(
            15
        )

        engine_title = QLabel(
            "Analytics Engine"
        )

        engine_title.setStyleSheet(
            """
            QLabel {
                font-size: 17px;
                font-weight: bold;
                color: #333333;
            }
            """
        )

        content_layout.addWidget(
            engine_title
        )

        self.engine_status = QLabel(
            "Detection: Ready"
        )

        self.tracker_status = QLabel(
            "Tracker: ByteTrack"
        )

        self.analytics_status = QLabel(
            "Analytics: Ready"
        )

        content_layout.addWidget(
            self.engine_status
        )

        content_layout.addWidget(
            self.tracker_status
        )

        content_layout.addWidget(
            self.analytics_status
        )

        content_layout.addStretch()

        scroll_area.setWidget(
            analytics_content
        )

        analytics_layout.addWidget(
            scroll_area
        )

        # -------------------------------------------------
        # MAIN LAYOUT
        # -------------------------------------------------

        main_layout.addWidget(
            video_container,
            3
        )

        main_layout.addWidget(
            analytics_container,
            1
        )

        self.setCentralWidget(
            main_widget
        )

    # -----------------------------------------------------
    # OPEN VIDEO
    # -----------------------------------------------------

    def open_video(self):

        video_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv)",
        )

        if not video_path:
            return

        self.start_video(
            video_path
        )

    # -----------------------------------------------------
    # START VIDEO
    # -----------------------------------------------------

    def start_video(
        self,
        video_path,
    ):

        self.stop_video()

        self.video_worker = VideoWorker(
            video_path
        )

        self.video_worker.frame_ready.connect(
            self.update_frame
        )

        self.video_worker.object_counts_ready.connect(
            self.update_object_counts
        )

        self.video_worker.line_crossing_ready.connect(
            self.update_line_crossing
        )

        self.video_worker.zone_analytics_ready.connect(
            self.update_zone_analytics
        )

        self.video_worker.dwell_time_ready.connect(
            self.update_dwell_time
        )

        self.video_worker.status_changed.connect(
            self.update_status
        )

        self.video_worker.finished_processing.connect(
            self.video_finished
        )

        self.open_video_button.setEnabled(
            False
        )

        self.stop_video_button.setEnabled(
            True
        )

        self.processing_status.setText(
            "Starting..."
        )

        self.engine_status.setText(
            "Detection: Starting"
        )

        self.tracker_status.setText(
            "Tracker: ByteTrack"
        )

        self.analytics_status.setText(
            "Analytics: Starting"
        )

        self.dwell_status.setText(
            "Tracked Classes: 0"
        )

        self.dwell_time.setText(
            "Maximum Dwell Time: 0.0 s"
        )

        self.video_worker.start()

    # -----------------------------------------------------
    # UPDATE VIDEO FRAME
    # -----------------------------------------------------

    def update_frame(
        self,
        frame,
    ):

        frame_rgb = np.ascontiguousarray(
            frame[:, :, ::-1]
        )

        height, width, channels = (
            frame_rgb.shape
        )

        bytes_per_line = (
            channels * width
        )

        image = QImage(
            frame_rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )

        pixmap = QPixmap.fromImage(
            image
        )

        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self.video_label.setPixmap(
            scaled_pixmap
        )

        self.tracking_status.setText(
            "Tracking: Active"
        )

        self.engine_status.setText(
            "Detection: Active"
        )

        self.analytics_status.setText(
            "Analytics: Active"
        )

    # -----------------------------------------------------
    # UPDATE OBJECT COUNTS
    # -----------------------------------------------------

    def update_object_counts(
        self,
        counts,
    ):

        self.person_count.setText(
            f"Person: {counts.get(0, 0)}"
        )

        self.bicycle_count.setText(
            f"Bicycle: {counts.get(1, 0)}"
        )

        self.car_count.setText(
            f"Car: {counts.get(2, 0)}"
        )

        self.motorcycle_count.setText(
            f"Motorcycle: {counts.get(3, 0)}"
        )

        self.bus_count.setText(
            f"Bus: {counts.get(5, 0)}"
        )

        self.truck_count.setText(
            f"Truck: {counts.get(7, 0)}"
        )

    # -----------------------------------------------------
    # UPDATE LINE CROSSING
    # -----------------------------------------------------

    def update_line_crossing(
        self,
        entry_counts,
        exit_counts,
    ):

        self.line_person.setText(
            "Person: Entry "
            f"{entry_counts.get(0, 0)} | Exit "
            f"{exit_counts.get(0, 0)}"
        )

        self.line_bicycle.setText(
            "Bicycle: Entry "
            f"{entry_counts.get(1, 0)} | Exit "
            f"{exit_counts.get(1, 0)}"
        )

        self.line_car.setText(
            "Car: Entry "
            f"{entry_counts.get(2, 0)} | Exit "
            f"{exit_counts.get(2, 0)}"
        )

        self.line_motorcycle.setText(
            "Motorcycle: Entry "
            f"{entry_counts.get(3, 0)} | Exit "
            f"{exit_counts.get(3, 0)}"
        )

        self.line_bus.setText(
            "Bus: Entry "
            f"{entry_counts.get(5, 0)} | Exit "
            f"{exit_counts.get(5, 0)}"
        )

        self.line_truck.setText(
            "Truck: Entry "
            f"{entry_counts.get(7, 0)} | Exit "
            f"{exit_counts.get(7, 0)}"
        )

    # -----------------------------------------------------
    # UPDATE ZONE ANALYTICS
    # -----------------------------------------------------

    def update_zone_analytics(
        self,
        zone_counts,
        zone_entry_counts,
        zone_exit_counts,
    ):

        # Current total objects inside zone
        zone_total = sum(
            zone_counts.values()
        )

        self.zone_total.setText(
            f"Objects in Zone: {zone_total}"
        )

        # Current objects by class
        self.zone_person.setText(
            f"Person: {zone_counts.get(0, 0)}"
        )

        self.zone_bicycle.setText(
            f"Bicycle: {zone_counts.get(1, 0)}"
        )

        self.zone_car.setText(
            f"Car: {zone_counts.get(2, 0)}"
        )

        self.zone_motorcycle.setText(
            f"Motorcycle: {zone_counts.get(3, 0)}"
        )

        self.zone_bus.setText(
            f"Bus: {zone_counts.get(5, 0)}"
        )

        self.zone_truck.setText(
            f"Truck: {zone_counts.get(7, 0)}"
        )

        # Zone Entry / Exit
        self.zone_person_movement.setText(
            "Person: In "
            f"{zone_entry_counts.get(0, 0)} | Out "
            f"{zone_exit_counts.get(0, 0)}"
        )

        self.zone_bicycle_movement.setText(
            "Bicycle: In "
            f"{zone_entry_counts.get(1, 0)} | Out "
            f"{zone_exit_counts.get(1, 0)}"
        )

        self.zone_car_movement.setText(
            "Car: In "
            f"{zone_entry_counts.get(2, 0)} | Out "
            f"{zone_exit_counts.get(2, 0)}"
        )

        self.zone_motorcycle_movement.setText(
            "Motorcycle: In "
            f"{zone_entry_counts.get(3, 0)} | Out "
            f"{zone_exit_counts.get(3, 0)}"
        )

        self.zone_bus_movement.setText(
            "Bus: In "
            f"{zone_entry_counts.get(5, 0)} | Out "
            f"{zone_exit_counts.get(5, 0)}"
        )

        self.zone_truck_movement.setText(
            "Truck: In "
            f"{zone_entry_counts.get(7, 0)} | Out "
            f"{zone_exit_counts.get(7, 0)}"
        )

    # -----------------------------------------------------
    # UPDATE DWELL TIME
    # -----------------------------------------------------

    def update_dwell_time(
        self,
        dwell_times,
    ):

        tracked_classes = sum(
            1
            for dwell_time in dwell_times.values()
            if dwell_time > 0
        )

        maximum_dwell_time = max(
            dwell_times.values(),
            default=0.0,
        )

        self.dwell_status.setText(
            f"Tracked Classes: {tracked_classes}"
        )

        self.dwell_time.setText(
            "Maximum Dwell Time: "
            f"{maximum_dwell_time:.1f} s"
        )

    # -----------------------------------------------------
    # UPDATE STATUS
    # -----------------------------------------------------

    def update_status(
        self,
        message,
    ):

        self.processing_status.setText(
            message
        )

    # -----------------------------------------------------
    # STOP VIDEO
    # -----------------------------------------------------

    def stop_video(self):

        if self.video_worker is not None:

            if self.video_worker.isRunning():

                self.video_worker.stop()

                self.video_worker.wait()

            self.video_worker = None

        self.open_video_button.setEnabled(
            True
        )

        self.stop_video_button.setEnabled(
            False
        )

        self.processing_status.setText(
            "Ready"
        )

        self.tracking_status.setText(
            "Tracking: Ready"
        )

        self.engine_status.setText(
            "Detection: Ready"
        )

        self.analytics_status.setText(
            "Analytics: Ready"
        )

        self.dwell_status.setText(
            "Tracked Classes: 0"
        )

        self.dwell_time.setText(
            "Maximum Dwell Time: 0.0 s"
        )

    # -----------------------------------------------------
    # VIDEO FINISHED
    # -----------------------------------------------------

    def video_finished(self):

        self.open_video_button.setEnabled(
            True
        )

        self.stop_video_button.setEnabled(
            False
        )

        self.processing_status.setText(
            "Video finished"
        )

    # -----------------------------------------------------
    # WINDOW CLOSE
    # -----------------------------------------------------

    def closeEvent(
        self,
        event,
    ):

        self.stop_video()

        event.accept()


def run_gui():

    app = QApplication(
        sys.argv
    )

    window = MainWindow()

    window.show()

    sys.exit(
        app.exec()
    )
