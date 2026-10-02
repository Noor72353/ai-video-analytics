import sys

import cv2
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QIcon,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
    QToolButton,
)

from app.gui.video_worker import VideoWorker


class VideoDisplay(QLabel):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.current_frame = None

        self.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.setMinimumSize(
            640,
            420,
        )

        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        self.setStyleSheet(
            """
            QLabel {
                background-color: #05080d;
                color: #64748b;
                border: none;
            }
            """
        )

        self.setText(
            "VIDEO FEED"
        )

    def set_frame(self, frame):

        if frame is None:
            return

        self.current_frame = frame.copy()

        self.update()

    def clear_frame(self):

        self.current_frame = None

        self.setText(
            "VIDEO FEED"
        )

        self.update()

    def paintEvent(self, event):

        painter = QPainter(self)

        painter.setRenderHint(
            QPainter.RenderHint.SmoothPixmapTransform
        )

        painter.fillRect(
            self.rect(),
            QColor("#05080d"),
        )

        if self.current_frame is None:

            painter.setPen(
                QColor("#64748b")
            )

            painter.setFont(
                QFont(
                    "Segoe UI",
                    22,
                    QFont.Weight.DemiBold,
                )
            )

            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "VIDEO FEED",
            )

            painter.end()

            return

        frame_rgb = cv2.cvtColor(
            self.current_frame,
            cv2.COLOR_BGR2RGB,
        )

        frame_rgb = np.ascontiguousarray(
            frame_rgb
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
        ).copy()

        pixmap = QPixmap.fromImage(
            image
        )

        scaled_pixmap = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        x = (
            self.width()
            - scaled_pixmap.width()
        ) // 2

        y = (
            self.height()
            - scaled_pixmap.height()
        ) // 2

        painter.drawPixmap(
            x,
            y,
            scaled_pixmap,
        )

        painter.end()


class MainWindow(QMainWindow):

    def __init__(self):

        super().__init__()

        # -------------------------------------------------
        # WINDOW
        # -------------------------------------------------

        self.setWindowTitle(
            "AI Video Analytics System"
        )

        self.resize(
            1500,
            900,
        )

        self.setMinimumSize(
            1100,
            700,
        )

        # -------------------------------------------------
        # APPLICATION ICON
        # IMPORTANT:
        # Create this BEFORE build_ui()
        # -------------------------------------------------

        self.app_icon = (
            self.create_cv_icon()
        )

        self.setWindowIcon(
            self.app_icon
        )

        # -------------------------------------------------
        # STATE
        # -------------------------------------------------

        self.video_worker = None

        self.source_type = None

        self.loaded_video_path = None

        self.video_duration = 0

        self.video_position = 0

        self.is_paused = False

        self.is_video_finished = False

        # -------------------------------------------------
        # BUILD UI
        # -------------------------------------------------

        self.build_ui()

    # =====================================================
    # COMPUTER VISION APPLICATION ICON
    # =====================================================

    def create_cv_icon(self):

        pixmap = QPixmap(
            64,
            64,
        )

        pixmap.fill(
            Qt.GlobalColor.transparent
        )

        painter = QPainter(
            pixmap
        )

        painter.setRenderHint(
            QPainter.RenderHint.Antialiasing
        )

        # Background

        painter.setBrush(
            QBrush(
                QColor("#0b1220")
            )
        )

        painter.setPen(
            QPen(
                QColor("#0ea5e9"),
                2,
            )
        )

        painter.drawRoundedRect(
            4,
            4,
            56,
            56,
            14,
            14,
        )

        # Camera lens

        painter.setBrush(
            QBrush(
                QColor("#111827")
            )
        )

        painter.setPen(
            QPen(
                QColor("#38bdf8"),
                3,
            )
        )

        painter.drawEllipse(
            17,
            17,
            30,
            30,
        )

        # Lens center

        painter.setBrush(
            QBrush(
                QColor("#38bdf8")
            )
        )

        painter.setPen(
            Qt.PenStyle.NoPen
        )

        painter.drawEllipse(
            27,
            27,
            10,
            10,
        )

        # Computer vision bounding box

        painter.setPen(
            QPen(
                QColor("#a78bfa"),
                3,
            )
        )

        painter.drawLine(
            10,
            10,
            20,
            10,
        )

        painter.drawLine(
            10,
            10,
            10,
            20,
        )

        painter.drawLine(
            54,
            10,
            44,
            10,
        )

        painter.drawLine(
            54,
            10,
            54,
            20,
        )

        painter.drawLine(
            10,
            54,
            20,
            54,
        )

        painter.drawLine(
            10,
            54,
            10,
            44,
        )

        painter.drawLine(
            54,
            54,
            44,
            54,
        )

        painter.drawLine(
            54,
            54,
            54,
            44,
        )

        painter.end()

        return QIcon(
            pixmap
        )

    # =====================================================
    # BUILD UI
    # =====================================================

    def build_ui(self):

        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #080c13;
            }

            QWidget {
                font-family: "Segoe UI";
            }

            QFrame#mainCard {
              background-color: #0f172a;
              border: 1px solid #1e293b;
              border-radius: 12px;}

            QFrame#header {
                background-color: #0b111a;
                border-bottom: 1px solid #1e293b;
            }

            QLabel#appTitle {
                color: #f8fafc;
                font-size: 22px;
                font-weight: 700;
            }

            QLabel#appSubtitle {
                color: #64748b;
                font-size: 12px;
            }

            QLabel#sectionTitle {
                color: #f8fafc;
                font-size: 15px;
                font-weight: 700;
            }

            QLabel#cardTitle {
                color: #94a3b8;
                font-size: 11px;
                font-weight: 600;
            }

            QLabel#cardValue {
                color: #f8fafc;
                font-size: 19px;
                font-weight: 700;
            }

            QLabel#normalText {
                color: #cbd5e1;
                font-size: 12px;
            }

            QPushButton {
                background-color: #111827;
                color: #e2e8f0;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 9px 14px;
                font-size: 12px;
                font-weight: 600;
            }

            QPushButton:hover {
                background-color: #172033;
                border-color: #38bdf8;
            }

            QPushButton:pressed {
                background-color: #0f172a;
            }

            QPushButton:disabled {
                color: #475569;
                background-color: #0b111a;
                border-color: #1e293b;
            }

            QToolButton {
                background: transparent;
                color: #f8fafc;
                border: none;
                padding: 0px;
                margin: 0px;
            }

            QToolButton:hover {
                color: #38bdf8;
            }

            QToolButton:disabled {
                color: #475569;
            }

            QSlider#timeline {
                background: transparent;
            }

            QSlider#timeline::groove:horizontal {
                height: 5px;
                background: #334155;
                border-radius: 2px;
            }

            QSlider#timeline::sub-page:horizontal {
                background: #0ea5e9;
                border-radius: 2px;
            }

            QSlider#timeline::handle:horizontal {
                width: 12px;
                height: 12px;
                margin: -4px 0;
                border-radius: 6px;
                background: #f8fafc;
            }

            QScrollArea {
    background-color: #0f172a;
    border: none;
}

QScrollArea > QWidget > QWidget {
    background-color: #0f172a;
}

            QScrollBar:vertical {
                background: #0b111a;
                width: 8px;
                border-radius: 4px;
            }

            QScrollBar::handle:vertical {
                background: #334155;
                border-radius: 4px;
            }

            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
            """
        )

        # -------------------------------------------------
        # CENTRAL WIDGET
        # -------------------------------------------------

        central_widget = QWidget()

        central_layout = QVBoxLayout(
            central_widget
        )

        central_layout.setContentsMargins(
            12,
            12,
            12,
            12,
        )

        central_layout.setSpacing(
            10
        )

        self.setCentralWidget(
            central_widget
        )

        # -------------------------------------------------
        # HEADER
        # -------------------------------------------------

        header = QFrame()

        header.setObjectName(
            "header"
        )

        header.setFixedHeight(
            70
        )

        header_layout = QHBoxLayout(
            header
        )

        header_layout.setContentsMargins(
            18,
            8,
            18,
            8,
        )

        # Header icon

        icon_label = QLabel()

        icon_pixmap = self.app_icon.pixmap(
            42,
            42,
        )

        icon_label.setPixmap(
            icon_pixmap
        )

        header_layout.addWidget(
            icon_label
        )

        # Title

        title_layout = QVBoxLayout()

        title_layout.setSpacing(
            1
        )

        title = QLabel(
            "AI VIDEO ANALYTICS"
        )

        title.setObjectName(
            "appTitle"
        )

        subtitle = QLabel(
            "Real-Time Computer Vision Monitoring"
        )

        subtitle.setObjectName(
            "appSubtitle"
        )

        title_layout.addWidget(
            title
        )

        title_layout.addWidget(
            subtitle
        )

        header_layout.addLayout(
            title_layout
        )

        header_layout.addStretch()

        # Status badge

        self.header_status = QLabel(
            "● SYSTEM READY"
        )

        self.header_status.setStyleSheet(
            """
            QLabel {
                color: #22c55e;
                background-color: #0b2115;
                border: 1px solid #14532d;
                border-radius: 14px;
                padding: 6px 12px;
                font-size: 11px;
                font-weight: 700;
            }
            """
        )

        header_layout.addWidget(
            self.header_status
        )

        central_layout.addWidget(
            header
        )

        # -------------------------------------------------
        # MAIN CONTENT
        # -------------------------------------------------

        content_layout = QHBoxLayout()

        content_layout.setSpacing(
            10
        )

        central_layout.addLayout(
            content_layout,
            1,
        )

        # =================================================
        # VIDEO SIDE
        # =================================================

        video_card = QFrame()

        video_card.setObjectName(
            "mainCard"
        )

        video_layout = QVBoxLayout(
            video_card
        )

        video_layout.setContentsMargins(
            10,
            10,
            10,
            10,
        )

        video_layout.setSpacing(
            8
        )

        # -------------------------------------------------
        # VIDEO STAGE
        # -------------------------------------------------

        self.video_stage = QWidget()

        self.video_stage.setMinimumHeight(
            500
        )

        stage_layout = QGridLayout(
            self.video_stage
        )

        stage_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        stage_layout.setSpacing(
            0
        )

        # Video

        self.video_label = VideoDisplay()

        stage_layout.addWidget(
            self.video_label,
            0,
            0,
        )

        # -------------------------------------------------
        # TRANSPARENT CONTROL OVERLAY
        # -------------------------------------------------

        self.player_overlay = QWidget()

        self.player_overlay.setStyleSheet(
            """
            QWidget {
                background: transparent;
                border: none;
            }
            """
        )

        overlay_layout = QVBoxLayout(
            self.player_overlay
        )

        overlay_layout.setContentsMargins(
            18,
            18,
            18,
            12,
        )

        overlay_layout.setSpacing(
            5
        )

        overlay_layout.addStretch()

        # -------------------------------------------------
        # CENTERED ICON CONTROLS
        # -------------------------------------------------

        controls_row = QHBoxLayout()

        controls_row.setSpacing(
            12
        )

        controls_row.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        # Replay

        self.replay_button = (
            self.create_control_button(
                "↻",
                42,
            )
        )

        self.replay_button.setToolTip(
            "Replay video"
        )

        self.replay_button.clicked.connect(
            self.replay_video
        )

        self.replay_button.setEnabled(
            False
        )

        controls_row.addWidget(
            self.replay_button
        )

        # Play / Pause

        self.play_pause_button = (
            self.create_control_button(
                "▶",
                50,
            )
        )

        self.play_pause_button.setToolTip(
            "Play / Pause"
        )

        self.play_pause_button.clicked.connect(
            self.toggle_pause
        )

        self.play_pause_button.setEnabled(
            False
        )

        controls_row.addWidget(
            self.play_pause_button
        )

        # Stop

        self.stop_button = (
            self.create_control_button(
                "■",
                42,
            )
        )

        self.stop_button.setToolTip(
            "Stop"
        )

        self.stop_button.clicked.connect(
            self.stop_video
        )

        self.stop_button.setEnabled(
            False
        )

        controls_row.addWidget(
            self.stop_button
        )

        overlay_layout.addLayout(
            controls_row
        )

        # -------------------------------------------------
        # TIMELINE
        # -------------------------------------------------

        timeline_row = QHBoxLayout()

        timeline_row.setSpacing(
            8
        )

        self.current_time_label = QLabel(
            "00:00"
        )

        self.current_time_label.setStyleSheet(
            """
            QLabel {
                color: #cbd5e1;
                font-size: 11px;
                background: transparent;
                border: none;
            }
            """
        )

        timeline_row.addWidget(
            self.current_time_label
        )

        self.timeline = QSlider(
            Qt.Orientation.Horizontal
        )

        self.timeline.setObjectName(
            "timeline"
        )

        self.timeline.setRange(
            0,
            1000,
        )

        self.timeline.setValue(
            0
        )

        self.timeline.sliderMoved.connect(
            self.seek_video
        )

        self.timeline.sliderPressed.connect(
            self.timeline_pressed
        )

        self.timeline.sliderReleased.connect(
            self.timeline_released
        )

        timeline_row.addWidget(
            self.timeline,
            1,
        )

        self.total_time_label = QLabel(
            "00:00"
        )

        self.total_time_label.setStyleSheet(
            """
            QLabel {
                color: #cbd5e1;
                font-size: 11px;
                background: transparent;
                border: none;
            }
            """
        )

        timeline_row.addWidget(
            self.total_time_label
        )

        overlay_layout.addLayout(
            timeline_row
        )

        stage_layout.addWidget(
            self.player_overlay,
            0,
            0,
        )

        video_layout.addWidget(
            self.video_stage,
            1,
        )

        # -------------------------------------------------
        # SOURCE BUTTONS
        # -------------------------------------------------

        source_row = QHBoxLayout()

        source_row.setSpacing(
            8
        )

        self.open_video_button = QPushButton(
            "Open Video"
        )

        self.open_video_button.clicked.connect(
            self.open_video
        )

        source_row.addWidget(
            self.open_video_button
        )

        self.camera_button = QPushButton(
            "Live Camera"
        )

        self.camera_button.clicked.connect(
            self.start_camera
        )

        source_row.addWidget(
            self.camera_button
        )

        source_row.addStretch()

        self.stop_video_button = QPushButton(
            "Stop Processing"
        )

        self.stop_video_button.clicked.connect(
            self.stop_video
        )

        self.stop_video_button.setEnabled(
            False
        )

        source_row.addWidget(
            self.stop_video_button
        )

        video_layout.addLayout(
            source_row
        )

        # =================================================
        # ANALYTICS SIDE
        # =================================================

        analytics_card = QFrame()

        analytics_card.setObjectName(
            "mainCard"
        )

        analytics_card.setMinimumWidth(
            360
        )

        analytics_layout = QVBoxLayout(
            analytics_card
        )

        analytics_layout.setContentsMargins(
            12,
            12,
            12,
            12,
        )

        analytics_layout.setSpacing(
            8
        )

        # Scroll area

        scroll_area = QScrollArea()

        scroll_area.setWidgetResizable(
            True
        )

        analytics_content = QWidget()

        analytics_content.setStyleSheet(
            """
            QWidget {
                background-color: #0f172a;
            }
            """
        )

        content_layout = QVBoxLayout(
            analytics_content
        )

        content_layout.setContentsMargins(
            4,
            4,
            4,
            4,
        )

        content_layout.setSpacing(
            8
        )
        # -------------------------------------------------
        # ANALYTICS TITLE
        # -------------------------------------------------

        analytics_title = QLabel(
            "LIVE ANALYTICS"
        )

        analytics_title.setObjectName(
            "sectionTitle"
        )

        content_layout.addWidget(
            analytics_title
        )

        # -------------------------------------------------
        # PROCESSING STATUS
        # -------------------------------------------------

        self.processing_status = (
            self.create_info_label(
                "STATUS",
                "Ready",
            )
        )

        content_layout.addWidget(
            self.processing_status
        )

        self.tracking_status = (
            self.create_info_label(
                "TRACKING",
                "Tracking: Ready",
            )
        )

        content_layout.addWidget(
            self.tracking_status
        )

        # -------------------------------------------------
        # OBJECT DETECTION
        # -------------------------------------------------

        object_title = QLabel(
            "OBJECT DETECTION"
        )

        object_title.setObjectName(
            "sectionTitle"
        )

        content_layout.addWidget(
            object_title
        )

        self.person_count = (
            self.create_info_label(
                "PERSON",
                "Person: 0",
            )
        )

        self.bicycle_count = (
            self.create_info_label(
                "BICYCLE",
                "Bicycle: 0",
            )
        )

        self.car_count = (
            self.create_info_label(
                "CAR",
                "Car: 0",
            )
        )

        self.motorcycle_count = (
            self.create_info_label(
                "MOTORCYCLE",
                "Motorcycle: 0",
            )
        )

        self.bus_count = (
            self.create_info_label(
                "BUS",
                "Bus: 0",
            )
        )

        self.truck_count = (
            self.create_info_label(
                "TRUCK",
                "Truck: 0",
            )
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

        line_title = QLabel(
            "LINE CROSSING"
        )

        line_title.setObjectName(
            "sectionTitle"
        )

        content_layout.addWidget(
            line_title
        )

        self.line_person = (
            self.create_info_label(
                "PERSON",
                "Person: Entry 0 | Exit 0",
            )
        )

        self.line_bicycle = (
            self.create_info_label(
                "BICYCLE",
                "Bicycle: Entry 0 | Exit 0",
            )
        )

        self.line_car = (
            self.create_info_label(
                "CAR",
                "Car: Entry 0 | Exit 0",
            )
        )

        self.line_motorcycle = (
            self.create_info_label(
                "MOTORCYCLE",
                "Motorcycle: Entry 0 | Exit 0",
            )
        )

        self.line_bus = (
            self.create_info_label(
                "BUS",
                "Bus: Entry 0 | Exit 0",
            )
        )

        self.line_truck = (
            self.create_info_label(
                "TRUCK",
                "Truck: Entry 0 | Exit 0",
            )
        )

        for widget in [
            self.line_person,
            self.line_bicycle,
            self.line_car,
            self.line_motorcycle,
            self.line_bus,
            self.line_truck,
        ]:
            content_layout.addWidget(
                widget
            )

        # -------------------------------------------------
        # ZONE ANALYTICS
        # -------------------------------------------------

        zone_title = QLabel(
            "ZONE ANALYTICS"
        )

        zone_title.setObjectName(
            "sectionTitle"
        )

        content_layout.addWidget(
            zone_title
        )

        self.zone_total = (
            self.create_info_label(
                "ZONE",
                "Objects in Zone: 0",
            )
        )

        self.zone_person = (
            self.create_info_label(
                "PERSON",
                "Person: 0",
            )
        )

        self.zone_bicycle = (
            self.create_info_label(
                "BICYCLE",
                "Bicycle: 0",
            )
        )

        self.zone_car = (
            self.create_info_label(
                "CAR",
                "Car: 0",
            )
        )

        self.zone_motorcycle = (
            self.create_info_label(
                "MOTORCYCLE",
                "Motorcycle: 0",
            )
        )

        self.zone_bus = (
            self.create_info_label(
                "BUS",
                "Bus: 0",
            )
        )

        self.zone_truck = (
            self.create_info_label(
                "TRUCK",
                "Truck: 0",
            )
        )

        for widget in [
            self.zone_total,
            self.zone_person,
            self.zone_bicycle,
            self.zone_car,
            self.zone_motorcycle,
            self.zone_bus,
            self.zone_truck,
        ]:
            content_layout.addWidget(
                widget
            )

        # -------------------------------------------------
        # ZONE ENTRY / EXIT
        # -------------------------------------------------

        zone_movement_title = QLabel(
            "ZONE ENTRY / EXIT"
        )

        zone_movement_title.setObjectName(
            "sectionTitle"
        )

        content_layout.addWidget(
            zone_movement_title
        )

        self.zone_person_movement = (
            self.create_info_label(
                "PERSON",
                "Person: In 0 | Out 0",
            )
        )

        self.zone_bicycle_movement = (
            self.create_info_label(
                "BICYCLE",
                "Bicycle: In 0 | Out 0",
            )
        )

        self.zone_car_movement = (
            self.create_info_label(
                "CAR",
                "Car: In 0 | Out 0",
            )
        )

        self.zone_motorcycle_movement = (
            self.create_info_label(
                "MOTORCYCLE",
                "Motorcycle: In 0 | Out 0",
            )
        )

        self.zone_bus_movement = (
            self.create_info_label(
                "BUS",
                "Bus: In 0 | Out 0",
            )
        )

        self.zone_truck_movement = (
            self.create_info_label(
                "TRUCK",
                "Truck: In 0 | Out 0",
            )
        )

        for widget in [
            self.zone_person_movement,
            self.zone_bicycle_movement,
            self.zone_car_movement,
            self.zone_motorcycle_movement,
            self.zone_bus_movement,
            self.zone_truck_movement,
        ]:
            content_layout.addWidget(
                widget
            )

        # -------------------------------------------------
        # DWELL TIME
        # -------------------------------------------------

        dwell_title = QLabel(
            "DWELL TIME"
        )

        dwell_title.setObjectName(
            "sectionTitle"
        )

        content_layout.addWidget(
            dwell_title
        )

        self.dwell_status = (
            self.create_info_label(
                "TRACKED",
                "Tracked Classes: 0",
            )
        )

        self.dwell_time = (
            self.create_info_label(
                "MAX DWELL",
                "Maximum Dwell Time: 0.0 s",
            )
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

        trajectory_title = QLabel(
            "TRAJECTORY TRACKING"
        )

        trajectory_title.setObjectName(
            "sectionTitle"
        )

        content_layout.addWidget(
            trajectory_title
        )

        self.trajectory_status = (
            self.create_info_label(
                "TRACKED",
                "Tracked Trajectories: 0",
            )
        )

        self.trajectory_length = (
            self.create_info_label(
                "TRAIL",
                "Trail Length: 30 points",
            )
        )

        content_layout.addWidget(
            self.trajectory_status
        )

        content_layout.addWidget(
            self.trajectory_length
        )

        # -------------------------------------------------
        # ENGINE
        # -------------------------------------------------

        engine_title = QLabel(
            "ANALYTICS ENGINE"
        )

        engine_title.setObjectName(
            "sectionTitle"
        )

        content_layout.addWidget(
            engine_title
        )

        self.engine_status = (
            self.create_info_label(
                "DETECTION",
                "Detection: Ready",
            )
        )

        self.tracker_status = (
            self.create_info_label(
                "TRACKER",
                "Tracker: ByteTrack",
            )
        )

        self.analytics_status = (
            self.create_info_label(
                "ANALYTICS",
                "Analytics: Ready",
            )
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
        # ADD SIDES
        # -------------------------------------------------

        content_layout_main = QHBoxLayout()

        content_layout_main.setSpacing(
            10
        )

        content_layout_main.addWidget(
            video_card,
            3,
        )

        content_layout_main.addWidget(
            analytics_card,
            1,
        )

        central_layout.addLayout(
            content_layout_main,
            1,
        )

    # =====================================================
    # UI HELPERS
    # =====================================================

    def create_control_button(
        self,
        text,
        size,
    ):

        button = QToolButton()

        button.setText(
            text
        )

        button.setFixedSize(
            size,
            size,
        )

        button.setFont(
            QFont(
                "Segoe UI Symbol",
                18,
                QFont.Weight.Bold,
            )
        )

        button.setStyleSheet(
            """
            QToolButton {
                background: transparent;
                color: #f8fafc;
                border: none;
                border-radius: 21px;
            }

            QToolButton:hover {
                color: #38bdf8;
                background: rgba(15, 23, 42, 80);
            }

            QToolButton:pressed {
                color: #0ea5e9;
            }

            QToolButton:disabled {
                color: #475569;
                background: transparent;
            }
            """
        )

        return button
    def create_info_label(
    self,
    title,
    value,
):

     label = QLabel(
        value
    )

     label.setObjectName(
        "normalText"
    )

     label.setText(
        value
    )

     label.setMinimumHeight(
        30
    )

     label.setAlignment(
        Qt.AlignmentFlag.AlignVCenter
        | Qt.AlignmentFlag.AlignLeft
    )

     label.setStyleSheet(
        """
        QLabel {
            color: #cbd5e1;
            background-color: #111827;
            border: 1px solid #1e293b;
            border-radius: 7px;
            padding: 6px 10px;
            font-size: 11px;
            font-weight: 500;
        }

        QLabel:hover {
            background-color: #151f2f;
            border-color: #334155;
        }
        """
    )

     return label

    # =====================================================
    # OPEN VIDEO
    # =====================================================

    def open_video(self):

        video_path, _ = (
            QFileDialog.getOpenFileName(
                self,
                "Select Video",
                "",
                (
                    "Video Files "
                    "(*.mp4 *.avi *.mov *.mkv)"
                ),
            )
        )

        if not video_path:
            return

        self.loaded_video_path = (
            video_path
        )

        self.start_video(
            video_path,
            "video",
        )

    # =====================================================
    # START CAMERA
    # =====================================================

    def start_camera(self):

        self.loaded_video_path = None

        self.start_video(
            0,
            "camera",
        )

    # =====================================================
    # START VIDEO / CAMERA
    # =====================================================

    def start_video(
        self,
        source,
        source_type="video",
    ):

        self.stop_video(
            reset_display=False
        )

        self.source_type = (
            source_type
        )

        if (
            source_type == "video"
            and isinstance(source, str)
        ):

            self.loaded_video_path = (
                source
            )

        self.is_paused = False

        self.is_video_finished = False

        self.video_position = 0

        self.video_duration = 0

        self.timeline.blockSignals(
            True
        )

        self.timeline.setValue(
            0
        )

        self.timeline.blockSignals(
            False
        )

        self.current_time_label.setText(
            "00:00"
        )

        self.total_time_label.setText(
            "00:00"
        )

        self.play_pause_button.setText(
            "Ⅱ"
        )

        self.play_pause_button.setEnabled(
            True
        )

        self.stop_button.setEnabled(
            True
        )

        if source_type == "video":
            self.replay_button.setEnabled(
                True
            )
        else:
            self.replay_button.setEnabled(
                False
            )

        self.video_worker = (
            VideoWorker(source)
        )

        # -------------------------------------------------
        # FRAME
        # -------------------------------------------------

        self.video_worker.frame_ready.connect(
            self.update_frame
        )

        # -------------------------------------------------
        # ANALYTICS
        # -------------------------------------------------

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

        self.video_worker.trajectory_ready.connect(
            self.update_trajectory
        )

        self.video_worker.status_changed.connect(
            self.update_status
        )

        self.video_worker.finished_processing.connect(
            self.video_finished
        )

        # -------------------------------------------------
        # PLAYBACK SIGNALS
        # -------------------------------------------------

        if hasattr(
            self.video_worker,
            "duration_ready",
        ):

            self.video_worker.duration_ready.connect(
                self.update_duration
            )

        if hasattr(
            self.video_worker,
            "position_ready",
        ):

            self.video_worker.position_ready.connect(
                self.update_position
            )

        if hasattr(
            self.video_worker,
            "paused",
        ):

            try:
                self.video_worker.paused.connect(
                    self.worker_pause_state_changed
                )
            except Exception:
                pass

        # -------------------------------------------------
        # BUTTON STATES
        # -------------------------------------------------

        self.open_video_button.setEnabled(
            False
        )

        self.camera_button.setEnabled(
            False
        )

        self.stop_video_button.setEnabled(
            True
        )

        # -------------------------------------------------
        # STATUS
        # -------------------------------------------------

        if source_type == "camera":

            self.processing_status.setText(
                "Starting camera..."
            )

            self.header_status.setText(
                "● CAMERA STARTING"
            )

        else:

            self.processing_status.setText(
                "Starting video..."
            )

            self.header_status.setText(
                "● VIDEO STARTING"
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

        self.trajectory_status.setText(
            "Tracked Trajectories: 0"
        )

        self.trajectory_length.setText(
            "Trail Length: 30 points"
        )

        self.video_worker.start()

    # =====================================================
    # UPDATE VIDEO FRAME
    # =====================================================

    def update_frame(
        self,
        frame,
    ):

        self.video_label.set_frame(
            frame
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

        self.header_status.setText(
            "● ANALYZING"
        )

    # =====================================================
    # UPDATE OBJECT COUNTS
    # =====================================================

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

    # =====================================================
    # UPDATE LINE CROSSING
    # =====================================================

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

    # =====================================================
    # UPDATE ZONE ANALYTICS
    # =====================================================

    def update_zone_analytics(
        self,
        zone_counts,
        zone_entry_counts,
        zone_exit_counts,
    ):

        zone_total = sum(
            zone_counts.values()
        )

        self.zone_total.setText(
            f"Objects in Zone: {zone_total}"
        )

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

    # =====================================================
    # UPDATE DWELL TIME
    # =====================================================

    def update_dwell_time(
        self,
        dwell_times,
    ):

        tracked_classes = sum(
            1
            for value in dwell_times.values()
            if value > 0
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

    # =====================================================
    # UPDATE TRAJECTORY
    # =====================================================

    def update_trajectory(
        self,
        trajectories,
    ):

        tracked_trajectories = len(
            trajectories
        )

        self.trajectory_status.setText(
            "Tracked Trajectories: "
            f"{tracked_trajectories}"
        )

        self.trajectory_length.setText(
            "Trail Length: 30 points"
        )

    # =====================================================
    # UPDATE STATUS
    # =====================================================

    def update_status(
        self,
        message,
    ):

        self.processing_status.setText(
            message
        )

    # =====================================================
    # PLAYBACK
    # =====================================================

    def toggle_pause(self):

        if self.video_worker is None:
            return

        if not self.video_worker.isRunning():
            return

        if self.is_paused:

            if hasattr(
                self.video_worker,
                "resume",
            ):

                self.video_worker.resume()

            self.is_paused = False

            self.play_pause_button.setText(
                "Ⅱ"
            )

            self.processing_status.setText(
                "Video processing resumed"
            )

        else:

            if hasattr(
                self.video_worker,
                "pause",
            ):

                self.video_worker.pause()

            self.is_paused = True

            self.play_pause_button.setText(
                "▶"
            )

            self.processing_status.setText(
                "Video processing paused"
            )

    def worker_pause_state_changed(
        self,
        paused,
    ):

        self.is_paused = bool(
            paused
        )

        if self.is_paused:

            self.play_pause_button.setText(
                "▶"
            )

        else:

            self.play_pause_button.setText(
                "Ⅱ"
            )

    def timeline_pressed(self):

        pass

    def timeline_released(self):

        self.seek_video(
            self.timeline.value()
        )

    def seek_video(
        self,
        value,
    ):

        if (
            self.video_worker is None
        ):
            return

        if self.video_duration <= 0:
            return

        position = int(
            (
                value
                / 1000
            )
            * self.video_duration
        )

        if hasattr(
            self.video_worker,
            "seek",
        ):

            self.video_worker.seek(
                position
            )

    # =====================================================
    # DURATION
    # =====================================================

    def update_duration(
        self,
        duration,
    ):

        try:
            duration = int(
                duration
            )
        except (
            TypeError,
            ValueError,
        ):
            return

        self.video_duration = max(
            duration,
            0,
        )

        self.total_time_label.setText(
            self.format_time(
                self.video_duration
            )
        )

    # =====================================================
    # POSITION
    # =====================================================

    def update_position(
        self,
        position,
    ):

        try:
            position = int(
                position
            )
        except (
            TypeError,
            ValueError,
        ):
            return

        self.video_position = max(
            position,
            0,
        )

        self.current_time_label.setText(
            self.format_time(
                self.video_position
            )
        )

        if (
            self.video_duration > 0
        ):

            slider_value = int(
                (
                    self.video_position
                    / self.video_duration
                )
                * 1000
            )

            slider_value = max(
                0,
                min(
                    1000,
                    slider_value,
                ),
            )

            self.timeline.blockSignals(
                True
            )

            self.timeline.setValue(
                slider_value
            )

            self.timeline.blockSignals(
                False
            )

    # =====================================================
    # FORMAT TIME
    # =====================================================

    def format_time(
        self,
        seconds,
    ):

        try:
            seconds = max(
                int(seconds),
                0,
            )
        except (
            TypeError,
            ValueError,
        ):
            seconds = 0

        hours = seconds // 3600

        minutes = (
            seconds % 3600
        ) // 60

        secs = (
            seconds % 60
        )

        if hours > 0:

            return (
                f"{hours:02d}:"
                f"{minutes:02d}:"
                f"{secs:02d}"
            )

        return (
            f"{minutes:02d}:"
            f"{secs:02d}"
        )

    # =====================================================
    # REPLAY VIDEO
    # =====================================================

    def replay_video(self):

        if not self.loaded_video_path:
            return

        video_path = (
            self.loaded_video_path
        )

        self.start_video(
            video_path,
            "video",
        )

    # =====================================================
    # STOP VIDEO / CAMERA
    # =====================================================

    def stop_video(
        self,
        reset_display=True,
    ):

        if self.video_worker is not None:

            if self.video_worker.isRunning():

                self.video_worker.stop()

                self.video_worker.wait()

            self.video_worker = None

        self.source_type = None

        self.is_paused = False

        self.is_video_finished = False

        self.open_video_button.setEnabled(
            True
        )

        self.camera_button.setEnabled(
            True
        )

        self.stop_video_button.setEnabled(
            False
        )

        self.stop_button.setEnabled(
            False
        )

        self.play_pause_button.setEnabled(
            False
        )

        self.play_pause_button.setText(
            "▶"
        )

        if self.loaded_video_path:

            self.replay_button.setEnabled(
                True
            )

        else:

            self.replay_button.setEnabled(
                False
            )

        if reset_display:

            self.video_label.clear_frame()

            self.timeline.blockSignals(
                True
            )

            self.timeline.setValue(
                0
            )

            self.timeline.blockSignals(
                False
            )

            self.current_time_label.setText(
                "00:00"
            )

            self.total_time_label.setText(
                "00:00"
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

        self.trajectory_status.setText(
            "Tracked Trajectories: 0"
        )

        self.trajectory_length.setText(
            "Trail Length: 30 points"
        )

        self.header_status.setText(
            "● SYSTEM READY"
        )

    # =====================================================
    # VIDEO / CAMERA FINISHED
    # =====================================================

    def video_finished(self):

        source_type = (
            self.source_type
        )

        self.open_video_button.setEnabled(
            True
        )

        self.camera_button.setEnabled(
            True
        )

        self.stop_video_button.setEnabled(
            False
        )

        self.stop_button.setEnabled(
            False
        )

        self.play_pause_button.setEnabled(
            False
        )

        self.play_pause_button.setText(
            "▶"
        )

        self.is_paused = False

        self.is_video_finished = True

        if source_type == "camera":

            self.processing_status.setText(
                "Camera stopped"
            )

            self.header_status.setText(
                "● CAMERA STOPPED"
            )

            self.replay_button.setEnabled(
                False
            )

        else:

            self.processing_status.setText(
                "Video finished - press replay"
            )

            self.header_status.setText(
                "● VIDEO COMPLETE"
            )

            self.replay_button.setEnabled(
                self.loaded_video_path
                is not None
            )

    # =====================================================
    # WINDOW CLOSE
    # =====================================================

    def closeEvent(
        self,
        event,
    ):

        self.stop_video(
            reset_display=False
        )

        event.accept()


def run_gui():

    app = QApplication(
        sys.argv
    )

    app.setWindowIcon(
        MainWindow().app_icon
    )

    # Create the real window separately.
    # The temporary MainWindow above would be undesirable,
    # so QApplication icon is also set from the same CV icon
    # through the main window constructor.

    window = MainWindow()

    window.show()

    sys.exit(
        app.exec()
    )
