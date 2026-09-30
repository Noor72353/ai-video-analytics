import cv2

from app.detector import ObjectDetector
from app.tracker import ObjectTracker

from app.analytics.counter import ObjectCounter
from app.analytics.line_crossing import LineCrossingAnalyzer
from app.analytics.zone import ZoneAnalyzer
from app.analytics.trajectory import TrajectoryAnalyzer
from app.analytics.dwell_time import DwellTimeAnalyzer


# ---------------------------------------------------------
# CLASSES WE WANT TO TRACK
# ---------------------------------------------------------

TARGET_CLASSES = {
    0: "Person",
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

VIDEO_PATH = "test_video.mp4"

LINE_POSITION = 0.50

ZONE_X1_RATIO = 0.20
ZONE_Y1_RATIO = 0.20
ZONE_X2_RATIO = 0.80
ZONE_Y2_RATIO = 0.80

DISPLAY_WIDTH = 1000
DISPLAY_HEIGHT = 600


# ---------------------------------------------------------
# DETECTOR AND TRACKER
# ---------------------------------------------------------

detector = ObjectDetector("yolo11n.pt")

tracker = ObjectTracker(
    detector.model
)


# ---------------------------------------------------------
# ANALYTICS MODULES
# ---------------------------------------------------------

counter = ObjectCounter(
    TARGET_CLASSES
)

line_analyzer = LineCrossingAnalyzer(
    TARGET_CLASSES,
    LINE_POSITION
)

zone_analyzer = ZoneAnalyzer(
    TARGET_CLASSES
)

trajectory_analyzer = TrajectoryAnalyzer(
    max_length=30
)

dwell_analyzer = DwellTimeAnalyzer()


# ---------------------------------------------------------
# OPEN VIDEO
# ---------------------------------------------------------

video = cv2.VideoCapture(
    VIDEO_PATH
)


if not video.isOpened():

    print(
        f"ERROR: Could not open video: {VIDEO_PATH}"
    )

    raise SystemExit


# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------

while True:

    success, frame = video.read()


    if not success:

        print("Video finished.")

        break


    # -----------------------------------------------------
    # ORIGINAL VIDEO DIMENSIONS
    # -----------------------------------------------------

    frame_height = frame.shape[0]

    frame_width = frame.shape[1]


    # -----------------------------------------------------
    # LINE POSITION
    # -----------------------------------------------------

    line_y_original = int(
        frame_height * LINE_POSITION
    )


    # -----------------------------------------------------
    # ZONE POSITION
    # -----------------------------------------------------

    zone_x1 = int(
        frame_width * ZONE_X1_RATIO
    )

    zone_y1 = int(
        frame_height * ZONE_Y1_RATIO
    )

    zone_x2 = int(
        frame_width * ZONE_X2_RATIO
    )

    zone_y2 = int(
        frame_height * ZONE_Y2_RATIO
    )


    zone = (
        zone_x1,
        zone_y1,
        zone_x2,
        zone_y2,
    )


    # -----------------------------------------------------
    # YOLO TRACKING
    # -----------------------------------------------------

    results = tracker.track(
        frame,
        classes=list(
            TARGET_CLASSES.keys()
        ),
    )


    annotated_frame = results[0].plot()


    # -----------------------------------------------------
    # CLASS COUNTS
    # -----------------------------------------------------

    counter.reset()


    detected_classes = []


    if results[0].boxes is not None:

        detected_classes = (
            results[0]
            .boxes
            .cls
            .int()
            .tolist()
        )


    counter.update(
        detected_classes
    )


    class_counts = (
        counter.get_counts()
    )


    # -----------------------------------------------------
    # TRACKED OBJECT ANALYTICS
    # -----------------------------------------------------

    zone_counts = {
        class_id: 0
        for class_id in TARGET_CLASSES
    }


    dwell_times = {}


    if (
        results[0].boxes is not None
        and results[0].boxes.id is not None
    ):

        track_ids = (
            results[0]
            .boxes
            .id
            .int()
            .tolist()
        )


        class_ids = (
            results[0]
            .boxes
            .cls
            .int()
            .tolist()
        )


        boxes = (
            results[0]
            .boxes
            .xyxy
            .tolist()
        )


        centers = []

        center_ys = []

        inside_states = []


        # -------------------------------------------------
        # CALCULATE CENTERS
        # -------------------------------------------------

        for box in boxes:

            x1, y1, x2, y2 = box


            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
            )


            centers.append(
                (center_x, center_y)
            )

            center_ys.append(
                center_y
            )


        # -------------------------------------------------
        # LINE CROSSING ANALYTICS
        # -------------------------------------------------

        line_analyzer.update(
            track_ids,
            class_ids,
            center_ys,
            line_y_original,
        )


        entry_counts = (
            line_analyzer
            .get_entry_counts()
        )


        exit_counts = (
            line_analyzer
            .get_exit_counts()
        )


        # -------------------------------------------------
        # ZONE ANALYTICS
        # -------------------------------------------------

        zone_counts = (
            zone_analyzer.update(
                track_ids,
                class_ids,
                centers,
                zone,
            )
        )


        zone_entry_counts = (
            zone_analyzer
            .get_entry_counts()
        )


        zone_exit_counts = (
            zone_analyzer
            .get_exit_counts()
        )


        # -------------------------------------------------
        # TRAJECTORY ANALYTICS
        # -------------------------------------------------

        trajectory_analyzer.update(
            track_ids,
            centers,
        )


        # -------------------------------------------------
        # DETERMINE ZONE STATES
        # -------------------------------------------------

        for center in centers:

            center_x, center_y = center


            inside_zone = (
                zone_x1 <= center_x <= zone_x2
                and
                zone_y1 <= center_y <= zone_y2
            )


            inside_states.append(
                inside_zone
            )


        # -------------------------------------------------
        # DWELL TIME ANALYTICS
        # -------------------------------------------------

        dwell_times = (
            dwell_analyzer.update(
                track_ids,
                inside_states,
            )
        )


        # -------------------------------------------------
        # DRAW ZONE CENTER POINTS
        # -------------------------------------------------

        for center, inside_zone in zip(
            centers,
            inside_states,
        ):

            if inside_zone:

                cv2.circle(
                    annotated_frame,
                    center,
                    7,
                    (255, 0, 255),
                    -1,
                )


    else:

        entry_counts = (
            line_analyzer
            .get_entry_counts()
        )


        exit_counts = (
            line_analyzer
            .get_exit_counts()
        )


        zone_entry_counts = (
            zone_analyzer
            .get_entry_counts()
        )


        zone_exit_counts = (
            zone_analyzer
            .get_exit_counts()
        )


    # -----------------------------------------------------
    # DRAW OBJECT TRAJECTORIES
    # -----------------------------------------------------

    trajectories = (
        trajectory_analyzer
        .get_all_trajectories()
    )


    for track_id, points in trajectories.items():

        for i in range(
            1,
            len(points),
        ):

            cv2.line(
                annotated_frame,
                points[i - 1],
                points[i],
                (0, 255, 255),
                2,
            )


    # -----------------------------------------------------
    # RESIZE
    # -----------------------------------------------------

    annotated_frame = cv2.resize(
        annotated_frame,
        (
            DISPLAY_WIDTH,
            DISPLAY_HEIGHT,
        ),
    )


    # -----------------------------------------------------
    # COORDINATE SCALING
    # -----------------------------------------------------

    scale_x = (
        DISPLAY_WIDTH / frame_width
    )

    scale_y = (
        DISPLAY_HEIGHT / frame_height
    )


    line_y_display = int(
        line_y_original * scale_y
    )


    zone_x1_display = int(
        zone_x1 * scale_x
    )

    zone_y1_display = int(
        zone_y1 * scale_y
    )

    zone_x2_display = int(
        zone_x2 * scale_x
    )

    zone_y2_display = int(
        zone_y2 * scale_y
    )


    # -----------------------------------------------------
    # DRAW LINE
    # -----------------------------------------------------

    cv2.line(
        annotated_frame,
        (0, line_y_display),
        (
            DISPLAY_WIDTH,
            line_y_display,
        ),
        (255, 0, 255),
        3,
    )


    # -----------------------------------------------------
    # DRAW ZONE
    # -----------------------------------------------------

    cv2.rectangle(
        annotated_frame,
        (
            zone_x1_display,
            zone_y1_display,
        ),
        (
            zone_x2_display,
            zone_y2_display,
        ),
        (0, 255, 255),
        3,
    )


    cv2.putText(
        annotated_frame,
        "ANALYTICS ZONE",
        (
            zone_x1_display,
            zone_y1_display - 10,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )


    # -----------------------------------------------------
    # LIVE CLASS COUNTS
    # -----------------------------------------------------

    y_position = 35


    for class_id, class_name in (
        TARGET_CLASSES.items()
    ):

        count = class_counts[
            class_id
        ]


        cv2.putText(
            annotated_frame,
            f"{class_name}: {count}",
            (20, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            3,
        )


        y_position += 35


    # -----------------------------------------------------
    # LINE ENTRY / EXIT
    # -----------------------------------------------------

    y_position = 270


    for class_id, class_name in (
        TARGET_CLASSES.items()
    ):

        entry = entry_counts[
            class_id
        ]

        exit_count = exit_counts[
            class_id
        ]


        cv2.putText(
            annotated_frame,
            (
                f"{class_name} "
                f"Entry: {entry} "
                f"Exit: {exit_count}"
            ),
            (20, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )


        y_position += 30


    # -----------------------------------------------------
    # ZONE DWELL TIME DISPLAY
    # -----------------------------------------------------

    dwell_y_position = 430


    for track_id, elapsed_seconds in (
        dwell_times.items()
    ):

        cv2.putText(
            annotated_frame,
            (
                f"ID {track_id}: "
                f"{elapsed_seconds:.1f}s"
            ),
            (650, dwell_y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 255),
            2,
        )


        dwell_y_position += 25


    # -----------------------------------------------------
    # ZONE OBJECT COUNT
    # -----------------------------------------------------

    total_zone_objects = sum(
        zone_counts.values()
    )


    cv2.putText(
        annotated_frame,
        (
            f"Objects in Zone: "
            f"{total_zone_objects}"
        ),
        (20, 470),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 0, 255),
        3,
    )


    # -----------------------------------------------------
    # ZONE ENTRY / EXIT
    # -----------------------------------------------------

    cv2.putText(
        annotated_frame,
        "ZONE ENTRY / EXIT",
        (20, 505),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 255),
        2,
    )


    y_position = 535


    for class_id, class_name in (
        TARGET_CLASSES.items()
    ):

        zone_entry = zone_entry_counts[
            class_id
        ]

        zone_exit = zone_exit_counts[
            class_id
        ]


        if (
            zone_entry > 0
            or zone_exit > 0
        ):

            cv2.putText(
                annotated_frame,
                (
                    f"{class_name}: "
                    f"In {zone_entry} "
                    f"Out {zone_exit}"
                ),
                (20, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 0, 255),
                2,
            )


            y_position += 22


    # -----------------------------------------------------
    # DISPLAY
    # -----------------------------------------------------

    cv2.imshow(
        "AI Video Analytics - Video Test",
        annotated_frame,
    )


    key = (
        cv2.waitKey(1)
        & 0xFF
    )


    if (
        key == ord("q")
        or key == 27
    ):

        break


# ---------------------------------------------------------
# CLEANUP
# ---------------------------------------------------------

video.release()

cv2.destroyAllWindows()
