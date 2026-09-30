import cv2
from ultralytics import YOLO


# Load YOLO model
model = YOLO("yolo11n.pt")


# Video file
VIDEO_PATH = "test_video.mp4"


# Classes we want to track
TARGET_CLASSES = {
    0: "Person",
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}


# Line position
LINE_POSITION = 0.50


# Zone coordinates in the ORIGINAL video
ZONE_X1_RATIO = 0.20
ZONE_Y1_RATIO = 0.20
ZONE_X2_RATIO = 0.80
ZONE_Y2_RATIO = 0.80


# Track previous positions
previous_y = {}


# Entry / Exit counters
entry_counts = {
    class_id: 0
    for class_id in TARGET_CLASSES
}

exit_counts = {
    class_id: 0
    for class_id in TARGET_CLASSES
}


# Prevent duplicate crossings
counted_crossings = set()


# Display size
DISPLAY_WIDTH = 1000
DISPLAY_HEIGHT = 600


# Open video
video = cv2.VideoCapture(VIDEO_PATH)


if not video.isOpened():
    print(f"ERROR: Could not open video: {VIDEO_PATH}")
    raise SystemExit


while True:

    success, frame = video.read()

    if not success:
        print("Video finished.")
        break


    # Original video dimensions
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]


    # ---------------------------------------------------------
    # LINE POSITION
    # ---------------------------------------------------------

    line_y_original = int(
        frame_height * LINE_POSITION
    )


    # ---------------------------------------------------------
    # ZONE POSITION
    # ---------------------------------------------------------

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


    # ---------------------------------------------------------
    # YOLO TRACKING
    # ---------------------------------------------------------

    results = model.track(
        frame,
        persist=True,
        verbose=False,
        tracker="bytetrack.yaml",
        classes=list(TARGET_CLASSES.keys()),
    )


    annotated_frame = results[0].plot()


    # ---------------------------------------------------------
    # CLASS COUNTS
    # ---------------------------------------------------------

    class_counts = {
        class_id: 0
        for class_id in TARGET_CLASSES
    }


    if results[0].boxes is not None:

        detected_classes = (
            results[0].boxes.cls.int().tolist()
        )

        for class_id in detected_classes:

            if class_id in class_counts:
                class_counts[class_id] += 1


    # ---------------------------------------------------------
    # ZONE COUNTS
    # ---------------------------------------------------------

    zone_counts = {
        class_id: 0
        for class_id in TARGET_CLASSES
    }


    # ---------------------------------------------------------
    # TRACKING + LINE + ZONE
    # ---------------------------------------------------------

    if (
        results[0].boxes is not None
        and results[0].boxes.id is not None
    ):

        track_ids = (
            results[0].boxes.id.int().tolist()
        )

        class_ids = (
            results[0].boxes.cls.int().tolist()
        )

        boxes = (
            results[0].boxes.xyxy.tolist()
        )


        for track_id, class_id, box in zip(
            track_ids,
            class_ids,
            boxes,
        ):

            x1, y1, x2, y2 = box


            # Object center
            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
            )


            # -------------------------------------------------
            # LINE CROSSING
            # -------------------------------------------------

            if track_id in previous_y:

                old_y = previous_y[track_id]


                # Top -> Bottom
                if (
                    old_y < line_y_original
                    and center_y >= line_y_original
                ):

                    crossing = (
                        track_id,
                        "entry",
                    )


                    if crossing not in counted_crossings:

                        entry_counts[class_id] += 1

                        counted_crossings.add(
                            crossing
                        )


                # Bottom -> Top
                elif (
                    old_y > line_y_original
                    and center_y <= line_y_original
                ):

                    crossing = (
                        track_id,
                        "exit",
                    )


                    if crossing not in counted_crossings:

                        exit_counts[class_id] += 1

                        counted_crossings.add(
                            crossing
                        )


            previous_y[track_id] = center_y


            # -------------------------------------------------
            # ZONE DETECTION
            # -------------------------------------------------

            inside_zone = (
                zone_x1 <= center_x <= zone_x2
                and
                zone_y1 <= center_y <= zone_y2
            )


            if inside_zone:

                zone_counts[class_id] += 1


                # Draw center point
                cv2.circle(
                    annotated_frame,
                    (center_x, center_y),
                    7,
                    (255, 0, 255),
                    -1,
                )


    # ---------------------------------------------------------
    # RESIZE
    # ---------------------------------------------------------

    annotated_frame = cv2.resize(
        annotated_frame,
        (
            DISPLAY_WIDTH,
            DISPLAY_HEIGHT,
        ),
    )


    # Convert original coordinates to display coordinates
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


    # ---------------------------------------------------------
    # DRAW LINE
    # ---------------------------------------------------------

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


    # ---------------------------------------------------------
    # DRAW ZONE
    # ---------------------------------------------------------

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


    # ---------------------------------------------------------
    # LIVE CLASS COUNTS
    # ---------------------------------------------------------

    y_position = 35


    for class_id, class_name in TARGET_CLASSES.items():

        count = class_counts[class_id]


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


    # ---------------------------------------------------------
    # ENTRY / EXIT
    # ---------------------------------------------------------

    y_position = 270


    for class_id, class_name in TARGET_CLASSES.items():

        entry = entry_counts[class_id]

        exit_count = exit_counts[class_id]


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


    # ---------------------------------------------------------
    # ZONE COUNTS
    # ---------------------------------------------------------

    y_position = 470


    total_zone_objects = sum(
        zone_counts.values()
    )


    cv2.putText(
        annotated_frame,
        f"Objects in Zone: {total_zone_objects}",
        (20, y_position),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 0, 255),
        3,
    )


    y_position += 30


    for class_id, class_name in TARGET_CLASSES.items():

        zone_count = zone_counts[class_id]


        if zone_count > 0:

            cv2.putText(
                annotated_frame,
                f"{class_name} in Zone: {zone_count}",
                (20, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 255),
                2,
            )


            y_position += 25


    # ---------------------------------------------------------
    # DISPLAY
    # ---------------------------------------------------------

    cv2.imshow(
        "AI Video Analytics - Video Test",
        annotated_frame,
    )


    key = cv2.waitKey(1) & 0xFF


    if key == ord("q") or key == 27:
        break


# Cleanup
video.release()
cv2.destroyAllWindows()
