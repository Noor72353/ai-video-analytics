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


# Line position as a percentage of video height
LINE_POSITION = 0.50


# Track previous center positions
previous_y = {}


# Entry and Exit counters
entry_counts = {
    class_id: 0
    for class_id in TARGET_CLASSES
}

exit_counts = {
    class_id: 0
    for class_id in TARGET_CLASSES
}


# Prevent counting the same crossing repeatedly
counted_crossings = set()


# Open video
video = cv2.VideoCapture(VIDEO_PATH)


if not video.isOpened():
    print(f"ERROR: Could not open video: {VIDEO_PATH}")
    raise SystemExit


# Display size
DISPLAY_WIDTH = 1000
DISPLAY_HEIGHT = 600


while True:

    # Read next frame
    success, frame = video.read()

    if not success:
        print("Video finished.")
        break


    # Get original frame dimensions
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]


    # Calculate crossing line in ORIGINAL coordinates
    line_y_original = int(frame_height * LINE_POSITION)


    # Run YOLO tracking
    results = model.track(
        frame,
        persist=True,
        verbose=False,
        tracker="bytetrack.yaml",
        classes=list(TARGET_CLASSES.keys()),
    )


    # Draw YOLO annotations
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
    # ENTRY / EXIT DETECTION
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


            # Center of tracked object
            center_y = int(
                (y1 + y2) / 2
            )


            # Check whether this object was seen before
            if track_id in previous_y:

                old_y = previous_y[track_id]


                # ---------------------------------------------
                # TOP -> BOTTOM = ENTRY
                # ---------------------------------------------

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


                # ---------------------------------------------
                # BOTTOM -> TOP = EXIT
                # ---------------------------------------------

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


            # Save current position
            previous_y[track_id] = center_y


    # ---------------------------------------------------------
    # RESIZE FOR DISPLAY
    # ---------------------------------------------------------

    annotated_frame = cv2.resize(
        annotated_frame,
        (
            DISPLAY_WIDTH,
            DISPLAY_HEIGHT,
        ),
    )


    # Calculate line position for DISPLAY coordinates
    line_y_display = int(
        DISPLAY_HEIGHT * LINE_POSITION
    )


    # ---------------------------------------------------------
    # DRAW CROSSING LINE
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
    # DISPLAY LIVE CLASS COUNTS
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
    # DISPLAY ENTRY / EXIT COUNTS
    # ---------------------------------------------------------

    y_position = 270


    for class_id, class_name in TARGET_CLASSES.items():

        entry = entry_counts[class_id]

        exit_count = exit_counts[class_id]


        cv2.putText(
            annotated_frame,
            (
                f"{class_name}  "
                f"Entry: {entry}  "
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
    # SHOW VIDEO
    # ---------------------------------------------------------

    cv2.imshow(
        "AI Video Analytics - Video Test",
        annotated_frame,
    )


    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF


    # Q or ESC = quit
    if key == ord("q") or key == 27:
        break


# Cleanup
video.release()
cv2.destroyAllWindows()
