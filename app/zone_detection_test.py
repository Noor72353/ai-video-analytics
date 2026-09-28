import cv2
from ultralytics import YOLO


model = YOLO("yolo11n.pt")
camera = cv2.VideoCapture(0)

TARGET_CLASSES = {
    0: "Person",
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}

# Zone coordinates
ZONE_X1 = 150
ZONE_Y1 = 100
ZONE_X2 = 500
ZONE_Y2 = 400


if not camera.isOpened():
    print("ERROR: Could not open the webcam.")
    raise SystemExit


while True:
    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read a frame.")
        break

    results = model.track(
        frame,
        persist=True,
        verbose=False,
        tracker="bytetrack.yaml",
    )

    result = results[0]
    annotated_frame = result.plot()

    objects_in_zone = 0

    # Draw zone
    cv2.rectangle(
        annotated_frame,
        (ZONE_X1, ZONE_Y1),
        (ZONE_X2, ZONE_Y2),
        (255, 0, 255),
        2,
    )

    cv2.putText(
        annotated_frame,
        "ZONE",
        (ZONE_X1, ZONE_Y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 255),
        2,
    )

    if result.boxes.id is not None:
        tracking_ids = result.boxes.id.int().tolist()
        class_ids = result.boxes.cls.int().tolist()
        boxes = result.boxes.xyxy.tolist()

        for tracking_id, class_id, box in zip(
            tracking_ids,
            class_ids,
            boxes,
        ):
            if class_id not in TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = box

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            if (
                ZONE_X1 <= center_x <= ZONE_X2
                and ZONE_Y1 <= center_y <= ZONE_Y2
            ):
                objects_in_zone += 1

                cv2.circle(
                    annotated_frame,
                    (center_x, center_y),
                    7,
                    (255, 0, 255),
                    -1,
                )

    cv2.putText(
        annotated_frame,
        f"Objects in Zone: {objects_in_zone}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    cv2.imshow(
        "AI Video Analytics - Zone Detection",
        annotated_frame,
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break


camera.release()
cv2.destroyAllWindows()
