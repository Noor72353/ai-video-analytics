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

trajectories = {}

MAX_TRAJECTORY_LENGTH = 30


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

            if tracking_id not in trajectories:
                trajectories[tracking_id] = []

            trajectories[tracking_id].append(
                (center_x, center_y)
            )

            if len(trajectories[tracking_id]) > MAX_TRAJECTORY_LENGTH:
                trajectories[tracking_id].pop(0)

            points = trajectories[tracking_id]

            for i in range(1, len(points)):
                cv2.line(
                    annotated_frame,
                    points[i - 1],
                    points[i],
                    (0, 255, 255),
                    2,
                )

    cv2.imshow(
        "AI Video Analytics - Object Trajectory",
        annotated_frame,
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break


camera.release()
cv2.destroyAllWindows()
