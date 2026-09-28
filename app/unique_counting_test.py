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

seen_ids = {
    class_name: set()
    for class_name in TARGET_CLASSES.values()
}

if not camera.isOpened():
    print("ERROR: Could not open the webcam.")
    raise SystemExit


while True:
    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read a frame.")
        break

    results = model.track(frame, persist=True, verbose=False)
    result = results[0]

    annotated_frame = result.plot()

    if result.boxes.id is not None:
        tracking_ids = result.boxes.id.int().tolist()
        class_ids = result.boxes.cls.int().tolist()

        for tracking_id, class_id in zip(tracking_ids, class_ids):
            if class_id in TARGET_CLASSES:
                class_name = TARGET_CLASSES[class_id]
                seen_ids[class_name].add(tracking_id)

    y_position = 35

    for class_name, ids in seen_ids.items():
        cv2.putText(
            annotated_frame,
            f"Unique {class_name}: {len(ids)}",
            (20, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        y_position += 30

    cv2.imshow(
        "AI Video Analytics - Unique Object Counting",
        annotated_frame,
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break


camera.release()
cv2.destroyAllWindows()
