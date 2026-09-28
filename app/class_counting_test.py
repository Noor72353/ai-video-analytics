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

    counts = {name: 0 for name in TARGET_CLASSES.values()}

    if result.boxes.id is not None:
        class_ids = result.boxes.cls.int().tolist()

        for class_id in class_ids:
            if class_id in TARGET_CLASSES:
                counts[TARGET_CLASSES[class_id]] += 1

    y_position = 35

    for name, count in counts.items():
        cv2.putText(
            annotated_frame,
            f"{name}: {count}",
            (20, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        y_position += 30

    cv2.imshow("AI Video Analytics - Class Counting", annotated_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break


camera.release()
cv2.destroyAllWindows()
