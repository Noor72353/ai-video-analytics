import cv2
from ultralytics import YOLO


model = YOLO("yolo11n.pt")
camera = cv2.VideoCapture(0)

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

    object_count = 0

    if result.boxes.id is not None:
        object_count = len(result.boxes.id)

    cv2.putText(
        annotated_frame,
        f"Tracked Objects: {object_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.imshow("AI Video Analytics - Object Counting", annotated_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break


camera.release()
cv2.destroyAllWindows()
