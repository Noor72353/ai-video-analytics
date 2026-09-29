import cv2
from ultralytics import YOLO


model = YOLO("yolo11n.pt")

VIDEO_PATH = "test_video.mp4"

video = cv2.VideoCapture(VIDEO_PATH)

if not video.isOpened():
    print(f"ERROR: Could not open video: {VIDEO_PATH}")
    raise SystemExit


while True:
    success, frame = video.read()

    if not success:
        print("Video finished.")
        break

    results = model.track(
        frame,
        persist=True,
        verbose=False,
        tracker="bytetrack.yaml",
        classes=[0, 1, 2, 3, 5, 7],
    )

    annotated_frame = results[0].plot()
    annotated_frame = cv2.resize(annotated_frame, (1000, 600))

    cv2.imshow(
        "AI Video Analytics - Video Test",
        annotated_frame,
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break


video.release()
cv2.destroyAllWindows()
