import cv2
from ultralytics import YOLO


def run_camera():
    model = YOLO("yolo11n.pt")
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open the webcam.")
        return

    print("Webcam started with YOLO detection.")
    print("Press Q or ESC to exit.")

    while True:
        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read a frame.")
            break

        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()

        cv2.imshow("AI Video Analytics - YOLO Detection", annotated_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

    camera.release()
    cv2.destroyAllWindows()
    print("Webcam stopped.")
