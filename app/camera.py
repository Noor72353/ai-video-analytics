import cv2


def run_camera():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open the webcam.")
        return

    print("Webcam started.")
    print("Press Q or ESC to exit.")

    while True:
        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read a frame.")
            break

        cv2.imshow("AI Video Analytics - Webcam", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

    camera.release()
    cv2.destroyAllWindows()
    print("Webcam stopped.")
