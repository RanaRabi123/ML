import cv2
from ultralytics import YOLO

model = YOLO("yolov8x.pt")

video_path = r"img and Videos\test_video1.mp4"
cap = cv2.VideoCapture(video_path)

# 3. Create the resizable window once before the loop
window_name = "YOLO Video Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break  # End of video

    # 4. Run inference filtered by your specific class ID (e.g., 0 for person)
    results = model(frame) # Returns a list with one item

    # 5. Get the annotated frame
    annotated_frame = results[0].plot()

    # 6. Auto-size the window based on the video frame dimensions
    h, w, _ = annotated_frame.shape
    scale = min(800 / w, 800 / h, 1.0)
    cv2.resizeWindow(window_name, int(w * scale), int(h * scale))

    # 7. Display the frame
    cv2.imshow(window_name, annotated_frame)

    # 8. Press 'q' to quit the video playback early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
