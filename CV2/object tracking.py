import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt") 

cap = cv2.VideoCapture(r"img and Videos\test_video1.mp4")

window_name = 'Real-Time Tracking'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # .track() automatically handles detection and ID retention
    # persist=True forces the model to remember IDs across frames
    # tracker="bytetrack.yaml" uses the low-overhead tracker
    # sized_frames = cv2.resize(frame, (1000, 800))
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=360)

    # Render bounding boxes and tracking IDs onto the image
    annotated_frame = results[0].plot()

    # Display video output window
    cv2.imshow(window_name, annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
