from ultralytics import YOLO
import cv2

    #  yolo model trained on random data
model = YOLO(r'best.pt') 
cap = cv2.VideoCapture(r'test_video3.mp4')
target_class = [0, 2]

while True:
    ret, frame = cap.read()
    if not ret: break
    results = model(frame, conf=0.1, verbose=False, classes=target_class) 

    annotated_frame = results[0].plot()
    cv2.imshow("Fire Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
