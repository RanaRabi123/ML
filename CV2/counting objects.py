import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

video_path = r"img and Videos\test_video1.mp4"
cap = cv2.VideoCapture(video_path)

window_name = "YOLO Video Detection"
# fully flexible, resizable native window
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

#  Uncomment the line below if you want it to launch MAXIMIZED automatically
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

global_counted_objects = {}

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break  

    results = model.track(frame, persist=True, verbose=False, conf=0.25)
    
    annotated_frame = frame.copy()
    h, w, _ = annotated_frame.shape

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        
        names = model.names

        for box, conf, class_id, track_id in zip(boxes, confidences, class_ids, track_ids):
            class_name = names[class_id]

            if track_id not in global_counted_objects:
                global_counted_objects[track_id] = class_name

            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            custom_label = f"{class_name} {conf:.1f} (count {track_id})"
            
            (text_w, text_h), _ = cv2.getTextSize(custom_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 255, 0), -1)
            
            cv2.putText(annotated_frame, custom_label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    global_class_counts = {}
    for obj_id, name in global_counted_objects.items():
        global_class_counts[name] = global_class_counts.get(name, 0) + 1

    text_y_position = 30
    right_margin = w - 220  
    
    for class_name, count in global_class_counts.items():
        display_total_text = f"Total {class_name.upper()}s: {count}"
        
        cv2.putText(annotated_frame, display_total_text, (right_margin + 1, text_y_position + 1), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, display_total_text, (right_margin, text_y_position), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        
        text_y_position += 25  

    # 3. FIXED: Removed cv2.resizeWindow completely from inside the loop.
    # This stops the code from overriding you when you try to maximize it.
    cv2.imshow(window_name, annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
