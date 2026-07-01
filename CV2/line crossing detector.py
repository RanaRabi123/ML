import cv2
from ultralytics import YOLO
import easyocr

model = YOLO("yolov8n-seg.pt")
cap = cv2.VideoCapture(r'D:\intern preparation\CV2\img and Videos\test_video2.mp4')

    # use to read english/numbers 
reader = easyocr.Reader(['en'], gpu=False) # Set gpu=False if you don't have an Nvidia GPU

LINE_Y = 320
TARGET_CLASS = [7, 5, 3, 1]     #7 truck, 5 bus, 3 motorcycle,  2 car, 1 bicycle, 0 person

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False, classes=TARGET_CLASS)

    overlay = frame.copy()

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        bottom_center_y = y2
        bottom_center_x = (x1 + x2) // 2

        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        status='NORMAL'
        color=(190, 250 , 170)
        plate_text = "Unknown"
        number_plate=''

        if bottom_center_y < LINE_Y:
            color = (150, 140, 240)
            status= 'SUSPICIOUS'


            car_crop = frame[y1:y2, x1:x2]              # 1. Crop the car from the frame using your existing coordinates

            if car_crop.size > 0:
                # 2. Run OCR on the car crop
                results_ocr = reader.readtext(car_crop) # EasyOCR is smart enough to find text/plates inside the crop automatically
                
                for (bbox, text, prob) in results_ocr:
                    if prob > 0.4 and len(text) > 4:            # Filter out low-confidence reads or irrelevant text 
                        plate_text = text.upper().strip()
                        break # Grab the first solid match
                        
                number_plate = f"Plate: {plate_text}"
            
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.putText(overlay, f'{class_name} : {status}', (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(overlay, number_plate, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    cv2.line(frame, (0, LINE_Y), (frame.shape[1], LINE_Y), (255, 0, 0), 2)
    cv2.imshow("Segmentation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
