import cv2
import numpy as np
import os
from datetime import datetime
import pandas as pd
import time

IMAGE_FOLDER_PATH = r"D:\intern preparation\CV2\smart attendence\img"
ATTENDANCE_LOG_FILE = r"D:\intern preparation\CV2\smart attendence\Attendance.csv"
HOLD_SECONDS = 3
CONFIDENCE_THRESHOLD = 60   # lower is strict/good

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()


GREEN  = (0, 255, 0)      # just marked
YELLOW = (0, 200, 255)    # counting down
BLUE   = (255, 150, 0)    # already marked today
RED    = (0, 0, 255)      # unknown


# load, train 
def train_recognizer():
    faces, labels, label_map = [], [], {}

    print("Loading face images...")
    for idx, file_name in enumerate(os.listdir(IMAGE_FOLDER_PATH)):
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg', 'jfif')):
            continue


        path = os.path.join(IMAGE_FOLDER_PATH, file_name)
        raw = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img is None:
            print(f"  Skip (unreadable): {file_name}")
            continue

        # Detect face
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        if len(found) == 0:
            print(f"  Skip (no face): {file_name}")
            continue

        # Crop and store
        x, y, w, h = found[0]
        face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))

        file_stem = os.path.splitext(file_name)[0]  
        parts = file_stem.split('_', 1)             
        roll_no = parts[0]                           
        name = parts[1].title() if len(parts) > 1 else file_stem  

        label_map[idx] = {"name": name, "roll_no": roll_no}
        faces.append(face)
        labels.append(idx)
        print(f"  Loaded: {name}")

    if not faces:
        return False, {}

    recognizer.train(faces, np.array(labels))
    print(f"\nTrained on {len(faces)} face(s): {list(label_map.values())}\n")
    return True, label_map




#-----
def mark_attendance(name, roll_no):
    if not os.path.exists(ATTENDANCE_LOG_FILE):
        pd.DataFrame(columns=["Roll No", "Name", "Time", "Date"]).to_csv(ATTENDANCE_LOG_FILE, index=False)

    df = pd.read_csv(ATTENDANCE_LOG_FILE)
    today = datetime.now().strftime("%Y-%m-%d")

    if df[(df["Roll No"] == roll_no) & (df["Date"] == today)].empty:
        new_row = {
            "Roll No": roll_no,
            "Name": name,
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Date": today
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(ATTENDANCE_LOG_FILE, index=False)
        print(f"Marked: {roll_no} - {name}")



# draw  
def draw_box(frame, x, y, w, h, color, text):
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    cv2.rectangle(frame, (x, y+h-30), (x+w, y+h), color, cv2.FILLED)
    cv2.putText(frame, text, (x+6, y+h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1)

def draw_progress(frame, x, y, w, elapsed):
    progress = min(elapsed / HOLD_SECONDS, 1.0)
    bar_y = y - 12
    cv2.rectangle(frame, (x, bar_y), (x+w, bar_y+8), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, bar_y), (x + int(w * progress), bar_y+8), YELLOW, -1)


# main 
def main():
    ok, label_map = train_recognizer()
    if not ok:
        print("No faces loaded. Exiting.")
        return

    # Load today's already-marked names from CSV
    today = datetime.now().strftime("%Y-%m-%d")
    marked_today = set()
    if os.path.exists(ATTENDANCE_LOG_FILE):
        df = pd.read_csv(ATTENDANCE_LOG_FILE)
        marked_today = set(df[df["Date"] == today]["Name"].tolist())

    cam = cv2.VideoCapture(0)
    timers = {}   # name → time when they first appeared

    print("Webcam started. Press 'q' to quit.\n")

    while True:
        ok, frame = cam.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        visible = set()

        for (x, y, w, h) in faces:
            face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            label, confidence = recognizer.predict(face)

            if confidence >= CONFIDENCE_THRESHOLD:
                draw_box(frame, x, y, w, h, RED, "Unknown")
                continue

            
            roll_no = label_map[label]["roll_no"]
            name = label_map[label]["name"]
            visible.add(name)
            now = time.time()

            if roll_no in marked_today:
                draw_box(frame, x, y, w, h, BLUE, f"{roll_no} {name} - Done")

            else:
                if name not in timers:
                    timers[name] = now

                elapsed = now - timers[name]

                if elapsed >= HOLD_SECONDS:
                    mark_attendance(name)
                    marked_today.add(name)
                    timers.pop(name, None)
                    draw_box(frame, x, y, w, h, GREEN, f"{name} - Marked!")
                else:
                    remaining = HOLD_SECONDS - elapsed
                    draw_box(frame, x, y, w, h, YELLOW, f"{name} - Hold {remaining:.1f}s")
                    draw_progress(frame, x, y, w, elapsed)

        # Reset timer for anyone who left the frame
        for name in set(timers) - visible:
            timers.pop(name)

        cv2.putText(frame, "Look at camera and hold still", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        cv2.imshow("Attendance System - q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()