import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

img_path = r'img and Videos\test_image6.jpg'

# print(model.names)    # check classes names , so that we can detect custom objects
# target_class = [7, 5]   #  classes index
# results = model(img_path, classes=target_class)[0]  # Get results for the single image

results = model(img_path)[0]  

# 3. Get the annotated frame (NumPy array)
annotated_image = results.plot()

# 4. Extract original image dimensions (height, width)
height, width, _ = annotated_image.shape

# 5. Define a responsive target display size (e.g., max 800px wide/high)
max_dimension = 800
scale = min(max_dimension / width, max_dimension / height, 1.0)
display_width = int(width * scale)
display_height = int(height * scale)

# 6. Create a resizable window and scale it
window_name = "YOLO Image Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, display_width, display_height)

# 7. Show the image and wait for a key press to close
cv2.imshow(window_name, annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()