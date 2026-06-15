import  tensorflow as tf
import numpy as np
from PIL import Image
import io

loaded_model = tf.keras.models.load_model("tumor_detection_model.keras")

tumor_type = {
    0: 'Glioma Tumor',
    1: 'Meningioma Tumor',
    2: 'No Tumor',
    3: 'Pituitary Tumor'
}

def predict_tumor(img_bytes):

    # Convert bytes to PIL image
    img = Image.open(io.BytesIO(img_bytes))
    
    img = img.convert("RGB")        # Convert to RGB
    img = img.resize((224, 224))    # Resize
    img = np.array(img)     # Convert to numpy
    img = tf.keras.applications.resnet50.preprocess_input(img)       # Normalize
    img = np.expand_dims(img, axis=0)   # Add batch dimension

    prediction = loaded_model.predict(img, verbose=0)
    pred_label = np.argmax(prediction[0])

    return tumor_type[pred_label]