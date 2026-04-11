import keras
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')

img_path = 'elephant.jpg'
img = keras.utils.load_img(img_path, target_size=(224, 224))
x = keras.utils.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)         # it will adjust according to model need automatically 

preds = model.predict(x)

# decode the results into a list of tuples (class, description, probability)
prediction = decode_predictions(preds, top=3)

print('Predicted --> : ')
for i in range(3):
    print(f"\t{i+1}. {prediction[0][i][1]}: {prediction[0][i][2]*100:.2f}%")