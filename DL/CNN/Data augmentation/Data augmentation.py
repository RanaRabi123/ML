import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras 
from keras.layers import Dense, Conv2D, AveragePooling2D, Flatten
from keras.models import Sequential
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix
from keras import layers


(X_train, y_train), (X_test, y_test) = mnist.load_data()

        # normalizing pixel values
X_train = X_train/255
X_test = X_test/255

# Expand dimensions to (batch, height, width, 1) and then repeat the channel 3 times
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)
# X_train = np.repeat(X_train, 3, axis=-1)        # used to convert 1 channel into 3 
# X_test = np.repeat(X_test, 3, axis=-1)          # means make grayscale into 3 channel fake grayscale 


proces_pipeline = layers.Pipeline([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2),
    # layers.RandomErasing(0.1),
    layers.RandomPerspective(0.2),
    layers.RandomShear(0.2)
    # layers.RandomGrayscale(0.2)           # it requires input ot be in 3 channels 
])



model = Sequential()
model.add(layers.Input(shape=(28, 28, 1))) # Update input shape to 3 channels, if needed during augmentation 
model.add(proces_pipeline)       # during eachh epoch each image will be augmented by it 

        # use average pooling , and tanh for LeNET
model.add(Conv2D(6, kernel_size=(5, 5), padding='valid', activation='tanh'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid'))
model.add(Conv2D(16, kernel_size=(5, 5),padding='valid', activation='tanh'))
model.add(AveragePooling2D(pool_size=(2,2), strides=2, padding='valid'))

model.add(Flatten())

model.add(Dense(120, activation='tanh'))
model.add(Dense(84, activation='tanh'))
model.add(Dense(10, activation='softmax'))


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=8, validation_data=(X_test, y_test))



y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, y_pred_labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()





sns.lineplot(model.history.history['accuracy'], label='accuracy')
sns.lineplot(model.history.history['val_accuracy'], label='val_accuracy')
sns.lineplot(model.history.history['loss'], label='loss')
sns.lineplot(model.history.history['val_loss'], label='val_loss')
plt.title('Model Accuracy and Loss')
plt.ylabel('Value')
plt.xlabel('Epoch')
plt.legend()
plt.show()






for i in range(10):
  plt.subplot(2, 5, i+1)
  img = X_test[i]
  img_input = np.expand_dims(img, axis=0)

  prediction = model.predict(img_input)
  predicted_class = np.argmax(prediction)

#   print("Predicted:", predicted_class)

  plt.imshow(img.squeeze(), cmap='gray')
  plt.title(f"Predicted: {predicted_class},    Actual: {y_test[i]}")
  plt.axis('off')
  plt.show()