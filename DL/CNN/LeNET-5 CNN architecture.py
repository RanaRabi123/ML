import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import Dense, Conv2D, AveragePooling2D, Flatten
from keras.models import Sequential
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix


(X_train, y_train), (X_test, y_test) = mnist.load_data()

        # normalizing pixel values 
X_train = X_train/255
X_test = X_test/255

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)


model = Sequential()

        # use average pooling , and tanh for LeNET
model.add(Conv2D(6, kernel_size=(5, 5), padding='valid', activation='tanh', input_shape=(28, 28, 1)))
model.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid'))
model.add(Conv2D(16, kernel_size=(5, 5),padding='valid', activation='tanh'))
model.add(AveragePooling2D(pool_size=(2,2), strides=2, padding='valid'))

model.add(Flatten())

model.add(Dense(120, activation='tanh'))
model.add(Dense(84, activation='tanh'))
model.add(Dense(10, activation='softmax'))


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))



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



