from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

            # making them in normal range
X_train = X_train/255
X_test = X_test/255

model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))  # as it is multiclass classification , so softmax is used 

        # sparse loss is used , by using it we did not need to make one hot encode of the labels 
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_info = model.fit(X_train, y_train, epochs=15, validation_split=0.2)

y_prob = model.predict(X_test)  # model predicts probability 
y_pred = y_prob.argmax(axis=1)  # then we will select with larger one probability


score = accuracy_score(y_test, y_pred)
print(score)


        # show ant random data point to see the image 
# sns.heatmap(X_test[0])
# plt.show

# plt.imshow(X_test[0])
# plt.show()


sns.lineplot(y = model_info.history['loss'], x = model_info.epoch)
sns.lineplot(y = model_info.history['val_loss'], x = model_info.epoch)
plt.show()


sns.lineplot(x=model_info.epoch , y = model_info.history['accuracy'])
sns.lineplot(x=model_info.epoch , y = model_info.history['val_accuracy'])
plt.show()

