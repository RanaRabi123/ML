import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
import  matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/kaggle/input/datasets/muhammadshahidazeem/customer-churn-dataset/customer_churn_dataset-testing-master.csv')
df.head()

df.describe()
df.info()

df.drop(columns=['CustomerID', 'Age'], inplace=True, axis=1)
df = pd.get_dummies(df, columns=['Gender', 'Subscription Type', 'Contract Length'], dtype=int, drop_first=True) 
df.duplicated().sum()



X = df.drop(columns=['Churn'], axis=1)
y= df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33, test_size=0.15)

scale = StandardScaler()
X_train_scale = scale.fit_transform(X_train)
X_test_scale = scale.transform(X_test) 

print("X train shape after scaling ", X_train_scale.shape)



model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(11,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # using sigmoid , bcz target is in 0, 1

model.summary()

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

model_info = model.fit(X_train_scale, y_train, epochs=20, validation_split=0.2)


        # to see the model weights and bias 
# model.layers[0].get_weights()
# model.layers[1].get_weights()
# model.layers[2].get_weights()


y_pred = model.predict(X_test_scale)
y_pred_final = np.where(y_pred>0.5, 1, 0)    # to make prediction between 0 and 1 , as dataste also has between them 

                #  checking the model prediction's accuracy 
score = accuracy_score(y_test, y_pred_final)
print(score)




            # Plotting

print(dir(model_info))      # o see model attributes 
print(model_info.history)   # to model info , such as accuracy, loss with validation 


sns.lineplot(x=model_info.epoch, y=model_info.history['accuracy'])
sns.lineplot(y=model_info.history['val_accuracy'], x=model_info.epoch)
plt.show()


sns.lineplot(y=model_info.history['loss'], x=model_info.epoch)
sns.lineplot(y=model_info.history['val_loss'], x=model_info.epoch)
plt.show()