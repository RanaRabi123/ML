import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from  tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/kaggle/input/datasets/mohansacharya/graduate-admissions/Admission_Predict_Ver1.1.csv')
df.head()

df.drop(columns=['Serial No.'], axis=1, inplace=True)


X = df.iloc[:, 0:-1]
y=df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y , random_state = 22, test_size=0.2)

scale  = MinMaxScaler()     # scaling, as some columns have values in hundreds while others have only in tens
X_train_scale = scale.fit_transform(X_train)
X_test_scale = scale.transform(X_test)


model = Sequential()
model.add(Dense(12, activation='relu', input_shape=(7, )))
model.add(Dense(1, activation='linear'))    # linear function due to continuous value output(regression)

model.summary()

        # MSE loss due ot regression problem 
model.compile(loss='mean_squared_error', optimizer='adam')
model_info = model.fit(X_train_scale, y_train, epochs=55, validation_split=0.2)

y_pred = model.predict(X_test_scale)


score = r2_score(y_test, y_pred)
print(score)

sns.lineplot(x=model_info.epoch,  y=model_info.history['loss'])
sns.lineplot(x=model_info.epoch, y=model_info.history['val_loss'])
plt.show()


