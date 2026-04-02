import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import keras_tuner as kt 
from keras.layers import Dense,  Dropout
from keras.models import Sequential


import kagglehub
path_df = kagglehub.dataset_download("mathchi/diabetes-data-set")

df = pd.read_csv(path_df + "/diabetes.csv")  # filename may vary
df.head()

X = df.drop('Outcome', axis=1)
y = df['Outcome']


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=22, test_size=0.2)

scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)



def build_model_func(hp): 
  model1 = Sequential()
  
  count=0
  for i in range(hp.Int('num_layers', min_value=1, max_value=12)):
      if count==0:
        model1.add(Dense(units=hp.Int('units'+str(i), min_value=6, max_value=128), input_shape=(X.shape[1],), 
                        activation=hp.Choice('activation'+str(i), values=['relu', 'sigmoid', 'tanh'])))
        if hp.Boolean('dropout'):
          model1.add(Dropout(rate=hp.Float('rate'+str(i), min_value=0.1, max_value=0.9)))
      else:
        model1.add(Dense(units=hp.Int('units'+str(i), min_value=6, max_value=128), 
                        activation=hp.Choice('activation'+str(i), values=['relu', 'sigmoid', 'tanh'])))
        if hp.Boolean('dropout'):
          model1.add(Dropout(rate=hp.Float('rate'+str(i), min_value=0.1, max_value=0.9)))
      
      count+=1

  model1.add(Dense(1, activation='sigmoid'))
  model1.compile(optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop', 'momentum', 'sgd']), loss='binary_crossentropy', metrics=['accuracy'])
  return model1



      #  run for only 5 combinations from a total of hundreds of combinations 
tuner = kt.RandomSearch(
    hypermodel=build_model_func,
    objective='val_accuracy',
    max_trials=5,
    directory='my_dir',
    project_name='project_keras_tuner'
)


    # it will show all the parameters we are giving for the model 
tuner.search_space_summary()

    # it will simply train the model 
tuner.search(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

    #  getting best model 
model = tuner.get_best_models(num_models=1)[0]

model.summary()

    #  seeing best parameters for the model 
tuner.get_best_hyperparameters(1)[0].values

    #  resuming the training from the best parametrs 
history1 = model.fit(X_train, y_train, epochs=177, validation_data=(X_test, y_test) ,initial_epoch=5)








sns.lineplot(x=history1.epoch, y=history1.history['accuracy'])
sns.lineplot(x=history1.epoch, y=history1.history['val_accuracy'])
plt.show()


sns.lineplot(x=history1.epoch, y=history1.history['loss'], label='loss')
sns.lineplot(x=history1.epoch, y=history1.history['val_loss'], label='val_loss')
plt.show()
