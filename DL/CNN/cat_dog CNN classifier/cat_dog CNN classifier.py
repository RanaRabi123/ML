import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
import keras_tuner as kt
import numpy as np
from keras.layers import Dense, Conv2D, Flatten, Dropout , MaxPooling2D, BatchNormalization, Activation
from keras.models import Sequential
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import accuracy_score
import tensorflow as tf


import kagglehub
path = kagglehub.dataset_download("shaunthesheep/microsoft-catsvsdogs-dataset")


train_ds = tf.keras.utils.image_dataset_from_directory(
    directory=f'{path}/PetImages',
    validation_split=0.2,
    subset="training",
    seed=123,
    shuffle=True,
    image_size=(256, 256),
    batch_size=32,
    color_mode="rgb"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    directory=f'{path}/PetImages',
    validation_split=0.2,
    subset="validation",
    seed=123,
    shuffle=True,
    image_size=(256, 256),
    batch_size=32,
    color_mode="rgb"
)





AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.map(lambda x, y: (x / 255.0, y)) \
                   .apply(tf.data.experimental.ignore_errors()) \
                   .prefetch(AUTOTUNE)

val_ds = val_ds.map(lambda x, y: (x / 255.0, y)) \
               .apply(tf.data.experimental.ignore_errors()) \
               .prefetch(AUTOTUNE)






        #  to see the shape of the image
# for images, labels in train_ds.take(1):
#     print(images.shape)
#     print(labels.shape)




                #  keras hyper_parameter tuner to find good parameters
def build_model_func(hp):
  model1 = Sequential()

  count_conv=0
  for i in range(hp.Int('num_layers', min_value=1, max_value=12)):
      if count_conv==0:
        model1.add(Conv2D(filters=hp.Int(f'filters{i}', min_value=6, max_value=128), kernel_size=hp.Choice(f'kernelsize{i}', [3, 4, 5, 6]), input_shape=(256, 256, 3),
                        activation=hp.Choice(f'activation{i}', values=['relu', 'sigmoid', 'tanh']), padding=hp.Choice(f'padding{i}', values=['same', 'valid'])
                          ))
        model1.add(MaxPooling2D(pool_size=hp.Choice(f'poolsize{i}', [2, 3, 4]), strides=hp.Choice(f'stride{i}', [2, 3, 4, 5]),
                                padding=hp.Choice(f'padding{i}', values=['valid', 'same'])))
        if hp.Boolean(f'dropout{i}'):
          model1.add(Dropout(rate=hp.Float(f'rate{i}', min_value=0.1, max_value=0.9)))

      else:
        if hp.Boolean(f'conv_layer{i}'):
          model1.add(Conv2D(filters=hp.Int(f'filters{i}', min_value=6, max_value=128), kernel_size=hp.Choice(f'kernelsize{i}', [3, 4, 5, 6]),
                          activation=hp.Choice(f'activation{i}', values=['relu', 'sigmoid', 'tanh']), padding=hp.Choice(f'padding{i}', values=['same', 'valid'])
                            ))
          model1.add(MaxPooling2D(pool_size=hp.Choice(f'poolsize{i}', [2,3,4]), strides=hp.Choice(f'stride{i}', [2, 3, 4, 5]),
                                  padding=hp.Choice(f'padding{i}', values=['valid', 'same'])))
          if hp.Boolean(f'dropout{i}'):
            model1.add(Dropout(rate=hp.Float(f'rate{i}', min_value=0.1, max_value=0.9)))

      count_conv+=1



  model1.add(Flatten())


  count_dense=0
  for i in range(hp.Int('num_layers', min_value=1, max_value=12)):
      if count_dense==0:
        model1.add(Dense(units=hp.Int(f'units{i}', min_value=6, max_value=128),
                        activation=hp.Choice(f'activation{i}', values=['relu', 'sigmoid', 'tanh'])))
        if hp.Boolean(f'dropout{i}'):
          model1.add(Dropout(rate=hp.Float('rate'+str(i), min_value=0.1, max_value=0.9)))
      else:
        if hp.Boolean(f'dense_layers{i}'):
          model1.add(Dense(units=hp.Int('units'+str(i), min_value=6, max_value=128),
                          activation=hp.Choice('activation'+str(i), values=['relu', 'sigmoid', 'tanh'])))
          if hp.Boolean(f'dropout{i}'):
            model1.add(Dropout(rate=hp.Float('rate'+str(i), min_value=0.1, max_value=0.9)))

      count_dense+=1

  model1.add(Dense(1, activation='sigmoid'))
  model1.compile(optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop', 'sgd']), loss='binary_crossentropy', metrics=['accuracy'])
  return model1




#  run for only 99 combinations from a total of hundreds of combinations
tuner = kt.RandomSearch(
    hypermodel=build_model_func,
    objective='val_accuracy',
    max_trials=99,
    directory='my_dir',
    project_name='project_keras_tuner_2'
)



    # it will show all the parameters we are giving for the model
tuner.search_space_summary()


   # it will simply train the model
tuner.search(train_ds, epochs=5, validation_data=val_ds)


best_hp = tuner.get_best_hyperparameters(1)[0]
print(best_hp.values)


best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()





for images, labels in val_ds.take(1):
    print(images.shape)
    print(labels.shape)
    for i in range(16):

      img_input= np.expand_dims(images[i], axis=0)
      prediction = best_model.predict(img_input)

      key = 'Dog' if prediction >= 0.5 else 'Cat'


      plt.subplot(4, 4, i + 1)      # rows, columns, plot_index
      plt.title(f"Predicted: {key}")
      plt.imshow(images[i].numpy())
      plt.axis('off') 
    plt.show()      











                #  if only from a single image from local system 

# loaded_image = keras.utils.load_img(
#     path='/content/ginger-maine-coon-kitten-running-on-lawn-in-royalty-free-image-1719608142.avif',
#     color_mode='rgb',
#     target_size=(256, 256)
# )
# img_array = np.array(loaded_image)
# img_reshape = np.expand_dims(img_array, axis=0)
# img_normalize = img_reshape/255.0


# prediction = best_model.predict(img_normalize)

# if prediction[0][0] < 0.5:
#     print("Cat 🐱")
# else:
#     print("Dog 🐶")



















                    #  by using the already tarined model with 79% val_accuracy   
                    # if manually with a single image 

# loaded_model= tf.keras.models.load_model(filepath='/content/cat_dog_DL_model.keras')

# loaded_image = keras.utils.load_img(
#     path='/content/ginger-maine-coon-kitten-running-on-lawn-in-royalty-free-image-1719608142.avif',
#     color_mode='rgb',
#     target_size=(256, 256)
# )
# img_array = np.array(loaded_image)
# img_reshape = np.expand_dims(img_array, axis=0)
# img_normalize = img_reshape/255.0


# prediction = loaded_model.predict(img_normalize)

# if prediction[0][0] < 0.5:
#     print("Cat 🐱")
# else:
#     print("Dog 🐶")

