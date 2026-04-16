import keras_tuner as kt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Input
from keras import layers
import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image

import kagglehub
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")


train_ds = keras.utils.image_dataset_from_directory(
    validation_split=0.2,
    subset='training',
    directory=f'/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train',
    seed=123,
    shuffle=True,
    image_size=(224, 224),
    batch_size=32
)
            #  validation set was so small , that's y we are using training part 
val_ds = keras.utils.image_dataset_from_directory(
    validation_split=0.2,
    subset='validation',
    directory=f'/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train',
    seed=123,
    shuffle=True,
    image_size=(224, 224),
    batch_size=32
)

test_ds = keras.utils.image_dataset_from_directory(
    directory='/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test',
    shuffle=False,   # important for evaluation
    image_size=(224, 224),
    batch_size=32
)



train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
val_ds   = val_ds.map(lambda x, y: (preprocess_input(x), y))
test_ds  = test_ds.map(lambda x, y: (preprocess_input(x), y))

data_aug = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])


conv_base = ResNet50(weights='imagenet',
                 include_top=False
                 )

conv_base.trainable = False     # it will freeze the convlution layers

conv_base.summary()         # we can see here, that we no need to train the conv parameters

def build_model_func(hp):
  model = Sequential() # Instantiate model inside the function
  model.add(Input(shape=(224, 224, 3)))
  model.add(data_aug)
  model.add(conv_base) # Add conv_base to this new model
  model.add(Flatten())

  for i in range(hp.Int('num_layers', min_value=1, max_value=12)):
      if i == 0:
        model.add(Dense(units=hp.Int(f'units_{i}', min_value=6, max_value=512),
                        activation=hp.Choice(f'activation_{i}', values=['relu', 'sigmoid', 'tanh']),
                        name=f'dense_h_{i}'))
        if hp.Boolean(f'dropout_{i}'):
          model.add(Dropout(rate=hp.Float(f'rate_{i}', min_value=0.1, max_value=0.9), name=f'dropout_h_r_{i}'))
      else:
        if hp.Boolean(f'dense_layers_{i}'):
          model.add(Dense(units=hp.Int(f'units_{i}', min_value=6, max_value=512),
                          activation=hp.Choice(f'activation_{i}', values=['relu', 'sigmoid', 'tanh']),
                          name=f'dense_h_{i}'))
          if hp.Boolean(f'dropout_{i}'):
            model.add(Dropout(rate=hp.Float(f'rate_{i}', min_value=0.1, max_value=0.9), name=f'dropout_h_r_{i}'))

  model.add(Dense(1, activation='sigmoid', name='output_dense'))
  model.compile(optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop', 'sgd']), loss='binary_crossentropy', metrics=['accuracy'])
  return model



#  run for only 99 combinations from a total of hundreds of combinations
tuner = kt.RandomSearch(
    hypermodel=build_model_func,
    objective='val_accuracy',
    max_trials=15,
    directory='my_dir',
    project_name='project_keras_tuner_1'
)

    # it will show all the parameters we are giving for the model
tuner.search_space_summary()


   # it will simply train the model
tuner.search(train_ds, epochs=5, validation_data=test_ds)

best_hp = tuner.get_best_hyperparameters(1)[0]
print(best_hp.values)


best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()







        # use this pretrained model with 89% val_accuracy 
loaded_model = keras.models.load_model('x_ray best_model using feature_extraction.keras')


            # we are plotting it using loaded model 
plt.figure(figsize=(13, 22))
pred_list = []

for images, labels in test_ds.unbatch().shuffle(buffer_size=888).batch(32).take(1):
  for img in images:
    img = np.expand_dims(img, axis=0)
    prediction = loaded_model.predict(img)
    pred_list.append(np.round(prediction)[0][0])
    print(prediction[0][0])

  for k in range(len(labels)):
    plt.subplot(8, 5, k+1)
    plt.title(f'Pred: {['Normal' if pred_list[k] == 0 else 'Pneumonia']},\n Actual: {['Normal' if labels[k] == 0 else 'Pneumonia']}')
    img_normalized = images[k]/255.0
    plt.axis('off')
    plt.imshow(img_normalized)