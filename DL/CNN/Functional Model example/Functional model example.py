import kagglehub
k_path = kagglehub.dataset_download("jangedoo/utkface-new")
# sometimes the path is not complete , so check it before using path variable  


import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Input, Dense, GlobalAveragePooling2D
from keras.models import Model
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.xception import Xception
from keras.utils import plot_model

# path = '/root/.cache/kagglehub/datasets/jangedoo/utkface-new/versions/1/UTKFace'
# path = '/kaggle/input/utkface-new/UTKFace'

age, gender, img_path = [], [], []
for file in os.listdir(path):
    age.append(int(file.split('_')[0]))
    gender.append(int(file.split('_')[1]))
    img_path.append(file)

df = pd.DataFrame({'age': age, 'gender': gender, 'img': img_path})
df['age'] = df['age']/df['age'].max()
shuffled = df.sample(frac=1, random_state=0)
train_df = shuffled.iloc[:15000]
val_df   = shuffled.iloc[15000:17500]
test_df  = shuffled.iloc[17500:19000]


train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.2
)
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    directory=path, 
    x_col='img', y_col=['age', 'gender'],
    target_size=(224, 224), 
    class_mode='raw', 
    batch_size=16
)
val_generator = val_test_datagen.flow_from_dataframe(
    val_df, 
    directory=path, 
    x_col='img', y_col=['age', 'gender'],
    target_size=(224, 224), 
    class_mode='raw', 
    batch_size=16
)
test_generator = val_test_datagen.flow_from_dataframe(
    test_df, 
    directory=path, 
    x_col='img', y_col=['age', 'gender'],
    target_size=(224, 224), 
    class_mode='raw', 
    batch_size=16
)



    #  as model is makeing multi-output,  so it expects labels in dictionary format  
def make_dataset(generator):
    def gen():
        for images, labels in generator:
            yield images, {
                'age':    labels[:, 0],
                'gender': labels[:, 1]
            }

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(      # it is just ot tell the geenratr what is coming , menas shape , none means flexible batch 
            tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
            {
                'age':    tf.TensorSpec(shape=(None,), dtype=tf.float32),
                'gender': tf.TensorSpec(shape=(None,), dtype=tf.float32)
            }
        )
    )

train_ds = make_dataset(train_generator)
val_ds   = make_dataset(val_generator)
test_ds  = make_dataset(test_generator)








inputs = Input(shape=(224, 224, 3))

conv_base = Xception(
    weights='imagenet',
    include_top=False,
    input_tensor=inputs
)

conv_base.summary()

conv_base.trainable = True     # it will unfreeze the conv layers
set_trainable=False

for layer in conv_base.layers:
    if ('block13' in layer.name or
        'block14' in layer.name or
        'add_11' in layer.name or 
        'conv2d_3' in layer.name or
        'batch_normalization_3' in layer.name):      # we are training the last two blocks and their extra layers that lies between them 
            layer.trainable=True
    else:
            layer.trainable=False

for layer in conv_base.layers:          # to see which layers will be train
    print(layer.name, layer.trainable)




x = conv_base.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x= Dropout(0.33)(x)

age_branch1  = Dense(256, activation='relu')(x)
dropout_age_1 = Dropout(0.22)(age_branch1)
age_branch2  = Dense(64, activation='relu')(dropout_age_1)
dropout_age_2 = Dropout(0.13)(age_branch2)
age_output  = Dense(1, activation='linear', name='age')(dropout_age_2)

gender_branch1  = Dense(256, activation='relu')(x)
dropout_gender_1 = Dropout(0.22)(gender_branch1)
gender_branch2  = Dense(64, activation='relu')(dropout_gender_1)
dropout_gender_2 = Dropout(0.13)(gender_branch2)
gender_output  = Dense(1, activation='sigmoid', name='gender')(dropout_gender_2)

model = Model(inputs=inputs, outputs=[age_output, gender_output])

            #  it will show the model structure 
plot_model(model, show_shapes=True, show_layer_names=True)      

model.compile(
    optimizer='adam', loss={'age': 'mae', 'gender': 'binary_crossentropy'},
    loss_weights={'age': 0.5, 'gender': 1.0},   # measn how much giving importance to them 
    metrics={'age': 'mae', 'gender': 'accuracy'}  # still track MAE for readability
)


# history_1 = model.fit(
#     train_ds, steps_per_epoch=len(train_generator), validation_data=val_ds, validation_steps=len(val_generator), epochs=15
# )


plt.plot(history_1.epoch, history_1.history['gender_accuracy'], label='gender_accuracy')
plt.plot(history_1.epoch, history_1.history['val_gender_accuracy'], label='val_gender_accuracy')
plt.legend()


plt.plot(history_1.epoch, history_1.history['age_mae'], label='age_mae')
plt.plot(history_1.epoch, history_1.history['val_age_mae'], label='val_age_mae')
plt.legend()














            #  using the pretrained model with 94% val_accuracy , gender is mostly correct , but age is not 
loaded_model = tf.keras.models.load_model('/age_gender xception_model.keras')


                #  to predict the single image  
img_path = '/image.jpeg'
def predict_single_image(img_path):
    img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))  
    img_array = keras.preprocessing.image.img_to_array(img)
    img_expand = tf.expand_dims(img_array, 0)
    img_normalize = img_expand/255.0
    pred = loaded_model.predict(img_normalize)

    plt.title(f'Age: {int(pred[0][0][0]*100)}\n Gender: {"Male" if round(pred[1][0][0])==0 else "Female"}')
    return plt.imshow(img_normalize[0])


predict_single_image(img_path)












            #  seeing the test set 
plt.figure(figsize=(12, 12))
for images, labels in test_ds.take(1):
  prediction = loaded_model.predict(images)

  for i in range(len(images)):
    plt.subplot(4, 4, i + 1)
                        #  predictions made by the model 
    plt.title(f"Pred Age: {int((prediction[0][i][0])*100)} \nPred Gender: {'Male' if round(prediction[1][i][0])==0 else 'Female'}")
    
                        #  comparing actual VS prediction 
    # plt.title(f"Actual Gender: {'Male' if int((labels['gender'][i].numpy()))==0 else 'Female'} \nPred Gender: {'Male' if round(prediction[1][i][0])==0 else 'Female'}")
    # plt.title(f"Pred Age: {int((prediction[0][i][0])*100)} \nActual Age: {int((labels['age'][i].numpy())*100)}")
    plt.imshow(images[i].numpy())
    plt.axis('off')



