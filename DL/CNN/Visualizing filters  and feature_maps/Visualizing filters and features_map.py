from keras import Model
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
import math

model = ResNet50()
model.summary()




            #  just getting the conv layers 
for i in range(len(model.layers)):

    if 'conv' not in model.layers[i].name:
        continue

    weights = model.layers[i].get_weights()
    if len(weights) < 2:
    #   # print(f"Skipping layer {i} ({model.layers[i].name}) as it does not have enough weights (expected 2, got {len(weights)}).")
        continue

    filters, biases = weights[0], weights[1]

    if len(filters.shape) > 2:
        print(f'layer number {i}', model.layers[i].name, filters.shape)





                #  randomly picking layer 2 filter's weight adn bias
x = model.layers[2].get_weights()
weights_2, bias_2  = x[0], x[1]

                    #  it will simply show the one filter with 3 channels 
plt.figure(figsize=(7, 7))
for j in range(3):
    plt.title(f'layer number 2 conv1_conv (7, 7, {j}, 64)')
    for i in range(64):
        plt.subplot(8, 8, i+1)
        plt.imshow(weights_2[:, :, j, i], cmap='gray')
        plt.axis('off')
    plt.show()









                #  it is just to see , how a filter see the image
model1  = Model(inputs=model.input, outputs= model.layers[2].output)

image = keras.utils.load_img('cat.jpg', target_size=(224, 224))
image = keras.utils.img_to_array(image)
image = np.expand_dims(image, axis=0)
image = preprocess_input(image)


features = model1.predict(image)

plt.figure(figsize=(12, 12))
for i in range(64):
  plt.subplot(8, 8, i+1)
  plt.imshow(features[0, :, :, i], cmap='gray')
  plt.axis('off')
plt.show()










            #  randomly seeing how some layers see the image 

layers_index = [39, 57, 88, 161]
output = [model.layers[i].output for i in layers_index]
model2 = Model(inputs=model.input, outputs=output)

feature_map = model2.predict(image)



for i in range(len(layers_index)+1):
    plt.figure(figsize=(20, 20))
    plt.title(f'{model.layers[i].name}')
                #  adjusting subplot size 
    rows = math.floor(math.sqrt(len(feature_map[i][0][0][0])))
    columns = math.ceil(len(feature_map[i][0][0][0])/rows)
    
    for j in range(len(feature_map[i][0][0][0])):
        plt.subplot(rows, columns, j+1)
        plt.imshow(feature_map[i][0, :, :, j], cmap='gray')
        plt.axis('off')
    plt.show()
