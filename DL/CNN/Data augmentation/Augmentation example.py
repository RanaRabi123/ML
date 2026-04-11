import keras 
from keras import layers 
# from keras.preprocessing import image
import matplotlib.pyplot as plt


img_path = 'puppy.jpg'
img = keras.utils.load_img(img_path, target_size=(256, 256))

img_array = keras.utils.img_to_array(img)
img_array = img_array/255.0


            #  there are many more options available for it .....
proces_pipeline = layers.Pipeline([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2),
    layers.RandomErasing(0.1),
    layers.RandomPerspective(0.2),
    layers.RandomShear(0.2)
    # layers.RandomGrayscale(0.2)       # for it image should be 3 channel 
])



plt.figure(figsize=(8, 8))
for i in range(9):
    plt.subplot(4, 3, i+1)
    augmented_image= proces_pipeline(img_array)
    plt.imshow(augmented_image)
    plt.axis('off')