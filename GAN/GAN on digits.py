import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# --------------------
# Load Dataset
# --------------------
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 127.5 - 1.0
x_train = np.expand_dims(x_train, axis=-1)

BATCH_SIZE = 128
dataset = tf.data.Dataset.from_tensor_slices(x_train)
dataset = dataset.shuffle(60000).batch(BATCH_SIZE)

# # --------------------
# # Generator
# # --------------------
# generator = tf.keras.Sequential([
#     layers.Dense(128, activation="relu", input_shape=(100,)),
#     layers.Dense(256, activation="relu"),
#     layers.Dense(28 * 28, activation="tanh"),
#     layers.Reshape((28, 28, 1))
# ])

# # --------------------
# # Discriminator
# # --------------------
# discriminator = tf.keras.Sequential([
#     layers.Flatten(input_shape=(28, 28, 1)),
#     layers.Dense(256, activation="relu"),
#     layers.Dense(128, activation="relu"),
#     layers.Dense(1, activation="sigmoid")
# ])

# # --------------------
# # Loss Functions
# # --------------------
# bce = tf.keras.losses.BinaryCrossentropy()

# def discriminator_loss(real_output, fake_output):
#     real_loss = bce(tf.ones_like(real_output), real_output)
#     fake_loss = bce(tf.zeros_like(fake_output), fake_output)
#     return real_loss + fake_loss

# def generator_loss(fake_output):
#     return bce(tf.ones_like(fake_output), fake_output)

# # --------------------
# # Optimizers
# # --------------------
# gen_optimizer = tf.keras.optimizers.Adam(0.0002)
# disc_optimizer = tf.keras.optimizers.Adam(0.0002)

# # --------------------
# # Training Step
# # --------------------
# @tf.function
# def train_step(images):
#     noise = tf.random.normal([BATCH_SIZE, 100])

#     with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

#         fake_images = generator(noise, training=True)

#         real_output = discriminator(images, training=True)
#         fake_output = discriminator(fake_images, training=True)

#         gen_loss = generator_loss(fake_output)
#         disc_loss = discriminator_loss(real_output, fake_output)

#     gen_gradients = gen_tape.gradient(
#         gen_loss,
#         generator.trainable_variables
#     )

#     disc_gradients = disc_tape.gradient(
#         disc_loss,
#         discriminator.trainable_variables
#     )

#     gen_optimizer.apply_gradients(
#         zip(gen_gradients, generator.trainable_variables)
#     )

#     disc_optimizer.apply_gradients(
#         zip(disc_gradients, discriminator.trainable_variables)
#     )

#     return gen_loss, disc_loss

# # --------------------
# # Training Loop
# # --------------------
# EPOCHS = 20

# for epoch in range(EPOCHS):

#     for image_batch in dataset:
#         g_loss, d_loss = train_step(image_batch)

#     print(
#         f"Epoch {epoch+1}/{EPOCHS} | "
#         f"G Loss: {g_loss:.4f} | "
#         f"D Loss: {d_loss:.4f}"
#     )





# # --------------------
# # Generate Image
# # --------------------
# import matplotlib.pyplot as plt

# noise = tf.random.normal([1, 100])
# generated_image = generator(noise, training=False)

# plt.imshow(generated_image[0, :, :, 0], cmap="gray")
# plt.axis("off")
# plt.show()






















import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# --------------------
# Generator
# --------------------
generator = tf.keras.Sequential([
    layers.Input(shape=(100,)),
    layers.Dense(7 * 7 * 256, use_bias=False),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Reshape((7, 7, 256)),

    layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.ReLU(),

    layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.ReLU(),

    layers.Conv2DTranspose(1, (5, 5), strides=(1, 1), padding='same', activation='tanh')
])

# --------------------
# Discriminator
# --------------------
discriminator = tf.keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'),
    layers.LeakyReLU(0.2),
    layers.Dropout(0.3),
    layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
    layers.LeakyReLU(0.2),
    layers.Dropout(0.1),
    layers.Flatten(),
    layers.Dense(1)
])

# --------------------
# Loss Functions
# --------------------
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = bce(tf.ones_like(real_output), real_output)
    fake_loss = bce(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return bce(tf.ones_like(fake_output), fake_output)

# --------------------
# Optimizers
# --------------------
gen_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
disc_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# --------------------
# Training Step
# --------------------
@tf.function
def train_step(images):
    batch_size = tf.shape(images)[0]
    noise = tf.random.normal([batch_size, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        fake_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(fake_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gen_gradients = gen_tape.gradient(
        gen_loss,
        generator.trainable_variables
    )

    disc_gradients = disc_tape.gradient(
        disc_loss,
        discriminator.trainable_variables
    )

    gen_optimizer.apply_gradients(
        zip(gen_gradients, generator.trainable_variables)
    )

    disc_optimizer.apply_gradients(
        zip(disc_gradients, discriminator.trainable_variables)
    )

    return gen_loss, disc_loss

# --------------------
# Training Loop
# --------------------
EPOCHS = 100

for epoch in range(EPOCHS):

    for image_batch in dataset:
        g_loss, d_loss = train_step(image_batch)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"G Loss: {g_loss:.4f} | "
        f"D Loss: {d_loss:.4f}"
    )
    
    noise_plt = tf.random.normal([10, 100])
    generated_image_plt = generator(noise_plt, training=False)
    plt.figure(figsize=(18, 3))
    for i in range(10):
            plt.subplot(1, 10, i+1)

            plt.imshow(generated_image_plt[i, :, :, 0], cmap="gray")
            plt.axis("off")
    plt.show()