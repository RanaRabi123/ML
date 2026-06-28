import kagglehub
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

path = kagglehub.dataset_download("shaunthesheep/microsoft-catsvsdogs-dataset")

train_ds = tf.keras.utils.image_dataset_from_directory(
    directory=f'{path}/PetImages/Cat',
    labels=None,
    image_size=(256, 256),
    batch_size=32,
    color_mode="rgb"
)


for image in train_ds.take(1):
    # image = image[0]/64.0 -1.0
    # plt.imshow(image)
    plt.imshow(image[0].numpy().astype("uint8"))
    plt.show()
    # plt.axis("off")
    # plt.show()


import tensorflow as tf
from tensorflow.keras import layers

# Define a preprocessing function to normalize images
def preprocess_image(image):
    # image is already a batch of images from train_ds (batch_size=32)
    # label is already a batch of labels from train_ds
    image = tf.cast(image, tf.float32) # Cast to float32
    image = (image / 127.5) - 1.0      # Normalize to [-1, 1]
    return image # Only return the image batch for the GAN

# The BATCH_SIZE for the GAN training loop should match the dataset's batch size
BATCH_SIZE = 32

# Apply the preprocessing to the train_ds.
# train_ds is already shuffled and batched (with batch_size=32).
# The .map() operation applies preprocess_image to each element (batch) of train_ds.
dataset = train_ds.take(123).map(preprocess_image) \
          .apply(tf.data.experimental.ignore_errors())
dataset = dataset.prefetch(tf.data.AUTOTUNE) # Prefetch for performance










import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


# --------------------
# Generator
# --------------------
generator = tf.keras.Sequential([
    layers.Input(shape=(100,)),
    layers.Dense(64 * 64 * 128, use_bias=False),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Reshape((64, 64, 128)),

    layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.ReLU(),

    layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.ReLU(),

    layers.Conv2DTranspose(3, (5, 5), strides=(1, 1), padding='same', activation='tanh')
])

# --------------------
# Discriminator
# --------------------
discriminator = tf.keras.Sequential([
    layers.Input(shape=(256, 256, 3)),
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

    noise_plt = tf.random.normal([6, 100])
    generated_image_plt = generator(noise_plt, training=False)
    plt.figure(figsize=(18, 7))
    for i in range(6):
            plt.subplot(1, 6, i+1)
            norm_img = (generated_image_plt[i]+1)/2.0

            plt.imshow(norm_img)
            plt.axis("off")
    plt.show()









































import kagglehub
import tensorflow as tf
import matplotlib.pyplot as plt

path = kagglehub.dataset_download("shaunthesheep/microsoft-catsvsdogs-dataset")

train_ds = tf.keras.utils.image_dataset_from_directory(
    directory=f'{path}/PetImages/Cat',
    labels=None,
    image_size=(128, 128),
    batch_size=32,
    color_mode="rgb"
)

# Normalize to [-1, 1]
def preprocess(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1.0
    return image

dataset = train_ds.map(preprocess).shuffle(1000).prefetch(tf.data.AUTOTUNE)




from tensorflow.keras import layers
import tensorflow as tf

generator = tf.keras.Sequential([
    layers.Input(shape=(100,)),

    layers.Dense(8 * 8 * 512, use_bias=False),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Reshape((8, 8, 512)),

    layers.Conv2DTranspose(256, 4, strides=2, padding='same', use_bias=False),  # 16x16
    layers.BatchNormalization(),
    layers.ReLU(),

    layers.Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False),  # 32x32
    layers.BatchNormalization(),
    layers.ReLU(),

    layers.Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False),   # 64x64
    layers.BatchNormalization(),
    layers.ReLU(),

    layers.Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh')  # 128x128
])





discriminator = tf.keras.Sequential([
    layers.Input(shape=(128, 128, 3)),

    layers.Conv2D(64, 4, strides=2, padding='same'),
    layers.LeakyReLU(0.2),

    layers.Conv2D(128, 4, strides=2, padding='same'),
    layers.LeakyReLU(0.2),

    layers.Conv2D(256, 4, strides=2, padding='same'),
    layers.LeakyReLU(0.2),

    layers.Conv2D(512, 4, strides=2, padding='same'),
    layers.LeakyReLU(0.2),

    layers.Flatten(),
    layers.Dense(1)
])



bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = bce(tf.ones_like(real_output) * 0.9, real_output)
    fake_loss = bce(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return bce(tf.ones_like(fake_output), fake_output)






gen_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
disc_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)






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

    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

    return gen_loss, disc_loss







import numpy as np
import matplotlib.pyplot as plt

EPOCHS = 100

for epoch in range(EPOCHS):

    for image_batch in dataset:
        g_loss, d_loss = train_step(image_batch)

    print(f"Epoch {epoch+1}/{EPOCHS} | G: {g_loss:.4f} | D: {d_loss:.4f}")

    # preview images
    noise = tf.random.normal([6, 100])
    generated_images = generator(noise, training=False)

    plt.figure(figsize=(10, 4))
    for i in range(6):
        plt.subplot(1, 6, i+1)
        img = (generated_images[i] + 1) / 2.0
        plt.imshow(img)
        plt.axis("off")
    plt.show()


    








































































from google.colab import drive
drive.mount('/content/drive')

import os, numpy as np, tensorflow as tf
import matplotlib.pyplot as plt

SAVE_DIR = '/content/drive/MyDrive/cat_gan_checkpoints'
os.makedirs(SAVE_DIR, exist_ok=True)





import kagglehub
path = kagglehub.dataset_download("shaunthesheep/microsoft-catsvsdogs-dataset")

train_ds = tf.keras.utils.image_dataset_from_directory(
    directory=f'{path}/PetImages/Cat',
    labels=None,
    image_size=(128, 128),
    batch_size=32,
    color_mode="rgb"
)

def preprocess(image):
    image = tf.cast(image, tf.float32)
    return (image / 127.5) - 1.0

dataset = train_ds.map(preprocess).shuffle(1000).ignore_errors().prefetch(tf.data.AUTOTUNE)






from tensorflow.keras import layers

def build_generator():
    return tf.keras.Sequential([
        layers.Input(shape=(100,)),
        layers.Dense(8 * 8 * 512, use_bias=False),
        layers.BatchNormalization(), layers.ReLU(),
        layers.Reshape((8, 8, 512)),
        layers.Conv2DTranspose(256, 4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(), layers.ReLU(),
        layers.Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(), layers.ReLU(),
        layers.Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(), layers.ReLU(),
        layers.Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh')
    ])

def build_discriminator():
    return tf.keras.Sequential([
        layers.Input(shape=(128, 128, 3)),
        layers.Conv2D(64, 4, strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        layers.Conv2D(128, 4, strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        layers.Conv2D(256, 4, strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        layers.Conv2D(512, 4, strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        layers.Flatten(),
        layers.Dense(1)
    ])

generator     = build_generator()
discriminator = build_discriminator()

gen_optimizer  = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
disc_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)










def save_checkpoint(epoch):
    generator.save_weights(f'{SAVE_DIR}/gen_weights.weights.h5')
    discriminator.save_weights(f'{SAVE_DIR}/disc_weights.weights.h5')
    np.save(f'{SAVE_DIR}/gen_opt.npy',  gen_optimizer.get_weights(),  allow_pickle=True)
    np.save(f'{SAVE_DIR}/disc_opt.npy', disc_optimizer.get_weights(), allow_pickle=True)
    np.save(f'{SAVE_DIR}/meta.npy', {'epoch': epoch}, allow_pickle=True)
    print(f"✅ Saved at epoch {epoch + 1}")

def load_checkpoint():
    generator.load_weights(f'{SAVE_DIR}/gen_weights.weights.h5')
    discriminator.load_weights(f'{SAVE_DIR}/disc_weights.weights.h5')

    # One dummy pass to initialize optimizer slots before loading state
    dummy_noise = tf.random.normal([1, 100])
    dummy_image = tf.random.normal([1, 128, 128, 3])
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    with tf.GradientTape() as gt, tf.GradientTape() as dt:
        fake = generator(dummy_noise, training=True)
        r_out = discriminator(dummy_image, training=True)
        f_out = discriminator(fake, training=True)
        g_loss = bce(tf.ones_like(f_out), f_out)
        d_loss = bce(tf.ones_like(r_out)*0.9, r_out) + bce(tf.zeros_like(f_out), f_out)

    gen_optimizer.apply_gradients(zip(
        gt.gradient(g_loss, generator.trainable_variables),
        generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(
        dt.gradient(d_loss, discriminator.trainable_variables),
        discriminator.trainable_variables))

    # Now restore the real optimizer state
    gen_optimizer.set_weights(
        np.load(f'{SAVE_DIR}/gen_opt.npy', allow_pickle=True))
    disc_optimizer.set_weights(
        np.load(f'{SAVE_DIR}/disc_opt.npy', allow_pickle=True))

    meta = np.load(f'{SAVE_DIR}/meta.npy', allow_pickle=True).item()
    print(f"✅ Resumed from epoch {meta['epoch'] + 1}")
    return meta['epoch'] + 1







bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def disc_loss(real_out, fake_out):
    return (bce(tf.ones_like(real_out) * 0.9, real_out) +
            bce(tf.zeros_like(fake_out) + 0.1,  fake_out))  # soft labels both sides

def gen_loss(fake_out):
    return bce(tf.ones_like(fake_out), fake_out)

@tf.function
def train_step(images, step):
    noise = tf.random.normal([tf.shape(images)[0], 100])

    with tf.GradientTape() as gt, tf.GradientTape() as dt:
        fake_imgs  = generator(noise, training=True)
        real_out   = discriminator(images,     training=True)
        fake_out   = discriminator(fake_imgs,  training=True)
        g_loss_val = gen_loss(fake_out)
        d_loss_val = disc_loss(real_out, fake_out)

    disc_optimizer.apply_gradients(zip(
        dt.gradient(d_loss_val, discriminator.trainable_variables),
        discriminator.trainable_variables))

    # Update generator every other step to keep balance
    if step % 2 == 0:
        gen_optimizer.apply_gradients(zip(
            gt.gradient(g_loss_val, generator.trainable_variables),
            generator.trainable_variables))

    return g_loss_val, d_loss_val










EPOCHS     = 300
SAVE_EVERY = 10

def save_checkpoint_fixed(epoch):
    generator.save_weights(f'{SAVE_DIR}/gen_weights.weights.h5')
    discriminator.save_weights(f'{SAVE_DIR}/disc_weights.weights.h5')

    # Fix: Revert to using optimizer.variables to get internal state if get_weights() is not available.
    # This requires the optimizer to have been 'built' by calling apply_gradients at least once.
    gen_opt_state_numpy = [v.numpy() for v in gen_optimizer.variables]
    disc_opt_state_numpy = [v.numpy() for v in disc_optimizer.variables]

    # Fix: Explicitly save as a NumPy array of dtype=object to handle heterogeneous shapes.
    np.save(f'{SAVE_DIR}/gen_opt.npy',  np.array(gen_opt_state_numpy, dtype=object),  allow_pickle=True)
    np.save(f'{SAVE_DIR}/disc_opt.npy', np.array(disc_opt_state_numpy, dtype=object), allow_pickle=True)
    np.save(f'{SAVE_DIR}/meta.npy', {'epoch': epoch}, allow_pickle=True)
    print(f"✅ Saved at epoch {epoch + 1}")

def load_checkpoint_fixed():
    try:
        generator.load_weights(f'{SAVE_DIR}/gen_weights.weights.h5')
        discriminator.load_weights(f'{SAVE_DIR}/disc_weights.weights.h5')

        # One dummy pass to initialize optimizer slots before loading state
        dummy_noise = tf.random.normal([1, 100])
        dummy_image = tf.random.normal([1, 128, 128, 3])
        bce_temp = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        with tf.GradientTape() as gt, tf.GradientTape() as dt:
            fake = generator(dummy_noise, training=True)
            r_out = discriminator(dummy_image, training=True)
            f_out = discriminator(fake, training=True)
            g_loss_temp = bce_temp(tf.ones_like(f_out), f_out)
            d_loss_temp = bce_temp(tf.ones_like(r_out)*0.9, r_out) + bce_temp(tf.zeros_like(f_out), f_out)

        gen_optimizer.apply_gradients(zip(
            gt.gradient(g_loss_temp, generator.trainable_variables),
            generator.trainable_variables))
        disc_optimizer.apply_gradients(zip(
            dt.gradient(d_loss_temp, discriminator.trainable_variables),
            discriminator.trainable_variables))

        # Fix: Load the NumPy array of objects and use .item() to extract the Python list.
        loaded_gen_opt_state = np.load(f'{SAVE_DIR}/gen_opt.npy', allow_pickle=True)
        loaded_disc_opt_state = np.load(f'{SAVE_DIR}/disc_opt.npy', allow_pickle=True)

        gen_optimizer.set_weights(loaded_gen_opt_state.item())
        disc_optimizer.set_weights(loaded_disc_opt_state.item())

        meta = np.load(f'{SAVE_DIR}/meta.npy', allow_pickle=True).item()
        print(f"✅ Resumed from epoch {meta['epoch'] + 1}")
        return meta['epoch'] + 1
    except (EOFError, FileNotFoundError, ValueError) as e:
        print(f"⚠️ Error loading checkpoint: {e}. Starting fresh.")
        # If any essential file is corrupt or missing, start fresh.
        return 0

# ── auto-detect checkpoint ──────────────────────────────────────
# Check for all necessary checkpoint files
checkpoint_files_exist = all([
    os.path.exists(f'{SAVE_DIR}/gen_weights.weights.h5'),
    os.path.exists(f'{SAVE_DIR}/disc_weights.weights.h5'),
    os.path.exists(f'{SAVE_DIR}/gen_opt.npy'),
    os.path.exists(f'{SAVE_DIR}/disc_opt.npy'),
    os.path.exists(f'{SAVE_DIR}/meta.npy')
])

if checkpoint_files_exist:
    start_epoch = load_checkpoint_fixed()
else:
    start_epoch = 0
    print("🚀 Starting fresh")

# ── loop ────────────────────────────────────────────────────────
for epoch in range(start_epoch, EPOCHS):
    for i, batch in enumerate(dataset):
      try :
        g_loss, d_loss = train_step(batch, tf.constant(i))
      except tf.errors.InvalidArgumentError:
        continue

    print(f"Epoch {epoch+1}/{EPOCHS} | G: {g_loss:.4f} | D: {d_loss:.4f}")

    if (epoch + 1) % SAVE_EVERY == 0:
        save_checkpoint_fixed(epoch)

    # Preview 6 images every 10 epochs
    if (epoch + 1) % 10 == 0:
        noise = tf.random.normal([6, 100])
        imgs  = generator(noise, training=False)
        plt.figure(figsize=(12, 3))
        for i in range(6):
            plt.subplot(1, 6, i+1)
            plt.imshow((imgs[i] + 1) / 2.0)
            plt.axis('off')
        plt.suptitle(f'Epoch {epoch+1}')
        plt.show()










