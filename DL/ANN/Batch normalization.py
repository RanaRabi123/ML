from tensorflow.keras.layers import Dense, BatchNormalization, Activation
from tensorflow.keras.models import Sequential
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt
import seaborn as sns

X, y = make_swiss_roll(n_samples=900, noise=0.08)

sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y)
plt.show()


            # without batch_normalization 
model1 = Sequential()
model1.add(Dense(8, input_shape=(3,) , activation='relu'))
model1.add(Dense(4, activation='relu' ))
model1.add(Dense(1, activation='linear'))

model1.compile(loss='mse' , optimizer='adam', metrics=['mse'])
history1= model1.fit(X, y, epochs=188, verbose=1, validation_split=0.2)










            # with Bacth normalization
model2 = Sequential()
model2.add(Dense(8, input_shape=(3,)))
model2.add(BatchNormalization())
model2.add(Activation('relu'))

model2.add(Dense(4))
model2.add(BatchNormalization())
model2.add(Activation('relu'))

model2.add(Dense(1, activation='linear'))

model2.compile(loss='mse' , optimizer='adam', metrics=['mse'])
    # here bydefault batch_size will be used , which is 32 , we will not specify it , no need to specify it manually
    # as it has affects on normalization 
history2= model2.fit(X, y, epochs=188, verbose=1, validation_split=0.2)










# get output of first Dense layer(8 neurons) (before BN)
dense1_out = model2.layers[0](X).numpy()   # Z before BN

# get output of first BatchNorm layer (after normalization, before activation)
bn1_out = model2.layers[1](dense1_out, training=True).numpy()  # training=True ensures BN uses batch stats

# get output after activation
act1_out = model2.layers[2](bn1_out).numpy()

print("Dense output (before BN):\n", dense1_out[:3])   # only first 3 samples
print("\nBatchNorm output (after BN, before ReLU):\n", bn1_out[:3])
print("\nAfter activation:\n", act1_out[:3])




sns.histplot(dense1_out.flatten(), color='r', label='Dense output')
sns.histplot(bn1_out.flatten(), color='y', label='BatchNorm output')
sns.histplot(act1_out.flatten(), color='b', label='After ReLU')
plt.legend()
plt.show()





plt.title('validation MSE of models')
sns.lineplot(x=history1.epoch, y=history1.history['val_mse'], label="without notmalization")
sns.lineplot(x=history2.epoch, y=history2.history['val_mse'], label="with Batch notmalization")
plt.show()




