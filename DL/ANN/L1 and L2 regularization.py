from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from mlxtend.plotting import plot_decision_regions
from tensorflow.keras import regularizers
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import seaborn as sns

X, y = make_moons(n_samples=600, noise=0.02)

sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y)
plt.show()


            # without regularization to see overfitting
model1 = Sequential()
model1.add(Dense(16, input_shape=(2,) , activation='relu'))
model1.add(Dense(8, activation='relu' ))
model1.add(Dense(1, activation='sigmoid'))

model1.compile(loss='binary_crossentropy' , optimizer='adam', metrics=['accuracy'])
history1= model1.fit(X, y, epochs=188, verbose=1, validation_split=0.2)

plot_decision_regions(X, y, clf=model1, legend=2)
plt.show()











            # with L2
model2 = Sequential()
model2.add(Dense(16, input_shape=(2,) , activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model2.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model2.add(Dense(1, activation='sigmoid'))

model2.compile(loss='binary_crossentropy' , optimizer='adam', metrics=['accuracy'])
history2= model2.fit(X, y, epochs=188, verbose=1, validation_split=0.2)

plot_decision_regions(X, y, clf=model2, legend=2)
plt.show()











                #  with L1 
model3 = Sequential()
model3.add(Dense(16, input_shape=(2,) , activation='relu', kernel_regularizer=regularizers.l1(0.01)))
model3.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l1(0.01)))
model3.add(Dense(1, activation='sigmoid'))

model3.compile(loss='binary_crossentropy' , optimizer='adam', metrics=['accuracy'])
history3= model3.fit(X, y, epochs=188, verbose=1, validation_split=0.2)

plot_decision_regions(X, y, clf=model3, legend=2)
plt.show()










            #  getting weights of layer 1 of the neural network of all models 
m1_w = model1.get_weights()[0].reshape(32)
m2_w = model2.get_weights()[0].reshape(32)
m3_w = model3.get_weights()[0].reshape(32)


        # it will show how weights are affected by the regularizers 
        # weights in L1 will be close to 0, or may be zero  
        # weights in L2 is rarely close to zero , but not be zero 
plt.title('weights of layer 1 of neural network')
plt.boxplot([m1_w, m2_w, m3_w], labels=['Without regularize', 'with L2', 'with L1'])
plt.show()



sns.distplot(m1_w, color='r', label='without regularize')
sns.distplot(m2_w, color='y', label='L2 regularize')
sns.distplot(m3_w, color='b', label='L1 regularize')
plt.legend()
plt.show()





plt.title('validation Loss of models')
sns.lineplot(x=history1.epoch, y=history1.history['val_loss'], label="without regularize")
sns.lineplot(x=history2.epoch, y=history2.history['val_loss'], label="with L2")
sns.lineplot(x=history3.epoch, y=history3.history['val_loss'], label='with L1')
plt.show()

