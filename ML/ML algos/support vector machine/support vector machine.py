import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

df = datasets.load_digits()
print(dir(df))

x = df.data
y = df.target

    # Principal component analysis , used for dimensionality reduction
pcaa = PCA(n_components=2)
x_reduce = pcaa.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_reduce, y, test_size=0.18, random_state=33)

model = SVC()
model.fit(x_train, y_train)
y_prediction = model.predict(x_test)

model_score  = model.score(x_test, y_test)
print("The score of the model is : \n", model_score)

cm = confusion_matrix(y_test, y_prediction)
print('Confusion matrix : \n:', cm)

sns.heatmap(cm, annot=True)     # here annot short for annotation s, which means show numbers 
plt.show()






            #decision boundary plotting
min_x, max_x = x_train[:, 0].min()-1, x_train[:, 0].max()+1
min_y, max_y = x_train[:, 1].min()-1, x_train[:, 1].max()+1

xx, yy = np.meshgrid(np.arange(min_x, max_x, 1), np.arange(min_y, max_y, 1))

    # used to convert to 1_dimension , same as flatten works 
z = model.predict(np.c_[xx.ravel(), yy.ravel()])    # c_ short for concatenation in numpy
z = z.reshape(xx.shape)

plt.contourf(xx, yy, z, alpha=0.5)      # for colorful decision boundries 
sns.scatterplot(x=x_train[:, 0], y=x_train[:, 1], hue=y_train, palette='viridis', legend=True, edgecolor='k')   #for data points
plt.colorbar()
plt.show()