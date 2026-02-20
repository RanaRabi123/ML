import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.18, random_state=33)
model = RandomForestClassifier()
model.fit(x_train, y_train)
y_prediction = model.predict(x_test)

model_score = model.score(x_test, y_test)
print('The score of the RandomForestClassifier model is : \n', model_score)

cm = confusion_matrix(y_test, y_prediction)
print('Confusion Matrix : \n', cm)

sns.heatmap(cm, annot=True, cmap='YlOrBr')
plt.show()

















        # this part is just for visualization to see the plotting 



                    # dimensionality reduction
pcaa = PCA(n_components=2)
x_reduce = pcaa.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_reduce, y, test_size=0.18, random_state=33)
model = RandomForestClassifier()
model.fit(x_train, y_train)

min_x, max_x = x_train[:, 0].min()-1, x_train[:, 0].max()+1
min_y, max_y = x_train[:, 1].min()-1, x_train[:, 1].max()+1

        # use 0.05 step fro smooth boundary 
xx, yy = np.meshgrid(np.arange(min_x, max_x, 0.05), np.arange(min_y, max_y, 0.05))

z = model.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)

plt.contourf(xx, yy, z, alpha=0.5)
sns.scatterplot(x=x_train[:, 0], y=x_train[:, 1],  hue=y_train, palette='viridis', legend=True, edgecolor='k')
plt.savefig('Randomforestclassify.jpg', dpi=300, bbox_inches='tight')
plt.colorbar()
plt.show()