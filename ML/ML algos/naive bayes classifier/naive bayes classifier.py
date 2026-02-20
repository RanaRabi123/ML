import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r'titanic.csv')
print(df.head())

df_1 = df.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis=1)

        # to encode the textual category column 
sex_en = OrdinalEncoder()
df_1['Sex'] = sex_en.fit_transform(df_1[['Sex']])
# print("The feature has the category : ", sex_en.categories_)   # to see the categories of the encoded column

            # to remove high age outlier 
q1 = df_1.Fare.quantile(0)
q3 = df_1.Fare.quantile(0.99)
df_1 = df_1[(df_1.Fare>= q1) & (df_1.Fare<=q3)]

        #  filling missing values 
df_1.Age.fillna(df_1.Age.mean(), inplace=True)
df_1.Age = df_1.Age.astype(int)



x = df_1.drop('Survived', axis=1)
y = df_1['Survived']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(x_train, y_train)
y_prediction = model.predict(x_test)

accuracy = accuracy_score(y_test, y_prediction)
print("The accuarcy of the model is : ", accuracy)

cm = confusion_matrix(y_test, y_prediction)
print("Confusion matrix : \n", cm)

sns.heatmap(cm, annot=True, fmt='d')        #  visualizing confusion matrix
plt.savefig('decision_tree_cm.jpg', dpi=300, bbox_inches='tight')       # svaing the heatmap
plt.show()


            # plotting of survived and not_survived
sns.catplot(x=df_1.Survived.map({0:'not survived', 1:'survived'}), y=df_1.Age, 
            hue='Pclass',      # Color bars by class
            col= df_1.Sex.map({0:'female', 1:'male'}),         # Separate plots for Male and Female
            kind='strip', data=df_1)
plt.savefig('decision_tree_plot.jpg', dpi=300, bbox_inches='tight')       # saving the plot 
plt.show()









            # this part is just for plotting 
# step1 is just data processing
# step2 is getting temporary valus to make the grid
# step3 is trainging is just for this plotting part 
# step4 will predict for the temporary points that were made during step2 in meshgird
# step5 will plot the predictions with background colorful_boxes  


# 1. Select only 2 features for plotting 
# We use the full x data just to get the range, but usually we visualize the train set
X_plot = x_train[['Pclass', 'Fare']].values
y_plot = y_train.values


# 2. Create the mesh grid (background)
# 1 is for padding so that boundary don't overlap
x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 1), np.arange(y_min, y_max, 1))


    #  it will give the best 2d representation,  reduce the dimension from 4d to 2d 
pca = PCA(n_components=2)
X_pca = pca.fit_transform(x_train)


# 3. Train a temporary model using ONLY these 2 features
model_2d = GaussianNB()
model_2d.fit(X_pca, y_plot)

# 4. Predict for every point in the mesh
Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


# 5. Plot the boundary
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)          # colorful backgorund obtained during prediction result
plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y_plot, edgecolors='k', cmap=plt.cm.Paired)   # actual data points
plt.xlabel('Pclass')
plt.ylabel('Fare')
plt.title('Decision Tree Boundary (Pclass vs Fare)')
plt.legend()
plt.savefig('k-means plot.jpg', dpi=300, bbox_inches='tight')
plt.show()