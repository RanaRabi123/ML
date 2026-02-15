import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

df = datasets.load_iris()

df_1 = pd.DataFrame(data=df.data, columns=df.feature_names)
df_1['target'] = df.target

x = df_1.drop('target', axis=1)
y = df_1.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=33)

        # finding best k value for the model 
score = []
for i in range(1, 20):
  c_v_score = cross_val_score(KNeighborsClassifier(n_neighbors=i), x, y, cv=10)
  score.append((i, c_v_score.mean()))


best_k, best_score = max(score, key=lambda x: x[1])
print(f"Best K: {best_k},\nScore: {best_score}\n")

            #  model 
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(x_train, y_train)
y_prediction = knn.predict(x_test)


print('Classification report of the model is : \n', classification_report(y_test, y_prediction))

cm = confusion_matrix(y_test, y_prediction)
print('Confusion matrix : \n', cm)
plt.figure(figsize=(10, 6))
sns.heatmap(cm , annot=True)
plt.show()





        # as score is in list format
k_values = [x[0] for x in score]
accuracy_values = [x[1] for x in score]

max_score = max(accuracy_values)
clrs = ['red' if j == max_score  else 'grey' for j in accuracy_values]
plt.figure(figsize=(10, 6))
sns.lineplot(x=k_values, y=accuracy_values, color='grey')
sns.scatterplot(x=k_values, y=accuracy_values, color=clrs, marker='o', s=123, zorder=5)     # zorder means it will draw on front/top
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.title(f'KNN Accuracy with k={best_k}')
plt.grid()
plt.show()








































            # plotting , just to visualize the k-nearest neigbor model 



# 1. Train a new KNN model using ONLY 2 features (e.g., Sepal Length & Width) for 2D plotting
feature_x = 'sepal length (cm)'
feature_y = 'sepal width (cm)'

X_2d = df_1[[feature_x, feature_y]]
y_2d = df_1.target

# Fit on 2D data
knn_2d = KNeighborsClassifier(n_neighbors=best_k).fit(X_2d, y_2d)

# 2. Create a random point to test
# Random values between min and max of the features
random_point = np.array([
    [np.random.uniform(X_2d[feature_x].min(), X_2d[feature_x].max()), np.random.uniform(X_2d[feature_y].min(), X_2d[feature_y].max())]
    ])

# Predict the class for the random point
mapping  = {0:'Setosa', 1:'Versicolor', 2:'Virginica'}
prediction = knn_2d.predict(random_point)
print(f"Random Point:    {random_point[0]}")
print(f"Predicted Class is : {mapping[prediction[0]]}\n")

# 3. Plot the Decision Boundary and Data
plt.figure(figsize=(10, 6))

# Create a mesh grid to plot decision boundary
x_min, x_max = X_2d[feature_x].min() - 1, X_2d[feature_x].max() + 1
y_min, y_max = X_2d[feature_y].min() - 1, X_2d[feature_y].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# Predict class for each point in the mesh grid
Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot background (Decision Boundary)
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

# Plot original training data
sns.scatterplot(x=X_2d[feature_x], y=X_2d[feature_y], hue=y_2d, palette='deep', edgecolor='k')

# 4. Plot the Random Point and its Neighbors
plt.scatter(random_point[:, 0], random_point[:, 1], color='black', s=150, label='Random Point', marker='X', zorder=5)

# Find and draw lines to the Nearest Neighbors
distances, indices = knn_2d.kneighbors(random_point)

for i in range(best_k):
    neighbor_idx = indices[0][i]
    # Get coordinates of the neighbor
    nx = X_2d.iloc[neighbor_idx][feature_x]
    ny = X_2d.iloc[neighbor_idx][feature_y]
    
    # Draw a line from random point to neighbor
    plt.plot([random_point[0, 0], nx], [random_point[0, 1], ny], 'k--', alpha=0.6)

plt.title(f'KNN (k={best_k}) - Random Point Prediction')
plt.legend()
plt.show()