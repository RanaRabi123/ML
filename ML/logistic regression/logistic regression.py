import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

data = datasets.load_iris()
print(dir(data))    # it will show info of the dataset 

# print(data.data[34])      # random data point
# print(data.target[34])    # it shoes above data's category

x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=55)

model = LogisticRegression()
model.fit(x_train, y_train)
y_prediction = model.predict(x_test)    # it will just predict flower type's number

y_prediciton_names =  data.target_names[y_prediction]   # it will show the coorresponding name

cm = confusion_matrix(y_test, y_prediction)
print("Confusion matrix : \n" , cm)

        #plotting
sns.heatmap(cm, annot=True)
# plt.savefig('iris_heatmap.jpg', dpi=300, bbox_inches='tight')
plt.show()