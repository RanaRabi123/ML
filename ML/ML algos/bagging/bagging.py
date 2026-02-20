from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from sklearn.ensemble import BaggingClassifier

df = datasets.load_digits()

x = df.data
y = df.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=33)
model = RandomForestClassifier()
model.fit(x_train, y_train)
y_prediction = model.predict(x_test)

print('The score of the RandomForestClassifier model is : ', model.score(x_test, y_test))



            #  bagging classification mdoel 
bag_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    oob_score=True      
)
bag_model.fit(x_train, y_train)
bag_model_y_prediction = bag_model.predict(x_test)


        #  oob score means , surprise test data
print("The OOB Score of the bagging model with DTC is : ", bag_model.oob_score_)
print('Score of bagging model is : ', bag_model.score(x_test, y_test))

print("CLassification report of the bagging model is : \n", classification_report(y_test, bag_model_y_prediction))