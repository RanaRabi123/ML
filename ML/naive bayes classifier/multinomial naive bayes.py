from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd


df = datasets.load_wine()
print('This dataset has the following data : \n',  dir(df))

x = pd.DataFrame(data=df.data, columns=df.feature_names)
y = df.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=33)


        # it will learn scaling ruls only from  training data, adn then apply it on both train and test data
preprocessor = ColumnTransformer(
    transformers=[('scale', MinMaxScaler(feature_range=(1, 33)), ['magnesium', 'proline'])],
                    remainder='passthrough' # Keeps other columns as they are
)

gnb = Pipeline([
    ('preprocessing', preprocessor),
    ('model', GaussianNB())
])


gnb.fit(x_train, y_train)
y_prediction = gnb.predict(x_test)

gnb_score = gnb.score(x_test, y_test)
print('Score of the GNB model is : ', gnb_score)


                #  checking a random sample from the datatset 
print('Actual target name is : ', df.target_names[y_test[11]])
print('predicted target name is : ', df.target_names[y_prediction[11]])

print('Confusion matrix is : \n', confusion_matrix(y_test, y_prediction))
print('The classification report is \n: ' , classification_report(y_test, y_prediction))