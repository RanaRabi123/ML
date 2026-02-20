import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

df = datasets.load_digits()

x = df.data
y = df.target


        # kfold is nothing but just a split method like --> train_test_split


        # it split the data into k groups
# kf = KFold(n_splits=4)

# for train_index, test_index in kf.split(df.data):                     # to see the indexes
#   print(len(train_index), len(test_index))

         #  it splits the data into k groups while ensuring class balance  
# skf = StratifiedKFold(n_splits=6)

# for train_index, test_index in skf.split(df.data, df.target):         # to see indexes
#   print(len(train_index), len(test_index))



            # cv is a splititng parametr, means data will be split 4 times 
rfc_score = cross_val_score(RandomForestClassifier(), df.data, df.target, cv=4)
print("The score of the R_F_C using k fold is : ", rfc_score)
print("The mean of the model score is : ", rfc_score.mean())
print('\t \t', '--'*22)


            # didgits dataset is multiclass and heavy , so use max_iteration otherwise they can show warnigns or underperform 
lr_score = cross_val_score(LogisticRegression(max_iter=1000), df.data, df.target, cv=4)
print("The score of the Logistic regresssion using k fold is :", lr_score)
print("the meanof the model score is : ", lr_score.mean())
print('\t \t', '--'*22)


svc_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc' , SVC())
]) 
svc_score = cross_val_score(svc_pipeline, df.data, df.target, cv=4)
print("The score of the SVC using k fold is : ", svc_score)
print("The mean of the model score is : ", svc_score.mean())
print('\t \t', '--'*22)






            # plotting

folds = np.arange(1, 5)
plt.plot(folds, rfc_score, marker='o', label='RFC score')
plt.plot(folds, lr_score, marker='o', label='LR score')
plt.plot(folds, svc_score, marker='o', label='SVC score')
plt.xlabel("Fold Number")
plt.ylabel("Accuracy")
plt.title("K-Fold Cross Validation Accuracy (cv=4)")
plt.grid()
plt.legend(loc='lower right')
plt.show()