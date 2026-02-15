import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('E:\intern preparation\ML\L1 & L2 regularization\Melbourne_housing_FULL.csv')

        # dropping irrelevant columns 
df.drop(['Bedroom2', 'Address', 'Type', 'Method', 'SellerG', 'Date', 'Distance', 'Postcode', 'Car', 'CouncilArea', 'Regionname', 'Propertycount'], axis=1, inplace=True)
df.dropna(subset=['Lattitude', 'Longtitude'] , axis=0, inplace=True)    # dropping empty rows 

print(df.isna().sum())  # show empty data info

        # filling up empty data
for i in df.columns:
  if df[i].isna().sum() > 0:
    if df[i].dtypes == 'object' :       # if data is str type/category
      df[i] = df[i].fillna('missing')
    else:                               # for numeric based data
      median_value = df[i].median()                                 
      df[i] = df[i].fillna(median_value).astype(int)

        # creatign dummy columns and dropping them 
for i in df.columns:
  if df[i].dtypes == 'object':
    dummy_column = pd.get_dummies(df[i], drop_first=True, prefix=i)  #here i means , add column_name before making new dummy name column
    df = pd.concat([df, dummy_column], axis=1)
    df.drop([i], inplace=True, axis=1)


x = df.drop(['Price'], axis=1)
y = df['Price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.23, random_state=22)

        #  Linear regression model
linear_r = LinearRegression().fit(x_train, y_train)
linear_r_predict = linear_r.predict(x_test)
linear_r_score = linear_r.score(x_test, y_test)
print('The score of the Linear Regression model is : ', linear_r_score)



                # L1 regularization
                # lasso regression
lasso_r = Lasso(alpha=50, tol=0.1, max_iter=100).fit(x_train, y_train)
lasso_r_predict = lasso_r.predict(x_test)
lasso_r_score = lasso_r.score(x_test, y_test)
print('The score of Lasso regression is : ', lasso_r_score)



                #  L2 regularization
                #  Ridge regression
ridge_r = Ridge(alpha=50, tol=0.1, max_iter=100).fit(x_train, y_train)
ridge_r_predict = ridge_r.predict(x_test)
ridge_r_score = ridge_r.score(x_test, y_test)
print('The score of the Ridge Resgression is : ', ridge_r_score)


print('MAE of Linear rge is : ', mean_absolute_error(y_test, linear_r_predict), '\n')
print('MAE of Lasso reg is : ', mean_absolute_error(y_test, lasso_r_predict), '\n')
print('MAE of Ridge reg is : ', mean_absolute_error(y_test, ridge_r_predict))


                    # plotting
clr = ['red' if x > linear_r_score else 'grey' for x in [linear_r_score, lasso_r_score, ridge_r_score]]
sns.barplot(x=['Linear Regression', 'Lasso Regression', 'Ridge Regression'], y=[linear_r_score, lasso_r_score, ridge_r_score], width=0.5, palette=clr)
plt.show()



# models = ['Linear', 'Lasso (L1)', 'Ridge (L2)']
# scores = [linear_r_score, lasso_r_score, ridge_r_score]

# plt.bar(models, scores)
# plt.ylabel("R2 Score")
# plt.title("Model Comparison")
# plt.show()
