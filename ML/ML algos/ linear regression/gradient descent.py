import pandas as pd
import  matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

df = pd.read_csv('E:\intern preparation\ML\ linear regression\Housing.csv')
x = df.area
y = df.price
        # normalizing values 
x_scaled = (x - x.mean()) / x.std()
y_scaled = (y - y.mean()) / y.std()


def gradient_descent(x, y):
    m_current, b_current = 0, 0
    iterations = 10000
    lr = 0.005    # learning_rate
    n = len(x)

    for i in range(iterations):
        if i != (iterations-1):
            y_predicted = m_current * x + b_current

            cost_func = (1/n)* sum((y-y_predicted)**2)

            m_d = -(2/n) * sum(x * (y - y_predicted))            # slope_derivative
            b_d = -(2/n) * sum(y - y_predicted)                  # intercept_derivative

            m_current = m_current - lr * m_d
            b_current = b_current - lr * b_d
        else:
            print('Cost {} , slope_derivative {}, intercept_derivative {}'.format(cost_func, m_d, b_d))

gradient_descent(x_scaled, y_scaled)






model = LinearRegression()
model.fit(x_scaled.values.reshape(-1, 1), y_scaled)
prediction = model.predict(x_scaled.values.reshape(-1, 1))

print('Coefficient of the model is : ', model.coef_)
print('intercept of the model is : ', model.intercept_)

sns.scatterplot(x = x_scaled, y=y_scaled)           # original data plotting
sns.scatterplot(x=x_scaled, y=prediction, color='y', markers='+')      # original VS prediction
sns.lineplot(x=x_scaled, y=prediction)
plt.savefig('gradient_descent.png', dpi=300, bbox_inches='tight')
plt.show()