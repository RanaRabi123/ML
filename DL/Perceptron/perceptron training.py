from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt

x, y = make_classification(n_samples=100, n_features=2, n_informative=1,n_redundant=0,
                           n_classes=2, n_clusters_per_class=1, random_state=41,hypercube=False,class_sep=15)

plt.scatter(x[:,0], x[:,1],c=y,cmap='winter',s=33)
plt.show()

def perceptron(x,y):
    x = np.insert(x, 0, 1, axis=1)  # (array, index, value)
    weights = np.ones(x.shape[1])
    lr = 0.01
    for _ in range(1000):
        j = np.random.randint(0, 100)
        y_hat = step(np.dot(x[j], weights))
        weights = weights + lr*(y[j]-y_hat)*x[j]

    return weights[1:], weights[0]

def step(z):
    return 1 if z>0 else 0

coef, intercept = perceptron(x,y)



        # making line equation
m = -(coef[0]/coef[1])          # -a/b
c = -(intercept/coef[1])        # -c/b

x_input = np.linspace(-3,3,100)
y_input = m*x_input + c

plt.plot(x_input,y_input,color='red',linewidth=3)
plt.scatter(x[:,0], x[:,1],c=y,cmap='winter',s=22)
plt.ylim(-3,2)