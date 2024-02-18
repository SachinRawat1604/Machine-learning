import numpy as np
from numpy.random import seed
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import rcParams

# Set the plot figure size
rcParams['figure.figsize'] = (10, 5)

class perceptron(object):
    """ 
    perceptron classifier.

    Parameters
    --------
    eta : float
          learning rate for the algorithm (Between 0.0 and 1.0).
    
    n_iter : int
         passes (epochs) over the training set.

    Attributes
    ----------
    W_ : id-array
        weight after fitting.
    errors_ : list
           Number of misclassification in every epoch
    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        """Fit the model on data.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Training vector, where 'n_sample' is the number of sample and
            'n_feature' is the number of features.
        
        y : array-like, shape (n_samples,)
            Traget values.
        
        Returns
        ------
        self : object
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        """Calculate net input."""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        """Predict class label after unit step."""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

# Load the iris dataset
csv_name = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
header = None

# Load the data into a pandas dataframe.
df = pd.read_csv(csv_name, header=header)

# Extract the first 100 class labels corresponding to the 50 Iris-setosa and 50 Iris-versicolor flowers
y = df.iloc[0:100, 4].values
negative_class = -1
positive_class = 1
y = np.where(y == 'Iris-setosa', positive_class, negative_class)  # Change 'Iris-setosa' to 1 and 'Iris-versicolor' to -1
x = df.iloc[0:100, [0, 2]].values

# Let's visualize our feature matrix 'x' using a 2D scatter plot
# Setosa plot
sepal_length_setosa = x[0:50, 0]  # Get sepal length for setosa flowers only
petal_length_setosa = x[0:50, 1]   # Get petal length for setosas as well
plt.scatter(sepal_length_setosa, petal_length_setosa, marker='o', label='setosa')

# Versicolor plot
sepal_length_versicolor = x[50:100, 0]   # Get sepal length for versicolors
petal_length_versicolor = x[50:100, 1]    # Get petal length for versicolors
plt.scatter(sepal_length_versicolor, petal_length_versicolor, marker='x', label='versicolor')

# Label the axes
plt.xlabel("Sepal Length [cm]")
plt.ylabel("Petal Length [cm]")

# Add the legend
plt.legend(loc='upper left')

# Instantiate a perceptron object
eta = 0.1
ppn = perceptron(eta=eta, n_iter=10)

# Fit the perceptron instance to our training data
ppn.fit(x, y)

# Let's plot our misclassification error
epochs = range(1, len(ppn.errors_) + 1)
plt.figure()
plt.plot(epochs, ppn.errors_, marker='o')
plt.xlabel("Epochs")
plt.ylabel("Number of misclassifications")

# Convenience function for visualizing decision boundaries
def plot_decision_regions(X, y, classifier, resolution=0.02):
    """Convenience function for visualizing decision boundaries"""
    markers = ("s", "x", "o", "^", "v")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

# Let's try it out
plot_decision_regions(x, y, classifier=ppn)
plt.xlabel("Sepal Length [cm]")
plt.ylabel("Petal Length [cm]")
plt.legend(loc="upper left")
plt.show()
