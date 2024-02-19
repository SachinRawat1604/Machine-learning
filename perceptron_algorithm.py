import numpy as np
#from numpy.random import seed
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

#sanity check
df.head()

# Extract the first 100 class labels corresponding to the 50 Iris-setosa and 50 Iris-versicolor flowers
y = df.iloc[0:100, 4].values

negative_class = -1
positive_class = 1
y = np.where(y == 'Iris-setosa', positive_class, negative_class)  # Change 'Iris-setosa' to 1 and 'Iris-versicolor' to -1
X = df.iloc[0:100, [0, 2]].values

# Let's visualize our feature matrix 'x' using a 2D scatter plot
# Setosa plot
sepal_length_setosa = X[0:50, 0]  # Get sepal length for setosa flowers only
petal_length_setosa = X[0:50, 1]   # Get petal length for setosas as well
color = "Red"
plt.scatter(sepal_length_setosa, petal_length_setosa, color=color, marker='o', label='setosa')

# Versicolor plot
sepal_length_versicolor = X[50:100, 0]   # Get sepal length for versicolors
petal_length_versicolor = X[50:100, 1]    # Get petal length for versicolors
color = "Blue"
plt.scatter(sepal_length_versicolor, petal_length_versicolor, color=color, marker='x', label='versicolor')

# Label the axes
plt.xlabel("Sepal Length [cm]")
plt.ylabel("Petal Length [cm]")

# Add the legend
plt.legend(loc='upper left')

# Instantiate a perceptron object
eta = 0.1 #learning rate
n_iter = 10 # number of epochs (passes) over the training data.
ppn = perceptron(eta=eta, n_iter=n_iter)

# Fit the perceptron instance to our training data
ppn.fit(X, y)

# Let's plot our misclassification error
epochs = range(1, len(ppn.errors_) + 1)
misclassifications = ppn.errors_
marker = "o"
plt.plot(epochs, misclassifications, marker)

# Add labels
plt.xlabel("Epochs")
plt.ylabel("Number of misclassifications")

# Convenience function for visualizing decision boundaries
def plot_decision_regions(X, y, classifier, resolution=0.02):
    """Convenience function for visualizing decision boundaries"""
    
    # setup marker generator and color map
    # This can handle multi-class plotting
    markers = ("s", "x", "o", "^", "v")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot hte decision surface.
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.figure()
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class sample.
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

# Let's try it out
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel("Sepal Length [cm]")
plt.ylabel("Petal Length [cm]")
plt.legend(loc="upper left")

class AdalineGD(object):
    """Adaptive linear neuron classifier.
    parameters
    -----------
    eta: float
        Learning rate (between 0.0 and 1.0)
    n_iter: int
            passes(epochs) over the training set
    
    Attributes
    -----------
    w_ : id-array
        weight after fitting.
    errors : list
            Number of misclassification in every epoch.
            """
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter
        
    def fit(self, X, y):
        """ fit training data.
        parameter
        ------------
        X : {array-like}, shape={n_sample,n_features} training vectors,
            where n_samples is the number of samples and n_feature is the
            number of feature.
            
        y : {array-like}, shape={n_sample} Target value.
        
        Returns
        -----------
        self : objects. """
        
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        
        # use batch gradient descent instead of increment updatong like perceptron.
        # incremental updating like perceptron.
        for i in range (self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta*X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum()/2.0
            self.cost_.append(cost)
        return self
    
    def net_input(self, X):
        #calculate net input.
        return np.dot(X,self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        #compute linear activation
        return self.net_input(X)
    
    def predict(self, X):
        #return class label after unit step
        return np.where(self.activation(X) >= 0.0, 1, -1)
    
    
# CHOOSE A LEARNING RATE :--->
#Choose a learning rate can take experimentation.
#let's plot the difference b/w two learning rates.

nrows = 1
ncols = 2
fig, ax = plt.subplots(nrows = nrows,
                       ncols = ncols)

#Use the first learning rate to fit the training data.
n_iter = 10
eta = 0.01
ada1 = AdalineGD(n_iter = n_iter, eta = eta)

#fit the model.
ada1.fit(X,y)
epochs = range (1,len(ada1.cost_) + 1)
misclassifications = np.log(ada1.cost_)
marker = "o"

#Plot the first learning rate.
ax[0].plot(epochs, misclassifications, marker= marker)

#Set the label and title
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("log (sum_squared-error)")
ax[0].set_label("Adaline-learning rate 0.01")

#use the second learning rate to fit the training data.
n_iter = 10
eta = 0.0001
ada2 = AdalineGD(n_iter = n_iter,
                 eta = eta)

# Fit the model
ada2.fit(X, y)
epochs = range(1, len(ada2.cost_) + 1)
misclassifications = ada2.cost_
marker = "o"

# Plot the second learning rate.
ax[1].plot(epochs, misclassifications, marker= marker)
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("log (sum_squared-error)")
ax[1].set_label("Adaline-learning rate 0.01")

plt.show()