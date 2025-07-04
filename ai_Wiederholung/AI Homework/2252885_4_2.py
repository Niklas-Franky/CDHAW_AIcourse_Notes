import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import learning_curve
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X) + 0.1 * np.random.randn(100, 1)
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Mean squared error")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = -np.mean(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    plt.grid()

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training error")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation error")

    plt.legend(loc="best")
    return plt
pipeline = make_pipeline(PolynomialFeatures(degree=10), Ridge(alpha=1))
plot_learning_curve(pipeline, 'Polynomial Regression with Ridge', X, y, cv=10)
plt.show()
X_new = np.linspace(0, 10, 1000).reshape(-1, 1)
y_new_pred = pipeline.predict(X_new)
plt.figure()
plt.plot(X, y, 'b.', label='Data')
plt.plot(X_new, y_new_pred, 'r-', label='Ridge Polynomial Regression')
plt.legend(loc='best')
plt.xlabel("X")
plt.ylabel("y")
plt.title("Polynomial Regression with Ridge")
plt.show()