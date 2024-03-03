import numpy as np
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class PolynomialRegressor2D:
    """Create a regressor that fits a 2D polynomial using the provided
    data points.
    """

    def __init__(self, degree: int, x_train: np.ndarray, y_train: np.ndarray, **kwargs):
        """Initializes the 2D polynomial regressor using some training data.

        Arguments:
            degree (int): degree of the polynomial
            x_train (np.ndarray): (N, 2) data for the coordinates of
                input points
            y_train (np.ndarray): (N, ) data for the coordinates of
                the output prediction.
        """
        self.degree = degree
        poly_features = PolynomialFeatures(degree)
        self.x_poly = poly_features.fit_transform(x_train)
        self.model = Lasso(**kwargs).fit(self.x_poly, y_train)
        self.y_train = y_train

    def __call__(self, x):
        poly_features = PolynomialFeatures(self.degree)
        x_poly = poly_features.fit_transform(x)
        return self.model.predict(x_poly)

    def training_error(self):
        predictions = self.model.predict(self.x_poly)
        return np.mean((self.y_train - predictions) ** 2)


def find_zero_points(x, y, z, tol=1e-5):
    mask = np.abs(z) < tol
    return x[mask], y[mask]
