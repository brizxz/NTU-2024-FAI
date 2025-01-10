import numpy as np


"""
Implementation of Principal Component Analysis.
"""
class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X: np.ndarray) -> None:
        #TODO: 10%
        self.mean = np.mean(X, axis=0)
        X_fit = X - self.mean
        X_cov =  np.cov(X_fit.T)

        # eigenval and eigenvector
        eigenval, eigenvector = np.linalg.eigh(X_cov)
        index = np.argsort(eigenval)[::-1]

        # top n componenets eigenvectors
        temp = eigenvector[:,index]
        self.components = temp[:, 0 : self.n_components]
        # raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        # ODO: 2%
        # Hint: Use the calculated principal components to project the data onto a lower dimensional space
        X_trans = X - self.mean
        result = X_trans.dot(self.components)
        return result

        # raise NotImplementedError

    def reconstruct(self, X):
        #raise NotImplementedError
        
        #TODO: 2%
        # Hint: Use the calculated principal components to reconstruct the data back to its original space

        X_buf = X - self.mean
        X = X_buf.dot(self.components)
        X_re = X.dot(self.components.T)
        return X_re + self.mean