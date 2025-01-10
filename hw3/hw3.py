from collections import Counter

import numpy as np
import pandas as pd

# set random seed
np.random.seed(0)

"""
Tips for debugging:
- Use `print` to check the shape of your data. Shape mismatch is a common error.
- Use `ipdb` to debug your code
    - `ipdb.set_trace()` to set breakpoints and check the values of your variables in interactive mode
    - `python -m ipdb -c continue hw3.py` to run the entire script in debug mode. Once the script is paused, you can use `n` to step through the code line by line.
"""


# 1. Load datasets
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    DO NOT MODIFY THIS FUNCTION.
    """
    # Load iris dataset
    iris = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        header=None,
    )
    iris.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "class",
    ]

    # Load Boston housing dataset
    boston = pd.read_csv(
        "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    )

    return iris, boston


# 2. Preprocessing functions
def train_test_split(
    df: pd.DataFrame, target: str, test_size: float = 0.3
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Shuffle and split dataset into train and test sets
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    # Split target and features
    X_train = train.drop(target, axis=1).values
    y_train = train[target].values
    X_test = test.drop(target, axis=1).values
    y_test = test[target].values

    return X_train, X_test, y_train, y_test


def normalize(X: np.ndarray) -> np.ndarray:
    # Normalize features to [0, 1]
    # You can try other normalization methods, e.g., z-score, etc.
    # TODO: 1%

    # normalization 
    # use Z-score (normalize)
    X_min = X.min()
    X_max = X.max()
    return (X - X_min) / (X_max - X_min)


def standize(X: np.ndarray) -> np.ndarray:
    # standization 
    # use Z-score (standardize)
    mean_X = np.mean(X, axis=0)
    std_X = np.std(X, axis=0)
    normalized_data = (X - mean_X) / std_X
    return normalized_data

def encode_labels(y: np.ndarray) -> np.ndarray:
    """
    Encode labels to integers.
    """
    # TODO: 1%
    buf = []
    for labels in y:
        if (labels not in buf):
            buf.append(labels)
    for i in range(len(y)):
        for j in range(len(buf)):
            if y[i] == buf[j]:
                y[i] = j    
    return y


# 3. Models
class LinearModel:
    def __init__(
        self, learning_rate=0.01, iterations=1000, model_type="linear", n_classes=0
    ) -> None:
        self.learning_rate = learning_rate
        self.iterations = iterations
        # You can try different learning rate and iterations
        self.model_type = model_type
        self.n_classes = n_classes

        assert model_type in [
            "linear",
            "logistic",
        ], "model_type must be either 'linear' or 'logistic'"

    def logistic_function(self, X: np.ndarray):    
        return 1/ (1 + np.exp(-X))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.insert(X, 0, 1, axis=1)
        self.n_classes = len(np.unique(y))
        n_features = X.shape[1]
        #self.theta = np.zeros(n_features)y_pred
        if self.model_type == "logistic":
            self.theta = np.zeros((n_features, self.n_classes))
            for _ in range(self.iterations):
                gradient = self._compute_gradients(X, y)
                self.theta -= self.learning_rate * gradient
            pass
        else:
            self.theta = np.zeros(n_features)
            for _ in range(self.iterations):
                gradient = self._compute_gradients(X, y) 
                gradient = gradient.astype("float64")
                self.theta -= self.learning_rate * gradient
            pass
        # raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.insert(X, 0, 1, axis=1)
        if self.model_type == "linear":
            # TODO: 2%
            res = np.dot(X, self.theta)
            return res
            #raise NotImplementedError
        elif self.model_type == "logistic":
            # TODO: 2%
            scores = np.dot(X, self.theta)
            probs = self._softmax(scores)
            res = np.argmax(probs, axis=1)
            res = (res + self.n_classes - 1)  % self.n_classes
            return res
            #raise NotImplementedError

    def _compute_gradients(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.model_type == "linear":
            # TODO: 3%
            predictions = np.dot(X, self.theta)
            errors = predictions - y
            gradient = np.dot(X.T, errors) / len(y)
            return gradient
            # raise NotImplementedError
        elif self.model_type == "logistic":
            # TODO: 3%
            self.n_classes = len(np.unique(y))
            scores = np.dot(X, self.theta)
            probs = self._softmax(scores)
            y = y.astype(int)
            y_one_hot = np.eye(self.n_classes)[y]  # 將y轉換為one-hot編碼
            now_gradient = np.dot(X.T, (probs - y_one_hot)) / X.shape[0]
            return now_gradient
            # raise NotImplementedError

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        exp = np.exp(z)
        return exp / np.sum(exp, axis=1, keepdims=True)


class DecisionTree:
    def __init__(self, max_depth: int = 5, model_type: str = "classifier", n_classes: int = 0):
        self.max_depth = max_depth
        self.model_type = model_type
        self.n_classes = n_classes

        assert model_type in [
            "classifier",
            "regressor",
        ], "model_type must be either 'classifier' or 'regressor'"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.n_classes = len(np.unique(y))
        self.tree = self._build_tree(X, y, 0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        res = np.array([self._traverse_tree(x, self.tree) for x in X])
        res = (res + self.n_classes - 1) % self.n_classes
        return res

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> dict:
        if depth >= self.max_depth or self._is_pure(y):
            return self._create_leaf(y)

        feature, threshold = self._find_best_split(X, y)
        # TODO: 4%
        mask = X[:, feature] <= threshold
        left_child = self._build_tree(X[mask], y[mask], depth + 1)
        right_child = self._build_tree(X[~mask], y[~mask], depth + 1)

        # raise NotImplementedError


        return {
            "feature": feature,
            "threshold": threshold,
            "left": left_child,
            "right": right_child,
        }

    def _is_pure(self, y: np.ndarray) -> bool:
        return len(set(y)) == 1

    def _create_leaf(self, y: np.ndarray):
        if self.model_type == "classifier":
            # TODO: 1%
            # Return the most common class
            y_int = y.astype(int)
            return np.bincount(y_int).argmax()            
            # raise NotImplementedError
        else:
            # TODO: 1%
            return np.mean(y)
            # raise NotImplementedError

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
        best_gini = float("inf")
        best_mse = float("inf")
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            sorted_indices = np.argsort(X[:, feature])
            for i in range(1, len(X)):
                if X[sorted_indices[i - 1], feature] != X[sorted_indices[i], feature]:
                    threshold = (
                        X[sorted_indices[i - 1], feature]
                        + X[sorted_indices[i], feature]
                    ) / 2
                    mask = X[:, feature] <= threshold
                    left_y, right_y = y[mask], y[~mask]

                    if self.model_type == "classifier":
                        gini = self._gini_index(left_y, right_y)
                        if gini < best_gini:
                            best_gini = gini
                            best_feature = feature
                            best_threshold = threshold
                    else:
                        mse = self._mse(left_y, right_y)
                        if mse < best_mse:
                            best_mse = mse
                            best_feature = feature
                            best_threshold = threshold

        return best_feature, best_threshold

    def _gini_index(self, left_y: np.ndarray, right_y: np.ndarray) -> float:
        # TODO: 4%
        # Calculate the Gini index for the split
        def gini(y):
            _, counts = np.unique(y, return_counts=True)
            probabilities = counts / counts.sum()
            return 1 - np.sum(probabilities ** 2)
        
        return gini(left_y) * (len(left_y) / (len(left_y) + len(right_y))) + \
               gini(right_y) * (len(right_y) / (len(left_y) + len(right_y)))
        # raise NotImplementedError


    def _mse(self, left_y: np.ndarray, right_y: np.ndarray) -> float:
        # TODO: 4%
        # Calculate the MSE for the split
        left_mse = np.mean((left_y - np.mean(left_y)) ** 2) if len(left_y) > 0 else 0
        right_mse = np.mean((right_y - np.mean(right_y)) ** 2) if len(right_y) > 0 else 0
        return (left_mse * len(left_y) + right_mse * len(right_y)) / (len(left_y) + len(right_y))
        #raise NotImplementedError

    def _traverse_tree(self, x: np.ndarray, node: dict):
        if isinstance(node, dict):
            feature, threshold = node["feature"], node["threshold"]
            if x[feature] <= threshold:
                return self._traverse_tree(x, node["left"])
            else:
                return self._traverse_tree(x, node["right"])
        else:
            return node


class RandomForest:
    def __init__(
        self,
        n_estimators: int = 20,
        max_depth: int = 50,
        model_type: str = "classifier",
    ):
        # TODO: 1%
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model_type = model_type
        self.trees = [DecisionTree(max_depth=max_depth, model_type=model_type) for _ in range(n_estimators)]
        # raise NotImplementedError


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples = X.shape[0]
        self.trees_ = []  # List to store fitted trees
        for tree in self.trees:
            # TODO: 2%
            # bootstrap_indices = np.random.choice(
            bootstrap_indices = np.random.choice(np.arange(n_samples), size=n_samples, replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees_.append(tree)           

            # raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: 2%
        # Collect predictions from each tree
        predictions = np.array([tree.predict(X) for tree in self.trees_])
        
        if self.model_type == "classifier":
            # For classification, find the most common label across trees for each sample
            predictions = self._most_common_label(predictions)
        else:
            # For regression, take the mean prediction across trees
            predictions = np.mean(predictions, axis=0)
        
        return predictions

        # raise NotImplementedError
    def _most_common_label(self, predictions):
        # Custom method to find the most common label without using mode
        n_samples = predictions.shape[1]
        final_predictions = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            labels, counts = np.unique(predictions[:, i], return_counts=True)
            final_predictions[i] = labels[np.argmax(counts)]
        
        return final_predictions

# 4. Evaluation metrics
def accuracy(y_true, y_pred):
    # TODO: 1%
    #print(y_true, y_pred)
    return np.sum(y_true == y_pred) / len(y_true)
    # raise NotImplementedError


def mean_squared_error(y_true, y_pred):
    # TODO: 1%
    #print(y_true, y_pred)
    return np.sum((y_true - y_pred) ** 2) / len(y_true)
    # raise NotImplementedError


# 5. Main function
def main():
    iris, boston = load_data()

    # Iris dataset - Classification
    X_train, X_test, y_train, y_test = train_test_split(iris, "class")
    X_train, X_test = normalize(X_train), normalize(X_test)
    # X_train, X_test = standize(X_train), standize(X_test)

    y_train, y_test = encode_labels(y_train), encode_labels(y_test)

    logistic_regression = LinearModel(model_type="logistic")
    logistic_regression.fit(X_train, y_train)
    y_pred = logistic_regression.predict(X_test)
    print("Logistic Regression Accuracy:", accuracy(y_test, y_pred))

    decision_tree_classifier = DecisionTree(model_type="classifier")
    decision_tree_classifier.fit(X_train, y_train)
    y_pred = decision_tree_classifier.predict(X_test)
    print("Decision Tree Classifier Accuracy:", accuracy(y_test, y_pred))

    random_forest_classifier = RandomForest(model_type="classifier")
    random_forest_classifier.fit(X_train, y_train)
    y_pred = random_forest_classifier.predict(X_test)
    print("Random Forest Classifier Accuracy:", accuracy(y_test, y_pred))

    # Boston dataset - Regression
    X_train, X_test, y_train, y_test = train_test_split(boston, "medv")
    X_train, X_test = normalize(X_train), normalize(X_test)
    # X_train, X_test = standize(X_train), standize(X_test)


    linear_regression = LinearModel(model_type="linear")
    linear_regression.fit(X_train, y_train)
    y_pred = linear_regression.predict(X_test)
    print("Linear Regression MSE:", mean_squared_error(y_test, y_pred))

    decision_tree_regressor = DecisionTree(model_type="regressor")
    decision_tree_regressor.fit(X_train, y_train)
    y_pred = decision_tree_regressor.predict(X_test)
    print("Decision Tree Regressor MSE:", mean_squared_error(y_test, y_pred))

    random_forest_regressor = RandomForest(model_type="regressor")
    random_forest_regressor.fit(X_train, y_train)
    y_pred = random_forest_regressor.predict(X_test)
    print("Random Forest Regressor MSE:", mean_squared_error(y_test, y_pred))


if __name__ == "__main__":
    main()
