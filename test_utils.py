from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def test_data_shape():
    data = load_breast_cancer()
    X, y = data.data, data.target
    assert X.shape == (569, 30)
    assert y.shape == (569,)

def test_train_test_split():
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    assert len(X_train) == 455
    assert len(X_test) == 114

def test_model_accuracy():
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    assert acc > 0.90
