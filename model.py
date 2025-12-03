import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris


X , y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=200)

model.fit(X, y)

pickle.dump(model, open('logistic_regression_model.pkl', 'wb'))
print("Model trained and saved as 'logistic_regression_model.pkl'")