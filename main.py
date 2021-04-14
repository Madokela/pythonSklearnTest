import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model

model: object = sklearn.linear_model.LinearRegression()
X = np.array([[6], [8], [10], [14], [18]]).reshape(-1, 1)
y = [7, 9, 13, 17.5, 18]
plt.figure()
plt.show()
model.fit(X, y)
test_pizza = np.array([[12]])
predicted_price = model.predict(test_pizza)[0]
print(predicted_price)
