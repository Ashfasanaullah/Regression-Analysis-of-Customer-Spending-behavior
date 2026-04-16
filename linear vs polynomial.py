import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load dataset
data = pd.read_csv("Mall_Customers.csv")

# Feature and target
X = data[['Annual Income (k$)']]
y = data['Spending Score']

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)

# Polynomial Regression (degree 3)
poly3 = PolynomialFeatures(degree=3)
X_poly3 = poly3.fit_transform(X)
poly_reg3 = LinearRegression()
poly_reg3.fit(X_poly3, y)
y_poly_pred3 = poly_reg3.predict(X_poly3)

# Polynomial Regression (degree 10) 🔥
poly10 = PolynomialFeatures(degree=10)
X_poly10 = poly10.fit_transform(X)
poly_reg10 = LinearRegression()
poly_reg10.fit(X_poly10, y)
y_poly_pred10 = poly_reg10.predict(X_poly10)

# Plot
plt.scatter(X, y, color='blue', label='Original Data')
plt.plot(X, y_pred_lin, color='green', label='Linear Regression')
plt.plot(X, y_poly_pred3, color='red', label='Polynomial Regression (deg 3)')
plt.plot(X, y_poly_pred10, color='purple', label='Polynomial Regression (deg 10)')  # 👈 new line

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.show()