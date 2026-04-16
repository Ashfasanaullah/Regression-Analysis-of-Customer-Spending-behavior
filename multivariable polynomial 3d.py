import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

# --------------------------
# Step 1: Generate synthetic data
# --------------------------
np.random.seed(42)
X = 7*np.random.rand(100,1) - 2.8  # feature 1
Y = 7*np.random.rand(100,1) - 2.8  # feature 2
Z = X*2 + Y*2 + 0.2*X + 0.2*Y + 0.1*X*Y + 2 + np.random.randn(100,1)

# --------------------------
# Step 2: Combine features
# --------------------------
XY = np.array([X,Y]).reshape(100,2)  # shape (100,2)

# --------------------------
# Step 3: Polynomial Features
# --------------------------
poly = PolynomialFeatures(degree=2)
XY_poly = poly.fit_transform(XY)

# --------------------------
# Step 4: Fit Linear Regression
# --------------------------
model = LinearRegression()
model.fit(XY_poly, Z)
Z_pred = model.predict(XY_poly)

# --------------------------
# Step 5: 3D Plot
# --------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter actual points
ax.scatter(X, Y, Z, color='blue', label='Actual Z')

# Create grid for smooth surface
X_grid, Y_grid = np.meshgrid(np.linspace(X.min(), X.max(), 30),
                             np.linspace(Y.min(), Y.max(), 30))
XY_grid = np.array([X_grid.ravel(), Y_grid.ravel()]).T
XY_grid_poly = poly.transform(XY_grid)
Z_grid_pred = model.predict(XY_grid_poly).reshape(X_grid.shape)

# Plot regression surface
ax.plot_surface(X_grid, Y_grid, Z_grid_pred, color='red', alpha=0.5)

ax.set_xlabel('X (Feature 1)')
ax.set_ylabel('Y (Feature 2)')
ax.set_zlabel('Z (Dependent Variable)')
ax.set_title('Polynomial Regression With More than 1 independent variable (Degree 2)')
ax.legend()
plt.show()