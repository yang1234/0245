import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Generate synthetic data
x1 = np.arange(0, 10, 0.1)
x2 = np.arange(0, 10, 0.1)
x1, x2 = np.meshgrid(x1, x2)
y = np.sin(x1) * np.cos(x2) + np.random.normal(scale=0.1, size=x1.shape)

# Flatten the arrays
x1 = x1.flatten()
x2 = x2.flatten()
y = y.flatten()
X = np.vstack((x1, x2)).T

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the decision tree regressor
tree = DecisionTreeRegressor(max_depth=5)
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Decision Tree MSE:", mse)
print("Decision Tree R2:", r2)

# Analysis of max_depth and splitter
depths = range(1, 20)
splitters = ['best', 'random']
results = []

for depth in depths:
    for splitter in splitters:
        model = DecisionTreeRegressor(max_depth=depth, splitter=splitter)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append((depth, splitter, mse, r2))

# Finding minimum MSE and corresponding R2
min_mse_result = min(results, key=lambda x: x[2])
min_depth, min_splitter, min_mse, min_r2 = min_mse_result

print(f"Minimum Decision Tree MSE: {min_mse}")
print(f"Corresponding R2: {min_r2}")
print(f"At Depth: {min_depth}, Splitter: {min_splitter}")

# Prepare data for plotting
mse_best = [result[2] for result in results if result[1] == 'best']
mse_random = [result[2] for result in results if result[1] == 'random']

# Plotting results
plt.figure(figsize=(10, 5))
plt.plot(depths, mse_best, label='Best Splitter', marker='o')
plt.plot(depths, mse_random, label='Random Splitter', marker='x')
plt.title('Decision Tree MSE by Max Depth and Splitter Type')
plt.xlabel('Max Depth')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.show()

# Polynomial Regression for comparison
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
X_train_poly, X_test_poly, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=42)
poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)
y_poly_pred = poly_reg.predict(X_test_poly)
poly_mse = mean_squared_error(y_test, y_poly_pred)
poly_r2 = r2_score(y_test, y_poly_pred)

print("Polynomial Regression MSE:", poly_mse)
print("Polynomial Regression R^2:", poly_r2)

# Decision tree vs. Polynomial Regression comparison
print(f"Decision Tree Best MSE: {min_mse}, Polynomial Regression MSE: {poly_mse}")
print(f"Decision Tree Best R^2: {min_r2}, Polynomial Regression R^2: {poly_r2}")
print(f"At Depth: {min_depth}, Splitter: {min_splitter}")
