import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression

# 1. 加载数据
data = fetch_california_housing()
X = data.data
y = data.target

# 2. 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

tree =DecisionTreeRegressor(max_depth=10)
# 3. 初始化模型
models = {
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    
    "AdaBoost": AdaBoostRegressor(tree, n_estimators=100, random_state=42),
    "Linear Regression": LinearRegression()
}

# 4. 训练和评估模型
mse_scores = {}
r2_scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse_scores[name] = mean_squared_error(y_test, y_pred)
    r2_scores[name] = r2_score(y_test, y_pred)

# 5. 输出每个模型的性能指标
for name in models:
    print(f"{name} - MSE: {mse_scores[name]:.2f}, R²: {r2_scores[name]:.2f}")

# 6. 选择最优模型
best_model_name = min(mse_scores, key=mse_scores.get)
best_model = models[best_model_name]

# 7. 使用最优模型进行预测（可选）
# 这里假设我们将再次用整个数据集来训练最优模型
best_model.fit(X, y)
predicted_values = best_model.predict(X)

# 可视化预测结果
plt.figure(figsize=(10, 5))
plt.scatter(y, predicted_values, alpha=0.2)
plt.plot([min(y), max(y)], [min(y), max(y)], 'r--')  # y=x 参考线
plt.xlabel('Actual House Prices')
plt.ylabel('Predicted House Prices')
plt.title('Actual vs Predicted House Prices')
plt.show()

