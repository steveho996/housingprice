# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (replace with your actual path)
data = pd.read_csv('/workspaces/housingprice/train.csv') 

# Select features and target
features = ['OverallQual', 'GrLivArea', 'GarageCars']
target = 'SalePrice'

X = data[features]
y = data[target]

# Handle categorical variables (if any)
# If you have categorical variables, you can use One-Hot Encoding
# For example, if 'Neighborhood' was a categorical feature:
# categorical_features = ['Neighborhood']
# encoder = OneHotEncoder(handle_unknown='ignore')
# encoded_features = encoder.fit_transform(data[categorical_features]).toarray()
# encoded_df = pd.DataFrame(encoded_features)
# X = pd.concat([X, encoded_df], axis=1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred_linear = linear_model.predict(X_test)

# Evaluate the model
linear_mse = mean_squared_error(y_test, y_pred_linear)
print(f'Linear Regression MSE: {linear_mse}')

# 2. Neural Network Model with Keras
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)  

# Make predictions on the testing set
y_pred_nn = model.predict(X_test).flatten()

# Evaluate the model
nn_mse = mean_squared_error(y_test, y_pred_nn)
print(f'Neural Network MSE: {nn_mse}')

# 3. Compare Model Performance
# Plotting the comparison of MSE
model_names = ['Linear Regression', 'Neural Network']
mse_values = [linear_mse, nn_mse]

plt.figure(figsize=(10, 6))
plt.bar(model_names, mse_values, color=['blue', 'green'])
plt.title('Model Comparison - MSE')
plt.ylabel('Mean Squared Error')
plt.show()

# 4. Visualizing Data Distributions
# Visualizing the distribution of features and target variable
plt.figure(figsize=(10, 6))
sns.histplot(data['OverallQual'], kde=True)
plt.title('Distribution of Overall Quality')
plt.xlabel('Overall Quality')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['GrLivArea'], kde=True)
plt.title('Distribution of Ground Living Area')
plt.xlabel('Ground Living Area (sq ft)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['GarageCars'], kde=True)
plt.title('Distribution of Garage Cars')
plt.xlabel('Number of Garage Cars')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['SalePrice'], kde=True)
plt.title('Distribution of Sale Price')
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.show()

# 5. Visualizing Actual vs Predicted Values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_linear, label='Linear Regression', alpha=0.6)
plt.scatter(y_test, y_pred_nn, label='Neural Network', alpha=0.6)
plt.plot([0, max(y_test)], [0, max(y_test)], '--k', label='Ideal Prediction')
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs Predicted Sale Price')
plt.legend()
plt.show()

# 6. Feature Importance (For Linear Regression)
# Getting the coefficients for linear regression model
coefficients = pd.DataFrame(linear_model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)
