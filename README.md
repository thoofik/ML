# ML

1.   


import matplotlib
matplotlib.use('TkAgg') # Use TkAgg backend
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
# Load the California Housing Dataset
data = fetch_california_housing(as_frame=True)
df = data.frame
# Set up the grid layout
n_columns = 3
n_rows = (len(df.select_dtypes(include=['float64', 'int64']).columns) * 2 + n_columns - 1) // n_columns
fig, axes = plt.subplots(n_rows, n_columns, figsize=(15, 10))
axes = axes.flatten()
# Create plots
columns = df.select_dtypes(include=['float64', 'int64']).columns
for i, column in enumerate(columns):
 # Histogram
 ax = axes[i]
 df[column].hist(bins=30, edgecolor='black', ax=ax)
 ax.set(title=f"Histogram of {column}", xlabel=column, ylabel="Frequency")
 # Box Plot
 ax = axes[len(columns) + i]
 df.boxplot(column=column, grid=False, ax=ax)
 ax.set(title=f"Box Plot of {column}")
# Adjust layout and display
plt.tight_layout()
plt.savefig("combined_plots.png")
plt.show()


2   



import matplotlib
matplotlib.use('TkAgg') # Use TkAgg backend
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
# Load the dataset
df = fetch_california_housing(as_frame=True).frame
# Set up the grid layout for plots (2 rows, 1 column)
fig, axes = plt.subplots(2, 1, figsize=(12, 12))
# Heatmap of correlation matrix (Top plot)
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=axes[0])
axes[0].set_title("Correlation Matrix Heatmap")
# Pair plot for selected features (Bottom plot)
sns.pairplot(df[['MedInc', 'HouseAge', 'AveRooms', 'AveOccup', 'MedHouseVal']], diag_kind="kde")
plt.subplots_adjust(hspace=0.4) # Adjust space between subplots
# Show the combined figure
plt.show()

3

import matplotlib
matplotlib.use('TkAgg') # Use TkAgg backend
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
# Standardize the features (important for PCA)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
# Apply PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)
# Create a DataFrame for the 2 principal components
pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
# Visualize the result
plt.figure(figsize=(8, 6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], c=iris.target, cmap='viridis')
plt.title("PCA of Iris Dataset (2 components)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label='Target')
plt.show()

4


import pandas as pd
# Load the dataset
df = pd.read_csv('training_data.csv')
# Assume the last column is the class (target variable)
X = df.iloc[:, :-1] # Features (all columns except the last)
y = df.iloc[:, -1] # Class (the last column)
# Find-S algorithm
def find_s_algorithm(X, y):
 # Initialize the hypothesis to the most general hypothesis (all attributes can be anything)
 hypothesis = ['?' for _ in range(X.shape[1])]
 # Loop through all examples in the dataset
 for i in range(len(X)):
 if y[i] == 'Yes': # If the example is a positive example
 for j in range(len(X.columns)):
 # If the hypothesis is still general or the feature matches the example, keep it
 if hypothesis[j] == '?' or hypothesis[j] == X.iloc[i, j]:
 hypothesis[j] = X.iloc[i, j]
 # If the feature doesn't match, make it specific to the example
 else:
 hypothesis[j] = '?'
 return hypothesis
# Get the most specific hypothesis
hypothesis = find_s_algorithm(X, y)
# Output the hypothesis
print("Hypothesis consistent with the positive examples:", hypothesis)

5


Python Code:
import matplotlib
matplotlib.use('TkAgg') # Use the TkAgg backend for stable display
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
# Step 1: Generate 100 random values of x in the range [0, 1]
np.random.seed(42) # For reproducibility
x_values = np.random.rand(100, 1) # 100 random values in the range [0,1]
# Step 2: Label the first 50 points as Class1 and the rest as Class2
y_labels = np.array(['Class1' if x <= 0.5 else 'Class2' for x in x_values.flatten()])
# Split into training and testing sets
X_train = x_values[:50] # First 50 points
y_train = y_labels[:50] # First 50 labels
X_test = x_values[50:] # Remaining 50 points
y_test = y_labels[50:] # Remaining 50 labels
# Step 3: Classify using KNN for different k values
k_values = [1, 2, 3, 4, 5, 20, 30]
plt.figure(figsize=(12, 8))
for i, k in enumerate(k_values, 1):
 # Initialize the k-NN classifier with the current k value
 knn = KNeighborsClassifier(n_neighbors=k)

 # Fit the model on the training data
 knn.fit(X_train, y_train)

 # Predict the labels for the test set
 y_pred = knn.predict(X_test)

 # Plot the decision boundary and the points
 plt.subplot(3, 3, i)
 plt.scatter(X_test, y_test, color='blue', label='True Label')
 plt.scatter(X_test, y_pred, color='red', marker='x', label='Predicted Label')

 plt.title(f"KNN with k={k}")
 plt.xlabel("X value")
 plt.ylabel("Class Label")
 plt.legend(loc='best')
 plt.grid(True)
 plt.tight_layout()
plt.show()
# Step 4: Evaluate classification accuracy for each k value
for k in k_values:
 knn = KNeighborsClassifier(n_neighbors=k)
 knn.fit(X_train, y_train)
 accuracy = knn.score(X_test, y_test)
 print(f"Accuracy for k={k}: {accuracy:.2f}")


6

import matplotlib
matplotlib.use('TkAgg') # Use TkAgg backend for interactive plotting
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
# Load the dataset and select a feature
data = fetch_california_housing(as_frame=True)
df = data.frame
X = df['MedInc'].values.reshape(-1, 1)
y = df['MedHouseVal'].values
# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Locally Weighted Regression (LWR)
def locally_weighted_regression(X_train, y_train, X_test, tau=0.1):
 predictions = []
 for x in X_test:
 weights = np.exp(-np.sum((X_train - x) ** 2, axis=1) / (2 * tau ** 2))
 X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]

 # Solve the weighted least squares problem using np.linalg.lstsq for efficiency
 theta, _, _, _ = np.linalg.lstsq(X_train_b * weights[:, np.newaxis], y_train * weights, rcond=None)

 # Make prediction for the current test point
 X_test_b = np.c_[1, x] # Add bias term
 predictions.append(X_test_b @ theta)
 return np.array(predictions)
# Predict values for the test set
y_pred = locally_weighted_regression(X_train, y_train, X_test, tau=0.1)
# Plot results
plt.scatter(X_test, y_test, color='blue', label='True values')
plt.scatter(X_test, y_pred, color='red', label='Predicted values')
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.title('Locally Weighted Regression (LWR)')
plt.legend()
plt.grid(True)
# Show plot (interactive window)
plt.show()
# Evaluate performance
mse = np.mean((y_pred - y_test) ** 2)
print(f"Mean Squared Error: {mse:.4f}")

7

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg') # Use TkAgg backend
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
# Load California Housing dataset for Linear Regression
data = fetch_california_housing(as_frame=True)
X = data.data[['AveRooms']]
y = data.target
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Linear Regression
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
y_pred = linear_reg.predict(X_test)
# Polynomial Regression
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X_train)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y_train)
y_pred_poly = poly_reg.predict(poly.fit_transform(X_test))
# Plotting results
plt.subplot(1, 2, 1)
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red')
plt.title('Linear Regression')
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred_poly, color='green')
plt.title('Polynomial Regression')
plt.tight_layout()
plt.show()
# Output MSE
print(f"Linear Regression - MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"Polynomial Regression - MSE: {mean_squared_error(y_test, y_pred_poly):.4f}")

8

Python Code:
# Import necessary libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
# Load the Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Create a DecisionTreeClassifier instance and train it
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
# Predict the test set results
y_pred = clf.predict(X_test)
# Evaluate the classifier performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on Test Set: {accuracy:.4f}')
# Classify a new sample (randomly selected from the test set for demonstration)
new_sample = X_test[0].reshape(1, -1) # Take the first sample from the test set
predicted_class = clf.predict(new_sample)
# Output the predicted class (0: malignant, 1: benign)
print(f'Predicted Class for New Sample: {"Benign" if predicted_class == 1 else "Malignant"}')

9

import numpy as np
from scipy.io import loadmat
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Load the olivettifaces.mat file (ensure it's in the same directory or update the path)
data = loadmat('olivettifaces.mat')
# Inspect the keys in the dataset
print("Keys in the dataset:", data.keys())
# Use 'faces' as the feature matrix
X = data['faces'] # Features (faces), this is the matrix of images
# Assuming labels are the index of faces (0-40 for 40 individuals, 10 images per individual)
y = np.repeat(np.arange(40), 10) # 40 classes (individuals), 10 images per class
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X.T, y, test_size=0.3, random_state=42) # Transpose
for correct shape
# Create and train the Naive Bayes classifier
model = GaussianNB()
model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

10

import matplotlib
matplotlib.use('TkAgg') # Use the TkAgg backend for interactive GUI rendering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target
# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Apply KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)
# Visualize the clustering result
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans, cmap='viridis', edgecolors='k')
plt.title('K-Means Clustering (2D) on Wisconsin Breast Cancer Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster')
# Show the plot interactively using TkAgg
plt.show()
# Optionally, print cluster centers
print("Cluster centers:\n", kmeans.cluster_centers_)
