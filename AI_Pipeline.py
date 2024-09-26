import pandas as pd
from ucimlrepo import fetch_ucirepo
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

# Fetch dataset
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

# Data as pandas dataframes
X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets
# metadata
#print(breast_cancer_wisconsin_diagnostic.metadata)

# variable information
#print(breast_cancer_wisconsin_diagnostic.variables)

#using describe to get the statistics of the data
stats = X.describe().T
print(stats[[ 'mean', '25%', '75%', 'min', 'max']])

#plotting the whisker plots
plt.figure(figsize=(15,30))
for i,col in enumerate(X.columns):
    plt.subplot(8,4,i+1)
    sns.boxplot(data=X[col])
    plt.title(f"Plot of {col}")
    plt.xlabel(col)
    plt.ylabel(f"Range of {col}")
plt.tight_layout()
plt.show()

#checking for any missing data
missing_data = X.isnull().sum()
print(missing_data) # We have no missing values in the 10 rows of data that we have selected

#normalizing the data between 0 and 1
X_copy = X.copy()
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(X_copy), columns = X_copy.columns)
print(df_scaled)

#Transforming labels to 0 for benign and 1 for Malignant
y_copy = y.copy()
y_copy['Diagnosis'] = (y_copy['Diagnosis'] == 'M').astype(int)
print(y_copy)

#Splitting the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(df_scaled, y_copy, test_size = 0.2, random_state = 42)
#COnverting the y_train and y_test data to 1D data
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()
print('X_train')
print(X_train.shape)
print('\nX_test')
print(X_test.shape)
print('\ny_train')
print(y_train.shape)
print('\ny_test')
print(y_test.shape)

f1 = []
#Applying the KNN algorithm with the K being set from 1 to 10 and seeing which one gives the highest f1 score
for i in range(1,11):
  knn_neighbours = KNeighborsClassifier(n_neighbors = i)
  knn_neighbours.fit(X_train, y_train)

  y_pred = knn_neighbours.predict(X_test)
  score = f1_score(y_test, y_pred)
  f1.append(score)

plt.xlabel("Number of Neighbors (K)")
plt.ylabel("F1 Score")
plt.title("KNN Model")
plt.plot(f1, "ob")
#showing the performance of each k
for i in range(0, len(f1)):
  print(f"When k is {i+1} and the f1 score is: {f1[i]}")