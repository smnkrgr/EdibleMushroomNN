import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Read the CSV file with the mushroom dataset
mushrooms_data = pd.read_csv("mushrooms.csv")

# Splitting into features and labels
x = mushrooms_data.drop('class',axis=1)
y = mushrooms_data['class']

# Encoding from chars to int
Encoder_x = LabelEncoder() 
for col in x.columns:
    x[col] = Encoder_x.fit_transform(x[col])
Encoder_y=LabelEncoder()
y = Encoder_y.fit_transform(y)

# Dummy veriables for 0/1 values
x=pd.get_dummies(x,columns=x.columns,drop_first=True)

# Split into training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Feature scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Principal component analysis
pca = PCA(n_components=2)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

# x_train shape: (5686, 2)
# y_train shape: (5686,)
# x_test shape: (2438, 2)
# y_test shape: (2438,)
