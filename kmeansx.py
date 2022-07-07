import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

import warnings

warnings.filterwarnings('ignore')

data = 'healthcare-dataset-stroke-data-clean.csv'
TARGET = "stroke"

df = pd.read_csv(data)


df.shape


df.head()

df.info()


df.isnull().sum()


#df.drop(['Column1', 'Column2', 'Column3', 'Column4'], axis=1, inplace=True)

df.info()

df.describe()


#define features and targets

X = df

y = df[TARGET]

le = LabelEncoder()

X[TARGET] = le.fit_transform(X[TARGET])

y = le.transform(y)

X.info()

X.head()

cols = X.columns


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=0) 

kmeans.fit(X)

kmeans.cluster_centers_

kmeans.inertia_

labels = kmeans.labels_

# check how many of the samples were correctly labeled
correct_labels = sum(y == labels)

print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))

print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))

