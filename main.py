import numpy as np import pandas as pd

import matplotlib.pyplot as plt import seaborn as sb

%matplotlib inline
wh = pd.read_csv('/Users/dishusharma/Desktop/Book1.csv') wh.head()

Gender	Height	Weight
0	Male	73.847017	241.893563
1	Male	68.781904	162.310473
2	Male	74.110105	212.740856
3	Male	71.730978	220.042470
4	Male wh.shape
(10000, 3)	69.881796	206.349801
print('Our Data has {} samples.'.format(wh.shape[0])) Our Data has 10000 samples.
plt.figure(figsize = (10, 8))
sb.relplot(x = 'Height', y = 'Weight', data = wh, hue = 'Gender') plt.show()
<Figure size 720x576 with 0 Axes>

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X = wh.iloc[:, [1, 2]].values y = wh.iloc[:, 0].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

from sklearn.svm import SVC
clf = make_pipeline(StandardScaler(), SVC(gamma='auto', probability = True))
clf.fit(X_train, y_train)
Pipeline(steps=[('standardscaler', StandardScaler()),
('svc', SVC(gamma='auto', probability=True))]) clf.score(X_train, y_train)
0.9147142857142857
clf.score(X_test, y_test)
 
0.919

clf.predict_proba([[64, 152]])
array([[0.80370245, 0.19629755]])
