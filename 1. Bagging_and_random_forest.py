from sklearn.model_selection import cross_validate
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import numpy as np
from matplotlib import pyplot as plt

dataset = datasets.load_digits()
X = dataset['data']
y = dataset['target']
print (dataset['DESCR'])

clf = DecisionTreeClassifier()
scores = cross_validate(clf, X, y, cv = 10)


clf = BaggingClassifier(n_estimators = 100)
scores = cross_validate(clf, X, y, cv = 10)

d = X.shape[1]
number_of_features = int(d ** 0.5)
number_of_features

clf = BaggingClassifier(n_estimators = 100, max_features = number_of_features)
scores = cross_validate(clf, X, y, cv = 10)


clf = BaggingClassifier(DecisionTreeClassifier(max_features = number_of_features), n_estimators = 100)
scores = cross_validate(clf, X, y, cv = 10)


clf = RandomForestClassifier(n_estimators = 100, max_features = number_of_features)
scores = cross_validate(clf, X, y, cv = 10)


#Depending on the number of trees
number_of_estimators = np.arange(5, 150, 5)
scores_estimators = []
for n in number_of_estimators:
    clf = RandomForestClassifier(n_estimators = n, max_features = number_of_features)
    scores_estimators.append(cross_validate(clf, X, y, cv = 10))
        

plt.plot(number_of_estimators, scores_estimators)
plt.xlabel('number of trees')
plt.ylabel('accuracy')
plt.title('RandomForest score')

#Dependence on the number of signs
number_of_features_list = np.arange(5, d, 5)
scores_features = []
for n in number_of_features_list:
    clf = RandomForestClassifier(n_estimators = 100, max_features = n)
    scores_features.append(cross_validate.cross_val_score(clf, X, y, cv = 10).mean())
    

plt.plot(number_of_features_list, scores_features)
plt.xlabel('number of features')
plt.ylabel('accuracy')
plt.title('RandomForest score')

#Dependence on the depth of trees
depths = np.arange(1, 30, 1)
scores_depth = []
for n in depths:
    clf = RandomForestClassifier(n_estimators = 100, max_features = number_of_features, max_depth = n)
    scores_depth.append(cross_validate.cross_val_score(clf, X, y, cv = 10).mean())

plt.plot(depths, scores_depth)
plt.xlabel('depth of trees')
plt.ylabel('accuracy')
plt.title('RandomForest score')
