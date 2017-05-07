
"""
>>> learn.find_params(
...     DecisionTreeClassifier(),{
...     'criterion' : ["gini", "entropy"],
...     'splitter' : ["best","random"],
...     'max_features' : ["auto","sqrt","log2"],              
...     'max_depth' : [2,3,4,5,6],
...     'random_state' : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]},
...     train, train_labels)
RandomizedSearchCV took 0.12 seconds for 20 candidates parameter settings.
Model with rank: 1
Mean validation score: 0.610 (std: 0.038)
Parameters: {'criterion': 'gini', 'splitter': 'best', 'max_features': 'auto', 'random_state': 2, 'max_depth': 6}
"""
