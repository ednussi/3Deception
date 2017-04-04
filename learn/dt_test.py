os.chdir(r'\users\gregory\dropbox\code\3deception')
import subprocess
import random
from sklearn.tree import DecisionTreeClassifier, export_graphviz

dt = DecisionTreeClassifier(min_samples_split=1, max_depth=4, random_state=347)

raw = pd.read_csv('grisha_features.csv')
raw_true = raw[raw.record_flag == 3]
raw_false = raw[raw.record_flag == 4]
train_true = raw_true[:20]
test_true = raw_true[20:]
train_false = raw_false[:100]
test_false = raw_false[100:]
features = raw.columns[4:].tolist()
train = pd.concat([train_false, train_true])
train_data = train[features].values.tolist()
test = pd.concat([test_false, test_true])
test_data = test[features].values.tolist()
labels_train = [0]*train_false.shape[0] + [1]*train_true.shape[0]
labels_test = [0]*test_false.shape[0] + [1]*test_true.shape[0]
train_with_labels = list(zip(train_data, labels_train))
test_with_labels = list(zip(test_data, labels_test))
random.shuffle(train_with_labels)
random.shuffle(test_with_labels)
shuffled_train_data, shuffled_train_labels = zip(*train_with_labels)
shuffled_test_data, shuffled_test_labels = zip(*test_with_labels)

dt.fit(shuffled_train_data, shuffled_train_labels)
with open("dt.dot", "w") as f:
    export_graphviz(dt, out_file=f, feature_names=features, class_names=['LIE', 'TRUTH'], filled=True, rounded=True, special_characters=True)

command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
try:
    subprocess.check_call(command)
except:
    print("Could not run graphviz")

test_prediction = dt.predict(shuffled_test_data)
acc = sum(np.array(shuffled_test_labels) == test_prediction) / len(test_prediction.tolist())
print(acc)