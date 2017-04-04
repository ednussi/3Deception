os.chdir(r'\users\gregory\dropbox\code\3deception')
import subprocess
from sklearn.tree import DecisionTreeClassifier, export_graphviz
dt = DecisionTreeClassifier(min_samples_split=1, random_state=347)
raw = pd.read_csv('grisha_features.csv')
raw_true = raw[raw.record_flag == 3]
raw_false = raw[raw.record_flag == 4]
train_true = raw_true[:20]
test_true = raw_true[20:]
train_false = raw_false[:100]
test_false = raw_false[100:]
train = pd.concat([train_false, train_true])
test = pd.concat([test_false, test_true])
features = raw.columns[4:].tolist()
labels_train = [False]*train_false.shape[0] + [True]*train_true.shape[0]
labels_test = [False]*test_false.shape[0] + [True]*test_true.shape[0]
dt.fit(train[features].values.tolist(), labels_train)
with open("dt.dot", "w") as f:
    export_graphviz(dt, out_file=f, feature_names=features)

command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
try:
    subprocess.check_call(command)
except:
    print("Could not run graphviz")
test_prediction = dt.predict(test[features].values.tolist())
num_right = sum(np.array(labels_test) == test_prediction)
accuracy = 1. * num_right / len(test_prediction)