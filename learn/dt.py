import subprocess
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from features.utils import SKIP_COLUMNS
from questionnaire.fs_receive import RecordFlags


class DecisionTreeLearner:
    def __init__(self):
        self.trained = False
        self.dt = DecisionTreeClassifier(min_samples_split=10, random_state=347)


    def fit(self, features_df):
        """
        Fit decision tree using samples from features data frame
        :param features_df:
        :return:
        """

        features = features_df.columns[1+SKIP_COLUMNS:]

        X = features_df[features]
        y = features_df["record_flag"] == RecordFlags.RECORD_FLAG_ANSWER_TRUE

        self.dt.fit(X, y)
        self.trained = True


    def predict(self, features_df):
        """
        Predict labels of feature samples in data frame using trained tree
        :param features_df:
        :return:
        """
        if not self.trained:
            print("Warning: the tree was not trained!")

        return self.dt.predict(features_df.columns[1+SKIP_COLUMNS:])


    def show(self, feature_names):
        """
        Create tree png using graphviz.
        :param feature_names:
        :return:
        """
        with open("dt.dot", "w") as f:
            export_graphviz(self.dt, out_file=f, feature_names=feature_names)

        command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
        try:
            subprocess.check_call(command)
        except:
            print("Could not run graphviz")