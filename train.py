#
# Purpose: Classify logs wrt their source (eg. Java, Apache, Nagios, etc.)
# Requires: sklearn, numpy, argparse
#

import glob
import argparse
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import svm, naive_bayes, linear_model, tree, ensemble, neighbors, semi_supervised, neural_network, discriminant_analysis
from sklearn.externals import joblib

parser = argparse.ArgumentParser()
parser.add_argument('--train_data_dir', type=str, default='data/train/laptop',
                    help='data directory containing training logs')
parser.add_argument('--test_data_dir', type=str, default='data/test/laptop',
                    help='data directory containing training logs')
parser.add_argument('--save_dir', type=str, default='save',
                    help='directory to store and load training models')
args = parser.parse_args()

def train(algorithm, training_feature_data, training_target_data):
    model = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', algorithm)])
    model.fit(training_feature_data,training_target_data)
    return model

def predict(model, new_docs):
    predicted = model.predict(new_docs['data'])
    accuracy = np.mean(predicted == new_docs['type'])
    return accuracy

def create_log_dict(logfile_path):
    log_collection = {}
    logfiles = glob.glob(logfile_path + "/*.log") # Get list of log files
    for logfile in logfiles:
        file_handle = open(logfile, "r")
        filedata_array = file_handle.read().split('\n')
        file_handle.close()
        # Remove empty lines
        for line in filedata_array:
            if len(line) == 0:
                del filedata_array[filedata_array.index(line)]
        # Add log file data and type
        if log_collection.has_key('data'):
            log_collection['data'] = log_collection['data'] + filedata_array
            # numerise log type for each line
            temp_types = [logfiles.index(logfile)] * len(filedata_array)
            log_collection['type'] = log_collection['type'] + temp_types # Add log type array
        # Cater for first time iteration
        else:
            log_collection['data'] = filedata_array
            temp_types = [logfiles.index(logfile)] * len(filedata_array)
            log_collection['type'] = temp_types

    return log_collection

def report(clf_type,accuracy):
    print("\033[1m" + clf_type + "\033[0m\033[92m")
    print("Success rate: " + str(round(accuracy * 100,2)) + "%\n")
    print

def save_model(algorithm, model):
    save_file = str(algorithm).split('(')[0] + '.pkl'
    joblib.dump(model, args.save_dir + "/" + save_file)

algorithms = [
#    svm.SVC(kernel='linear', C = 1.0),   # QUITE SLOW
    linear_model.SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None),
    naive_bayes.MultinomialNB(),
    naive_bayes.BernoulliNB(),
    tree.DecisionTreeClassifier(max_depth=1000),
    tree.ExtraTreeClassifier(),
    ensemble.ExtraTreesClassifier(),
    svm.LinearSVC(),
#    linear_model.LogisticRegressionCV(multi_class='multinomial'),   # A BIT SLOW
#    neural_network.MLPClassifier(),   # VERY SLOW
    neighbors.NearestCentroid(),
    ensemble.RandomForestClassifier(),
    linear_model.RidgeClassifier(),
]

train_log_collection = create_log_dict(args.train_data_dir)
test_log_collection = create_log_dict(args.test_data_dir)

print("Training log collection => " + str(len(train_log_collection['data'])) + " data entries")
print("Testing log collection => " + str(len(test_log_collection['data'])) + " data entries")
print

for algorithm in algorithms:
    model = train(algorithm, train_log_collection['data'], train_log_collection['type'])
    accuracy = predict(model,test_log_collection)
    report((str(algorithm).split('(')[0]),accuracy)
    save_model(algorithm, model)
