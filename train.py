#
# Purpose: Classify logs wrt their source (eg. Java, Apache, Nagios, etc.)
# Requires: sklearn, numpy, argparse
#

import glob
import timeit
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
                    help='directory to store and load training pipeline models')
args = parser.parse_args()

def train(algorithm, training_feature_data, training_target_data):
    model = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', algorithm)])
    model.fit(training_feature_data,training_target_data)
    save_file = str(algorithm).split('(')[0] + '.pkl'
    joblib.dump(model, args.save_dir + "/" + save_file)
    return model

def predict(model, new_docs):
    predicted = model.predict(new_docs['data'])
    success_ratio = np.mean(predicted == new_docs['type'])
    return success_ratio

def create_log_array(logfile_path):
    log_collection = {}
    log_types = []
    logfiles = glob.glob(logfile_path + "/*.log")
    for logfile in logfiles:
        log_types.append(logfile.split('.')[0].split('/')[-1])
        file_handle = open(logfile, "r")
        tempdata = file_handle.read().split('\n')
        file_handle.close()
        for i in tempdata:
            if len(i) == 0:
                del tempdata[tempdata.index(i)]
        if log_collection.has_key('data'):
            log_collection['data'] = log_collection['data'] + tempdata
            temptypes = [logfiles.index(logfile)] * len(tempdata)
            log_collection['type'] = log_collection['type'] + temptypes
        else:
            log_collection['data'] = tempdata
            temptypes = [logfiles.index(logfile)] * len(tempdata)
            log_collection['type'] = temptypes
    return log_collection

def report(clf_type,ratio):
    print("\033[1m" + clf_type + "\033[0m\033[92m")
    print("Success rate: " + str(round(ratio * 100,2)) + "%\n")
    print

algorithms = [
    svm.SVC(kernel='linear', C = 1.0, verbose=True),   # SLOW
    linear_model.SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None),
    naive_bayes.MultinomialNB(),
    naive_bayes.BernoulliNB(),
    tree.DecisionTreeClassifier(max_depth=5),
    tree.ExtraTreeClassifier(),
    ensemble.ExtraTreesClassifier(),
    neighbors.KNeighborsClassifier(),
    svm.LinearSVC(multi_class='crammer_singer'),   # SLOW
    linear_model.LogisticRegressionCV(multi_class='multinomial'),
    neural_network.MLPClassifier(),   # SLOW
    neighbors.NearestCentroid(),
    ensemble.RandomForestClassifier(),
    linear_model.RidgeClassifier(),
]

log_collection = create_log_array(args.train_data_dir)
test_log_collection = create_log_array(args.test_data_dir)

print("Training log collection => " + str(len(log_collection['data'])) + " data entries")
print("Testing log collection => " + str(len(test_log_collection['data'])) + " data entries")
print

for algorithm in algorithms:
    model = train(algorithm, log_collection['data'], log_collection['type'])
    success_ratio = predict(model,test_log_collection)
    report((str(algorithm).split('(')[0]),success_ratio)
