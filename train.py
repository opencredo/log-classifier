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
from sklearn import svm
from sklearn.externals import joblib

parser = argparse.ArgumentParser()
parser.add_argument('--train_data_dir', type=str, default='data/train/laptop',
                    help='data directory containing training logs')
parser.add_argument('--test_data_dir', type=str, default='data/test/laptop',
                    help='data directory containing training logs')
parser.add_argument('--save_dir', type=str, default='save',
                    help='directory to store and load training pipeline models')
args = parser.parse_args()

def classify(clf, new_docs):
    predicted = clf.predict(new_docs['data'])
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

def display_results(clf_type,ratio):
    print("\033[1m" + clf_type + "\033[0m\033[92m")
    print("Success rate: " + str(round(ratio * 100,2)) + "%\n")
    print

log_collection = create_log_array(args.train_data_dir)
test_log_collection = create_log_array(args.test_data_dir)

print("Training log collection => " + str(len(log_collection['data'])) + " data entries")
print("Testing log collection => " + str(len(test_log_collection['data'])) + " data entries")
print

mnb_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
mnb_clf.fit(log_collection['data'],log_collection['type'])
joblib.dump(mnb_clf, args.save_dir + "/mnb.pkl")
success_ratio = classify(mnb_clf,test_log_collection)
display_results("Naive Bayes",success_ratio)

sgd_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf',SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None))])
sgd_clf.fit(log_collection['data'],log_collection['type'])
joblib.dump(sgd_clf, args.save_dir + "/sgd.pkl")
success_ratio = classify(sgd_clf,test_log_collection)
display_results("SGD Classifier - 5 iterations",success_ratio)

svm_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', svm.SVC(kernel='linear', C = 1.0))])
svm_clf.fit(log_collection['data'],log_collection['type'])
joblib.dump(svm_clf, args.save_dir + "/svm.pkl")
success_ratio = classify(svm_clf,test_log_collection)
display_results("Support Vector Machine",success_ratio)
