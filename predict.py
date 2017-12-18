#
# Purpose: Classify logs wrt their source (eg. Java, Apache, Nagios, etc.)
# Requires: sklearn, numpy, argparse
# 

import glob
import timeit
import argparse
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.externals import joblib

parser = argparse.ArgumentParser()
parser.add_argument('--test_data_dir', type=str, default='data/test/laptop',
                    help='data directory containing training logs')
parser.add_argument('--save_dir', type=str, default='save',
                    help='directory to store training pipeline models')
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

count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()

test_log_collection = create_log_array('data/test/mac')

print("Testing log collection => " + str(len(test_log_collection['data'])) + " data entries")
print

clf = joblib.load(args.save_dir + "/mnb.pkl")
success_ratio = classify(clf,test_log_collection)
display_results("Naive Bayes",success_ratio)

clf = joblib.load(args.save_dir + "/sgd.pkl")
success_ratio = classify(clf,test_log_collection)
display_results("SGD Classifier - 5 iterations",success_ratio)

clf = joblib.load(args.save_dir + "/svm.pkl")
success_ratio = classify(clf,test_log_collection)
display_results("Support Vector Machine",success_ratio)
