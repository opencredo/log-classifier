#
# Purpose: Classify logs wrt their source (eg. Java, Apache, Nagios, etc.)
# Requires: sklearn, numpy, argparse
#

import glob
import argparse
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.externals import joblib

parser = argparse.ArgumentParser()
parser.add_argument('--test_data_dir', type=str, default='data/test/laptop',
                    help='data directory containing training logs')
parser.add_argument('--save_dir', type=str, default='save',
                    help='directory to store training models')
args = parser.parse_args()

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

def report(clf_type,ratio):
    print("\033[1m" + clf_type + "\033[0m\033[92m")
    print("Success rate: " + str(round(ratio * 100,2)) + "%\n")
    print

test_log_collection = create_log_dict(args.test_data_dir)

print("Testing log collection => " + str(len(test_log_collection['data'])) + " data entries")
print

for saved_file in glob.glob(args.save_dir + '/*.pkl'):
    model = joblib.load(saved_file)
    accuracy = predict(model,test_log_collection)
    report((str(saved_file).split('/')[1].split('.')[0]),accuracy)
