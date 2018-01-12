# log-classifier

## Scripts:

`train.py`

- trains on a set of training logs using various algorithms
- saves training models to `joblib` pickle files
- predicts accuracy of the training models
- takes the following parameters:  
`--train_data_dir` : sets the location of the training logs (default: `data/train/laptop`)  
`--test_data_dir` : sets the location of the testing logs (default: `data/test/laptop`)  
`--save-dir` : set location where the joblib pickle files are saved to (default: `save`)

### Install libraries

Make sure you have a recent version of python2.7 and python pip, then install the required libraries.

<pre>
pip install numpy sklearn
<pre>

### Collect logs

Create data directories.

<pre>
mkdir -p data/{train,test}/laptop
<pre>

Collect logs

<pre>
find /var/log -type f -size +10k -name "*.log" 2>/dev/null | while read log
do
  rows=$(wc -l "$log" | awk '{ print $1 }')
  head -$(($rows - ($rows / 10))) "$log" > data/train/laptop/"${log##*/}"
  tail -$(($rows / 10)) "$log" > data/test/laptop/"${log##*/}"
done
<pre>

### Run script

Run the script

<pre>
python2.7 train.py
<pre>

This should give something like the following:

<pre>
Training log collection => 250587 data entries
Testing log collection => 27843 data entries

<b>SGDClassifier</b>
Success rate: 97.38%


<b>MultinomialNB</b>
Success rate: 98.64%


<b>BernoulliNB</b>
Success rate: 96.36%


<b>DecisionTreeClassifier</b>
Success rate: 95.26%


<b>ExtraTreeClassifier</b>
Success rate: 94.52%


<b>ExtraTreesClassifier</b>
Success rate: 99.21%


<b>LinearSVC</b>
Success rate: 99.17%


<b>NearestCentroid</b>
Success rate: 92.29%


<b>RandomForestClassifier</b>
Success rate: 99.06%


<b>RidgeClassifier</b>
Success rate: 99.16%
</pre>

`predict.py`
- loads training models from `joblib` pickle files
- predicts accuracy of the training models
- takes the following parameters:  
`--test_data_dir` : sets the location of the testing logs (default: `data/test/laptop`)  
`--save-dir` : set location where the joblib pickle files are saved to (default: `save`)  

<pre>
$ python2.7 predict.py
Testing log collection => 27843 data entries

<b>SGDClassifier</b>
Success rate: 97.38%


<b>MultinomialNB</b>
Success rate: 98.64%


<b>BernoulliNB</b>
Success rate: 96.36%


<b>DecisionTreeClassifier</b>
Success rate: 95.26%


<b>ExtraTreeClassifier</b>
Success rate: 94.52%


<b>ExtraTreesClassifier</b>
Success rate: 99.21%


<b>LinearSVC</b>
Success rate: 99.17%


<b>NearestCentroid</b>
Success rate: 92.29%


<b>RandomForestClassifier</b>
Success rate: 99.06%


<b>RidgeClassifier</b>
Success rate: 99.16%
</pre>
<pre>
$ python2.7 predict.py 
Testing log collection => 3583 data entries

<b>Naive Bayes</b>
Success rate: 98.66%


<b>SGD Classifier</b>
Success rate: 76.25%


<b>Support Vector Machine</b>
Success rate: 99.3%
</pre>

## Algorithms

You can adjust the `algorithms` array to include any number of Scikit Learn algorithms that you want to run:

<pre>
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
</pre>


