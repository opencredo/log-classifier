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

<pre>
$ python2.7 train.py 
Training log collection => 261548 data entries
Testing log collection => 3583 data entries

<b>Naive Bayes</b>
Success rate: 98.66%


<b>SGD Classifier</b>
Success rate: 76.25%


<b>Support Vector Machine</b>
Success rate: 99.3%
</pre>

`predict.py`
- loads training models from `joblib` pickle files
- predicts accuracy of the training models
- takes the following parameters:  
`--test_data_dir` : sets the location of the testing logs (default: `data/test/laptop`)  
`--save-dir` : set location where the joblib pickle files are saved to (default: `save`)  

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

The following algorithms are currently used for classification:
- Naive Bayes Multinomial
- Stochastic Gradient Descent
- Support Vector Machine

