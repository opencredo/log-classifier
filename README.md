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

```
$ python2.7 train.py 
Training log collection => 261548 data entries
Testing log collection => 3583 data entries

**Naive Bayes**
Success rate: 98.66%


**SGD Classifier**
Success rate: 76.25%


**Support Vector Machine**
Success rate: 99.3%
```

`predict.py`
- loads training models from `joblib` pickle files
- predicts accuracy of the training models
- takes the following parameters:  
`--test_data_dir` : sets the location of the testing logs (default: `data/test/laptop`)  
`--save-dir` : set location where the joblib pickle files are saved to (default: `save`)  

```
$ python2.7 predict.py 
Testing log collection => 3583 data entries

**Naive Bayes**
Success rate: 98.66%


**SGD Classifier**
Success rate: 76.25%


**Support Vector Machine**
Success rate: 99.3%
```

## Algorithms

The following algorithms are currently used for classification:
- Naive Bayes Multinomial
- Stochastic Gradient Descent
- Support Vector Machine

