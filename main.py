#! /usr/bin/env python

import numpy as np
import pandas as pd
import csv
import logging as log
from sklearn.cross_validation import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import log_loss
from sklearn.base import TransformerMixin

FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
log.basicConfig(format=FORMAT, level=log.DEBUG)

class DenseTransformer(TransformerMixin):

    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self

def load_train_data(path):
    df = pd.read_csv(path)
    X = df.values.copy()
    np.random.shuffle(X)
    X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, encoder, scaler


def load_csv_file(csv_file, cut_end=True):
    X_train = []
    Y_train = []
    with open(csv_file) as file:
        next(file)
        test_reader = csv.reader(file)
        for row in test_reader:
            if cut_end:
                X_train.append(row[1:-1])
            else:
                X_train.append(row[1:])
            Y_train.append(row[-1])
    return (X_train, Y_train)


def write_out_submission(probas, filename):
    out_header = "id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9"
    index = 1
    log.info('Writing out submission file.')
    with open(filename, 'w') as outfile:
        outfile.write('%s\n' % out_header)
        for row in probas:
            strings = ["%.4f" % w for w in row]
            outfile.write("%d,%s\n" % (index, ','.join(strings)))
            index += 1


def main():
    X, Y, encoder, scale = load_train_data('train.csv')
    estimators = 500
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2, random_state=0)
    X_train_real, X_test_real, Y_train_real, Y_test_real = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
    log.info('Loaded training file')
    X_test, _ = load_csv_file('test.csv', cut_end=False)
    log.info('Loaded test file')

    #Classifier Setup
    tree_clf = ExtraTreesClassifier(n_estimators=estimators, n_jobs=-1,
                                    random_state=42, max_depth=55, min_samples_split=1)

    clf = make_pipeline(TfidfTransformer(), DenseTransformer(), tree_clf)
    log.info('Fitting GradientBoost')
    clf.fit(X_train_real, Y_train_real)
    clf_probs = clf.predict_proba(X_test_real)
    score = log_loss(Y_test_real, clf_probs)
    log.info('Log Loss score un-trained = %f' % score)
    # Calibrate Classifier using ground truth in X,Y_valid
    sig_clf = CalibratedClassifierCV(clf, method="isotonic", cv="prefit")
    log.info('Fitting CalibratedClassifierCV')
    sig_clf.fit(X_valid, Y_valid)
    sig_clf_probs = sig_clf.predict_proba(X_test_real)
    sig_score = log_loss(Y_test_real, sig_clf_probs)
    log.info('Log loss score trained = %f' % sig_score)

    # Ok lets predict the test data with our funky new classifier
    sig_submission_probs = sig_clf.predict_proba(X_test)

    write_out_submission(sig_submission_probs, 'submission.csv')

if __name__ == '__main__':
    main()
