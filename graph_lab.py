#! /usr/bin/env python

import graphlab as gl
import math
import random
gl.product_key.set_product_key('CC2F-572A-E2BD-0005-17F0-75C6-6890-AEC3')

def make_submission(m, test, filename):
    preds = m.predict_topk(test, output_type='probability', k=9)
    preds['id'] = preds['id'].astype(int) + 1
    preds = preds.unstack(['class', 'probability'], 'probs').unpack('probs', '')
    preds = preds.sort('id')
    preds.save(filename)

def multiclass_logloss(model, test):
    preds = model.predict_topk(test, output_type='probability', k=9)
    preds = preds.unstack(['class', 'probability'], 'probs').unpack('probs', '')
    preds['id'] = preds['id'].astype(int) + 1
    preds = preds.sort('id')
    preds['target'] = test['target']
    neg_log_loss = 0
    for row in preds:
        label = row['target']
        neg_log_loss += - math.log(row[label])
    return  neg_log_loss / preds.num_rows()

def evaluate_logloss(model, train, valid):
    return {'train_logloss': multiclass_logloss(model, train),
            'valid_logloss': multiclass_logloss(model, valid)}

def main():
    data =  gl.SFrame('train.csv')
    test = gl.SFrame.read_csv('data/test.csv')
    data.remove_column('id')
    train_data, test_data = data.random_split(0.8)
    params = {'target': 'target',
          'max_iterations': 500,
          'max_depth': 16,
          'min_child_weight': 15,
    }
    
    model = gl.boosted_trees_classifier.create(train_data, target='target',
                                                max_iterations=500, max_depth=16,
                                                min_child_weight=15)
    predictions = model.classify(test_data)
    print evaluate_logloss(model, train_data, test_data)

    # Make final submission by using full training set
    m = gl.boosted_trees_classifier.create(train, **params)
    make_submission(m, test, 'submission.csv')

if __name__=="__main__":
    main()
