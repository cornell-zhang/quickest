# -*- coding: utf-8 -*-
import os
import pickle
import sys
import argparse

import numpy as np
from sklearn import metrics


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type = str, default = './data/data_test.pkl', 
                    help = 'Directory to the testing dataset. ')
parser.add_argument('--model_dir', type = str, default = './saves/train/models.pkl', 
                    help = 'Directory to the pre-trained models. ')
parser.add_argument('--save_result_dir', type = str, default = './saves/test/results.csv', 
                    help = 'Directory to save the result. Input folder or file name.')


def score_REA(Y, Y_pre):
    """
    Score by the true and predicted Y
    """
    Y_pre = np.array(Y_pre)
    Y = np.array(Y)
    
    Y_mean = np.mean(Y)
    error = Y_pre - Y
    REA = np.mean(np.abs(error)) / (np.mean(np.abs(Y - Y_mean)) + np.finfo(float).eps)
    
    return REA


def load_data(file_name, silence=False):
    if not silence: print "Load data from: ", file_name
    
    if not os.path.exists(file_name):
        sys.exit("Data file " + file_name + " does not exist!")
        
    with open(file_name, "rb") as f:
        data = pickle.load(f)  
        
    return data[0], data[1]


def load_model(file_name, silence=False):
    if not silence: print "Load model from: ", file_name
    
    if not os.path.exists(file_name):
        sys.exit("Model file " + file_name + " does not exist!")
        
    with open(file_name, "rb") as f:
        models = pickle.load(f)  
        
    return models


def save_results(file_save, results, silence=False):
    if not os.path.exists("./saves/test/"):
        os.mkdir("./saves/test/")
        
    # input file name
    file_dir, file_name = os.path.split(file_save)
    if file_dir == '': file_dir = "./saves/train/"
    if file_name == '': file_name = 'models.pkl'
    
    # create folder
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
        
    # create file
    np.savetxt(os.path.join(file_dir, file_name), results, delimiter=',', header='LUT, FF', comments='')
        
    if not silence: print "Results are saved to: ", os.path.join(file_dir, file_name)


def test_models(models, X, Y, silence=False):
    # test model
    results = np.zeros([X.shape[0], len(models)], dtype=np.float)
    scores_RAE = np.zeros([len(models)], dtype=np.float)
    scores_R2 = np.zeros([len(models)], dtype=np.float)
    for ii in xrange(len(models)):
        # load models
        model_xgb = models[ii][0]
        model_lasso = models[ii][1]
        features = models[ii][2]
        
        # predict
        results0 = model_xgb.predict(X[:, features])
        results1 = model_lasso.predict(X)
        results[:, ii] = (results0 + results1) / 2.0
        
        scores_RAE[ii] = score_REA(Y[:, ii], results[:, ii])
        scores_R2[ii] =  metrics.r2_score(Y[:, ii], results[:, ii])
    
    # print scores
    if not silence: print "RAE of the results are: ", scores_RAE
    if not silence: print "R2 of the results are: ", scores_R2
    
    # return 
    return results
    

if __name__ == '__main__':
    # parser
    FLAGS, unparsed = parser.parse_known_args()
    
    # print info
    print "\n========== Start testing models ==========\n"
    
    # load testing data
    X, Y = load_data(FLAGS.data_dir)
    
    # load model
    models = load_model(FLAGS.model_dir)
    
    # train models
    results = test_models(models, X, Y)
    
    # save results
    save_results(FLAGS.save_result_dir, results)
    
    print "\n========== End ==========\n"
