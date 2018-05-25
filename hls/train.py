# -*- coding: utf-8 -*-
import os
import pickle
import sys
import argparse

import numpy as np
import xgboost as xgb
from sklearn.linear_model import Lasso


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type = str, default = './data/data_train.pkl', 
                    help = 'Directory to the training dataset. ')
parser.add_argument('--save_model_dir', type = str, default = './saves/train/models.pkl', 
                    help = 'Directory to save the trained model. Input folder or file name.')
parser.add_argument('--feature_select', action = 'store_true',
                    help = 'Use feature selection. ')

Target_Names = ['LUT', 'FF', 'DSP', 'BRAM']


def load_data(file_name, silence=False):
    if not silence: print "Load data from: ", file_name
    
    # check file exist
    if not os.path.exists(file_name):
        sys.exit("Data file " + file_name + " does not exist!")
    
    # load data
    with open(file_name, "rb") as f:
        data = pickle.load(f)
        
    return data[0], data[1]


def save_models(file_save, models, silence=False):
    # input file name
    file_dir, file_name = os.path.split(file_save)
    if file_dir == '': file_dir = "./saves/train/"
    if file_name == '': file_name = 'models.pkl'
    
    # create folder
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    
    # create file
    with open(os.path.join(file_dir, file_name), "wb") as f:
        pickle.dump(models, f)
        
    if not silence: print "Models are saved to: ", os.path.join(file_dir, file_name)
    
    
def get_params_xgb(targetid=None):
    """
    Get the Tuned parameters for xgboost
    """
    param_defaults = []
    
    # for Target 0
    param_defaults.append({'learning_rate': 0.1,
                           'n_estimators': 600,
                           'max_depth': 5,
                           'min_child_weight': 1,
                           'subsample': 1,
                           'colsample_bytree': 1,
                           'gamma':0 })
    
    # for Target 1
    param_defaults.append({'learning_rate': 0.1,
                           'n_estimators': 600,
                           'max_depth': 5,
                           'min_child_weight': 1,
                           'subsample': 1,
                           'colsample_bytree': 1,
                           'gamma':0 })
    
    # for Target 2
    param_defaults.append({'learning_rate': 0.1,
                           'n_estimators': 600,
                           'max_depth': 5,
                           'min_child_weight': 1,
                           'subsample': 1,
                           'colsample_bytree': 1,
                           'gamma':0 })
    
    # for Target 3
    param_defaults.append({'learning_rate': 0.1,
                           'n_estimators': 600,
                           'max_depth': 5,
                           'min_child_weight': 1,
                           'subsample': 1,
                           'colsample_bytree': 1,
                           'gamma':0 })
    
    # return
    if targetid is not None:
        return param_defaults[targetid]
    else:
        return param_defaults
    
    
def get_params_lasso(targetid=None):
    """
    Get the Tuned parameters for xgboost
    """
    param_defaults = []
    
    # for Target 0
    param_defaults.append({'alpha': 25.0})
    
    # for Target 1
    param_defaults.append({'alpha': 85.0})
    
    # for Target 2
    param_defaults.append({'alpha': 2.0})
    
    # for Target 3
    param_defaults.append({'alpha': 2.0})
    
    # return
    if targetid is not None:
        return param_defaults[targetid]
    else:
        return param_defaults
    
    
def train_models(X, Y, FLAGS, silence=False):
    
    models = []
    for ii in xrange(0, 2): 
        print 'Train model for', Target_Names[ii], '...'
    
        params_xgb = get_params_xgb(ii)
        params_lasso = get_params_lasso(ii)
    
        # fix the random seed
        np.random.seed(seed = 100)
        
        # xgboost - feature selection by xgboost
        if FLAGS.feature_select:
            model_xgb = xgb.XGBRegressor(learning_rate=params_xgb['learning_rate'],
                                         n_estimators=params_xgb['n_estimators'],
                                         max_depth=params_xgb['max_depth'],
                                         min_child_weight=params_xgb['min_child_weight'],
                                         subsample=params_xgb['subsample'],
                                         colsample_bytree=params_xgb['colsample_bytree'],
                                         gamma=params_xgb['gamma'])
            model_xgb.fit(X, Y[:, ii])
        
            b = model_xgb.get_booster()
            feature_weights = [b.get_score(importance_type='weight').get(f, 0.) for f in b.feature_names]
            feature_weights = np.array(feature_weights, dtype=np.float32)
            feature_select  = (feature_weights / feature_weights.sum()) > 0.05
        else:
            feature_select = np.ones(X.shape[1], dtype=np.bool)
        
        # xgboost - train model
        model_xgb = xgb.XGBRegressor(learning_rate=params_xgb['learning_rate'],
                                     n_estimators=params_xgb['n_estimators'],
                                     max_depth=params_xgb['max_depth'],
                                     min_child_weight=params_xgb['min_child_weight'],
                                     subsample=params_xgb['subsample'],
                                     colsample_bytree=params_xgb['colsample_bytree'],
                                     gamma=params_xgb['gamma'])
        model_xgb.fit(X[:, feature_select], Y[:, ii])
        
        # fix the random seed
        np.random.seed(seed = 100)
        
        # lasso
        model_lasso = Lasso(alpha=params_lasso['alpha'])
        model_lasso.fit(X, Y[:, ii])
        
        # add to list
        models.append([model_xgb, model_lasso, feature_select])
    
    # return
    return models
    

if __name__ == '__main__':
    # parser
    FLAGS, unparsed = parser.parse_known_args()
    
    # print info
    print "\n========== Start training models ==========\n"
    
    # load training data
    X, Y = load_data(FLAGS.data_dir)
    
    # train models
    models = train_models(X, Y, FLAGS)
    
    # save models
    save_models(FLAGS.save_model_dir, models)

    print "\n========== End ==========\n"