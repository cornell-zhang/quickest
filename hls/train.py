# -*- coding: utf-8 -*-
# This file is used to train models and save the model into model database.
import os
import pickle
import sys
import argparse

import numpy as np
import xgboost as xgb
import sklearn as sl
from sklearn.linear_model import Lasso

from params import get_defparams


#add examples and default values
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, \
                    default='./data/data_train.pkl', 
                    help='Directory or file of the training dataset. \
                    String. Default: ./data/data_train.pkl')

parser.add_argument('--params_dir', type=str, \
                    default='./saves/train/params.pkl',
                    help='Directory or file to load and save the parameters. \
                    String. Default: ./saves/train/params.pkl')

parser.add_argument('--models_dir', type=str, \
                    default='./saves/train/models.pkl', 
                    help='Directory or file to save the trained model. \
                    String. Default: ./saves/train/models.pkl')

parser.add_argument('-t', '--tune_parameter', action='store_false',
                     help='Whether to tune parameters or not. \
                     Boolean. Default: true')

parser.add_argument('--validation_ratio', type=float, default=0.25,
                     help='The ratio of the training data to do validation. \
                     Float. Default: 0.25')

parser.add_argument('-m', '--model_train', type=str, default='xgb',
                     help='The model to be trained. \
                     Empty means not training models. \
                     Value from "", "xgb"(default), "lasso"')

parser.add_argument('-s', '--model_fsel', type=str, default='lasso',
                     help='The model used to select features. \
                     Empty means not selecting features. \
                     Value from "", "xgb", "lasso"(default)') 

parser.add_argument('-a', '--model_assemble', type=str, default='',
                     help='Strategy used to assemble the trained models. \
                     Empty means not training models. \
                     Value from ""(default), "xgb+lasso+equal_weights", \
                     "xgb+lasso+learn_weights"')


def load_data(FLAGS, silence=False):
    """
    Load training dataset.
    """
    if not silence: print ''
    if not silence: print 'Load data from: ', FLAGS.data_dir
    
    # check file exist
    if not os.path.exists(FLAGS.data_dir):
        sys.exit("Data file " + FLAGS.data_dir + " does not exist!")
    
    # load data
    with open(FLAGS.data_dir, "rb") as f:
        data = pickle.load(f)
    
    # unpack the data
    X = data['x']
    Y = data['y']
    design_index = data['desii']
#    device_index = data['devii']
    feature_name = data['fname']
    target_name = data['tname']
    
    return X, Y, feature_name, target_name, design_index


def load_model_db(FLAGS, silence=False):
    """
    Load model database.
    """
    if not silence: print ''
    if not silence: print 'Load model from: ', FLAGS.models_dir
    
    if not os.path.exists(FLAGS.models_dir):
        return {}
        
    with open(FLAGS.models_dir, "rb") as f:
        model_db = pickle.load(f)  
        
    return model_db


def save_models(name, models, FLAGS, silence=False):
    """
    Save the model to the model database.
    --------------
    Parameters:
        name: The key of the model to be saved.
        models: The model to be saved.    
    """
    if models is None or len(models) == 0:
        return 
    
    # input file name
    file_dir, file_name = os.path.split(FLAGS.models_dir)
    if file_dir == '': file_dir = "./saves/train/"
    if file_name == '': file_name = 'models.pkl'
    file_path = os.path.join(file_dir, file_name)
    
    # create folder
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    
    # load model database
    model_db = load_model_db(FLAGS, silence=True)
    
    # modify model database
    model_db[name] = models
    
    # create file
    pickle.dump(model_db, open(file_path, "w"))
        
    if not silence: print ''
    if not silence: print 'Save Models to: ', file_path
    
    
def save_params(name, params, FLAGS, silence=False):
    """
    Save the parameters to the parameter database.
    --------------
    Parameters:
        name: The key of the parameter to be saved.
        params: The parameter to be saved.    
    """
    if params is None or len(params) == 0:
        return 
    
    # input file name
    file_dir, file_name = os.path.split(FLAGS.params_dir)
    if file_dir == '': file_dir = "./saves/params/"
    if file_name == '': file_name = 'params.pkl'
    file_path = os.path.join(file_dir, file_name)
    
    # create folder
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    
    # load model database
    if os.path.exists(file_path):
        param_db = pickle.load(open(file_path, "r"))
    else:
        param_db = {}
    
    param_db[name] = params
    
    # create file
    pickle.dump(param_db, open(file_path, "w"))
        
    if not silence: print ''
    if not silence: print 'Save Parameters to: ', file_path
    
    
def train_models(X, Y, design_index, FLAGS, silence=False):
    """
    Train model.
    --------------
    Parameters:
        X: Feature dataset.
        Y: Target dataset. 
        design_index: Design id of the dataset. It's used to partition the validation dataset.
    --------------
    Return:
        models: The trained model.
        params: The tuned parameters.
    """
    global Feature_Names, Target_Names
    
    if not silence: print ''
    
    # data to return
    models = {}
    params = {}
    
    # load parameters
    if os.path.exists(FLAGS.params_dir):
        params_db = pickle.load(open(FLAGS.params_dir, 'r'))
        if FLAGS.model_train in params_db.keys():
            params = params_db[FLAGS.model_train] 
    
    for ii in xrange(0, len(Target_Names)): 
        target = Target_Names[ii]
        x = X
        y = Y[:, ii]
        
        if not silence: print ''
        if not silence: print 'For target -', target, '...'
        
        # select features
        if FLAGS.model_fsel != '':
            if not silence: print 'Selecting features by', FLAGS.model_fsel, ' ...'
            
            feature_select = select_features(FLAGS.model_fsel, x, y)
        else:
            feature_select = np.ones(x.shape[1], dtype=bool)
        
        # load parameters
        if target not in params.keys():
            param = get_defparams(FLAGS.model_train)
        else: 
            param = params[target]['param']
            
        # tune parameters
        if FLAGS.tune_parameter:
            if not silence: print 'Tuning parameters for', FLAGS.model_train, '  ...'
            
            # tune param for LASSO
            if FLAGS.model_train == 'lasso':
                param = tune_parameters('lasso', x[:, feature_select], y, design_index, 
                                        param, [{'alpha': t / 100.0} for t in xrange(0, 50, 1)],
                                        valid_ratio=FLAGS.validation_ratio)
            
            # tune param for XGBoost
            if FLAGS.model_train == 'xgb':
                # sequential tuning for 5 iterations
                for tt in xrange(3): # 5
                    param = tune_parameters('xgb', x[:, feature_select], y, design_index, 
                                            param, [{'learning_rate': t / 50.0} for t in xrange(1, 40, 1)],
                                            valid_ratio=FLAGS.validation_ratio)
                    
                    param = tune_parameters('xgb', x[:, feature_select], y, design_index, 
                                            param, [{'n_estimators': t} for t in xrange(2, 100, 2)],
                                            valid_ratio=FLAGS.validation_ratio)
                    
                    param = tune_parameters('xgb', x[:, feature_select], y, design_index, 
                                            param, [{'max_depth': t} for t in xrange(2, 11, 1)],
                                            valid_ratio=FLAGS.validation_ratio)
                    
                    param = tune_parameters('xgb', x[:, feature_select], y, design_index, 
                                            param, [{'min_child_weight': t} for t in xrange(0, 51, 2)],
                                            valid_ratio=FLAGS.validation_ratio)
                    
                    param = tune_parameters('xgb', x[:, feature_select], y, design_index, 
                                            param, [{'subsample': t / 50.0} for t in xrange(10, 51, 1)],
                                            valid_ratio=FLAGS.validation_ratio)
                    
                    param = tune_parameters('xgb', x[:, feature_select], y, design_index, 
                                            param, [{'colsample_bytree': t / 50.0} for t in xrange(10, 51, 1)],
                                            valid_ratio=FLAGS.validation_ratio)
                    
                    param = tune_parameters('xgb', x[:, feature_select], y, design_index, 
                                            param, [{'gamma': t / 100.0} for t in xrange(0, 101, 2)],
                                            valid_ratio=FLAGS.validation_ratio)
                    
                    param['learning_rate'] = param['learning_rate'] / 2.0
                    param['n_estimators'] = param['n_estimators'] * 3
            
        # train model
        if not silence: print 'Training ...'
        if not silence: print 'Parameters for ' + FLAGS.model_train + ':', param
            
        # train model - xgboost
        if FLAGS.model_train == 'xgb':
            np.random.seed(seed = 100)
            model = xgb.XGBRegressor(learning_rate=param['learning_rate'],
                                     n_estimators=param['n_estimators'],
                                     max_depth=param['max_depth'],
                                     min_child_weight=param['min_child_weight'],
                                     subsample=param['subsample'],
                                     colsample_bytree=param['colsample_bytree'],
                                     gamma=param['gamma'])
            model.fit(x[:, feature_select], y)
            
        # train model - lasso
        elif FLAGS.model_train == 'lasso':
            np.random.seed(seed = 100)
            model = Lasso(alpha=param['alpha'])
            model.fit(x[:, feature_select], y)
        
                
        # add to list to save
        params[target] = {'param': param }
        
        # add to list to return
        models[target] = {'model': model, 
                          'fselect': feature_select,
                          'fnames': np.array(Feature_Names)[feature_select].tolist()}
    
    # return
    return models, params


def assemble_models(X, Y, model_db, FLAGS, silence=False):
    """
    Assemble model. Assemble 2 or more models in the model database to be one model.
    --------------
    Parameters:
        X: Feature dataset.
        Y: Target dataset. 
        model_db: Model database.
    """
    global Feature_Names, Target_Names
    
    if not silence: print ''
    
    # train model
    if not silence: print 'Assembling', FLAGS.model_assemble, ' ...'
        
    # data to return
    models = {}
    for ii in xrange(0, len(Target_Names)):
        target = Target_Names[ii]
        x = X
        y = Y[:, ii]
        
        if FLAGS.model_assemble == 'xgb+lasso+equal_weights':
            models[target] = {'models': [model_db['xgb'][target], model_db['lasso'][target]],
                              'weights': [0.5, 0.5]}
            
        elif FLAGS.model_assemble == 'xgb+lasso+learn_weights':
            # predict
            model = model_db['xgb'][target]['model']
            features = model_db['xgb'][target]['fselect']
            predict0 = model.predict(x[:, features])
            
            # predict
            model = model_db['lasso'][target]['model']
            features = model_db['lasso'][target]['fselect']
            predict1 = model.predict(x[:, features])
            
            # weight model
            model_weights = Lasso(alpha=0.02)
            model_weights.fit(np.array([predict0, predict1]).T, y)
        
            models[target] = {'models': [model_db['xgb'][target], model_db['lasso'][target]],
                              'weights': model_weights}
    # return
    return models


def select_features(strategy, X, Y, silence=False):
    """
    Select features.
    --------------
    Parameters:
        X: Feature dataset.
        Y: Target dataset. 
        strategy: Strategy used to select features.
            xgb - Select by XGBoost
            lasso - Select by LASSO
    --------------
    Return:
        feature_select: Feature selecting result. It's a boolean array. True means the feature is selected.
    """
    # fix the random seed
    np.random.seed(seed = 100)
    
    if strategy == 'xgb':
        model = xgb.XGBRegressor()
        model.fit(X, Y)
        
        b = model.get_booster()
        feature_weights = [b.get_score(importance_type='weight').get(f, 0.) for f in b.feature_names]
        feature_weights = np.array(feature_weights, dtype=np.float32)
        feature_select  = (feature_weights / feature_weights.sum()) > 0.03
        
    elif strategy == 'lasso':
        model = Lasso(alpha=0.01)
        model.fit(X, Y)
        feature_select = model.coef_ != 0
    
    return feature_select


def tune_parameters(model_name, X, Y, design_index, def_param, tune_params, 
                    valid_ratio=0.25, silence=False):
    """
    Tune parameters.
    --------------
    Parameters:
        X: Feature dataset.
        Y: Target dataset. 
        model_name: The model to tune parameters for.
        design_index: Design id of the dataset. It's used to partition the validation dataset.
        def_param: The default parameters before tuning. It's a dict.
        tune_params: The parameters list where the best parameter to be chosen from.
        valid_ratio: Ratio of the validation data in all the training data.
    --------------
    Return:
        param: The tuned parameters. It's a dict.
    """
    # fix the random seed
    np.random.seed(seed = 100)
    
    # tune the parameter of LASSO
    params = []
    scores = []
    
    # construct parameters
    if len(tune_params) == 0:
        params = [def_param]
    else:
        for tune_param in tune_params:
            param = dict(def_param)
            for key in tune_param.keys():
                param[key] = tune_param[key]
                params.append(param)
    
    # validation group
    ids = np.unique(design_index)
    vld_ids = []
    for ii in xrange(int(1 / valid_ratio)):
        start = int(len(ids) * ii * valid_ratio)
        end = int(len(ids) * (ii + 1) * valid_ratio)
        vld_ids.append(ids[start: end])
        
    # tune parameters
    for param in params:
        # get valid ids
        Y_pre = []
        Y_vld = []
        
        # cross validation
        for vld_id in vld_ids:
            # partition
            xt = X[[x not in vld_id for x in design_index]]
            yt = Y[[x not in vld_id for x in design_index]]
            xv = X[[x in vld_id for x in design_index]]
            yv = Y[[x in vld_id for x in design_index]]
            
            # init
            if model_name == 'lasso':
                model = Lasso(alpha=param['alpha'])
                
            elif model_name == 'xgb':
                model = xgb.XGBRegressor(learning_rate=param['learning_rate'],
                                         n_estimators=param['n_estimators'],
                                         max_depth=param['max_depth'],
                                         min_child_weight=param['min_child_weight'],
                                         subsample=param['subsample'],
                                         colsample_bytree=param['colsample_bytree'],
                                         gamma=param['gamma'])
            else:
                break
                
            # train
            model.fit(xt, yt)
            
            # predict
            yp = model.predict(xv)
            
            # add to list
            Y_vld.extend(yv)
            Y_pre.extend(yp)
            
        # score
        score = sl.metrics.r2_score(Y_vld, Y_pre)
        
        # add to list
        scores.append(score)
        
        if not silence: print model_name, ' -', param, 'R2 Score =', score
    
    # get tuned parameters
    param = params[np.argmax(scores)]
    
    # return 
    return param
    

if __name__ == '__main__':
    global Feature_Names, Target_Names
    
    # parser
    FLAGS, unparsed = parser.parse_known_args()
    
    # print info
    print "\n========== Start training models ==========\n"
    
    # load training data
    X_train, Y_train, Feature_Names, Target_Names, design_index = load_data(FLAGS)
    
    if FLAGS.model_train != '':
        # train models
        models, params = train_models(X_train, Y_train, design_index, FLAGS)
        
        # save models
        save_models(FLAGS.model_train, models, FLAGS)
        
        # save params
        save_params(FLAGS.model_train, params, FLAGS)
    
    if FLAGS.model_assemble != '':
        # load model database
        model_db = load_model_db(FLAGS, silence=True)
        
        # assemble models 
        models = assemble_models(X_train, Y_train, model_db, FLAGS)
    
        # save models
        save_models(FLAGS.model_assemble, models, FLAGS) # xgb+lasso+equal_weight

    print "\n========== End ==========\n"
