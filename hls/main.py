# basics
import os
import sys
import numpy as np

# train_test_split
from sklearn.model_selection import train_test_split

# L1 feature selection
from sklearn.feature_selection import SelectFromModel

# xgboost
import xgboost as xgb
# Lasso
from sklearn import linear_model
# ANN regressor
from sklearn.neural_network import MLPRegressor
# ANN classifier
from sklearn.neural_network import MLPClassifier

# for L2 norm
from numpy import linalg as LA

# pickle
import cPickle as pickle

# csv writter
import csv

# parser
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type = str, default = './data/', 
                    help = 'Directory to the input dataset. ')
parser.add_argument('--predict_area', action = 'store_true',
                    help = 'Run area estimation. ')
parser.add_argument('--classify_timing', action = 'store_true',
                    help = 'Run timing classification. ')
parser.add_argument('--train', action = 'store_true',
                    help = 'Run training for the specified task. ')
parser.add_argument('--store_model_path', type = str, default = './model.pkl',
                    help = 'Path to store the trained models')
parser.add_argument('--test', action = 'store_true',
                    help = 'Run test for the specified task. Should use with the \'--pretrained_model_path\' option. ')
parser.add_argument('--pretrained_model_path', type = str, default = './model.pkl',
                    help = 'Path to pretrained models. The models should be dumped in pickle files. ')
parser.add_argument('--feature_select', action = 'store_true',
                    help = 'Use feature selection. If this option is used with \'--train\', the generated model would contain a mask indicating the selected features. If used with \'--test\', then the produced model must provide the mask as well. ')
#================================ Preprocessing =================================#

# l1-based feature selection
# feed in training set, selects a set of features, and select the same features from the test set
def linearFeatureSelection(train_X, train_Y, test_X, feature_name, task = 'regression', alpha = 0.001, threshold = 0.01):

  print "\n---- Linear Feature Selection ----\n"

  if task == 'regression':
    # use Lasso for a regression task
    clf = linear_model.Lasso(alpha=alpha)
  elif task == 'classification':
    # use sgd for a classfication task (timing)
    clf = linear_model.SGDClassifier()
  # train it with the training data we have
  clf.fit(train_X, train_Y)
  # do feature selection
  model = SelectFromModel(clf, prefit=True, threshold=threshold)
  # get the mask of selected features, and select the test set
  # also get the names of selected features
  selected_mask = model.get_support(indices=True)
  new_test_X = test_X[:, selected_mask]
  selected_feature_names = feature_name[selected_mask]

  # uncomment this if you want to see which features are selected
  """
  for i in range(len(selected_mask)):
    # print out the name and the weight for that feature
    if train_Y.shape[1] == 1:
      print "selected feature ", feature_name[selected_mask[i]], ", weight = ", clf.coef_[selected_mask[i]]
    elif train_Y.shape[1] > 1:
      print "selected feature ", feature_name[selected_mask[i]], ", weight = ", LA.norm(clf.coef_[:, selected_mask[i]])
  """

  # return the new set of features for both training and test sets
  return model.transform(train_X), new_test_X, selected_mask

#================================ Normalize data =================================#

# normalize the input dataset X, normalize across columns
# return the normalized dataset, and the mean/std of each column
def normalize_data(X):

  normDataX = np.zeros(X.shape)    
  normVal  = []
  temp     = {}

  # normalize the features, normalize by column
  for i in range(X.shape[1]):
    temp['mean'] = X[:,i].mean()
    temp['std']  = X[:,i].std() + 1e-6
    normDataX[:,i] = (X[:,i] - temp['mean']) / temp['std']
    normVal.append(temp.copy())

  return normDataX, normVal


#========== Model training and testing functions ==========#

# supported models: lasso, xgb, multi-level perceptron regressor (MLP)
def model_training(X, Ys, mode = 'single', model_name = 'lasso', 
                   alpha=0.001, xgb_depth=3, xgb_trees=100, nn_solver='lbfgs', nn_size=(30,30)):

  clf_models = []

  # single-task: train one model for predicting each resource
  if mode == 'single':    
    for i in range(Ys.shape[1]):
      Y = Ys[:,i]
      clf = None
      if model_name == 'lasso':
          clf = linear_model.Lasso(alpha=alpha)
      elif model_name == 'xgb':
          clf = xgb.XGBRegressor(max_depth=xgb_depth, n_estimators=xgb_trees)
      elif model_name == 'ann':
          clf = MLPRegressor(solver=nn_solver, hidden_layer_sizes=nn_size, random_state=0)
            
      clf.fit(X, Y)
      clf_models.append(clf)
  # multi-task: train one model for predicting all resources
  elif mode == 'multi':
    if model_name == 'lasso':
        clf = linear_model.MultiTaskLasso(alpha=alpha)
    elif model_name == 'ann':
        clf = MLPRegressor(solver=nn_solver, hidden_layer_sizes=nn_size, random_state=0)
        
    clf.fit(X, Ys)
    clf_models.append(clf)

  # return the trained model(s)
  return clf_models

# test the models
def model_testing(clf_models, X, Ys, normVal, Y_normalized = True):

  RAE_res = []  
  norm_Y_pres = np.zeros(Ys.shape)

  # single-task: the clf_models array will contain multiple models
  if len(clf_models) > 1:
    # test the model for each Y
    for i in range(Ys.shape[1]):
      clf   = clf_models[i]
      norm_Y_pres[:, i] = clf.predict(X) 
  # multi-task: clf_models is one model
  elif len(clf_models) == 1:
    # directly predict
    norm_Y_pres = clf_models[0].predict(X) 

  # recover the predictions to the actual scale
  Y_pres = np.zeros(Ys.shape)
  for i in range(Ys.shape[1]):
    Y_pres[:, i] = norm_Y_pres[:, i] * normVal[i]['std'] + normVal[i]['mean']

  # recover the normalized ground truth if necessary
  if Y_normalized == True:
    new_Ys = np.zeros(Ys.shape)
    for i in range(Ys.shape[1]):
      new_Ys[:, i] = Ys[:, i] * normVal[i]['std'] + normVal[i]['mean']
    Ys = new_Ys
 
  # compute the overall error metric
  for i in range(Ys.shape[1]):
    Y = Ys[:, i]
    Y_pre = Y_pres[:, i]
    Y_mean = np.mean(Y)

    # relative absolute error
    RAE = np.sum(np.abs(Y_pre - Y)) / (np.sum(np.abs(Y - Y_mean)) + np.finfo(float).eps)

    RAE_res.append(RAE)

  return np.asarray(RAE_res)

#========== Main function for parameter sweeping ==========#

def sweep_params(hyperparam_list, train_val_pairs, data_pack, model_name):

  print "\n---- Sweeping", model_name, "Parameters,", len(train_val_pairs['train_X']), "train-val pairs  ----\n"

  # shortcut for array names
  train_Xs = train_val_pairs['train_X']           # normalized
  train_Ys = train_val_pairs['train_Y']           # normalized
  val_Xs = train_val_pairs['val_X']               # normalized
  val_Ys = train_val_pairs['val_Y']               # normalized
  train_and_val_X = data_pack['training_X']       # normalized
  train_and_val_Y = data_pack['norm_training_Y']  # normalized
  test_X = data_pack['test_X']                    # normalized
  test_Y = data_pack['test_Y']                    # NOT NORMALIZED!!!
  normVal = data_pack['normVal']

  # initialize
  best_RAE = np.full(train_Ys[0].shape[1], 1e10)
  best_RAE_param = [None] * train_Ys[0].shape[1]

  # this loop goes over different parameter settings
  for hyperparam in hyperparam_list:
    average_RAE = np.zeros((train_and_val_Y.shape[1]))

    # this loop runs over all training-validation pairs
    for train_val_pair_id in range(len(train_Xs)):
      # select the training/validation pair
      train_X = train_Xs[train_val_pair_id]
      train_Y = train_Ys[train_val_pair_id]
      val_X = val_Xs[train_val_pair_id]
      val_Y = val_Ys[train_val_pair_id]

      # run training
      # call different functions for different models

      # single-task ANN
      if model_name == 'ann':
        clf_models = model_training(train_X, train_Y, mode = 'single', model_name = 'ann', nn_size = hyperparam)
      # XGBoost
      elif model_name == 'xgb':
        clf_models = model_training(train_X, train_Y, mode = 'single', model_name = 'xgb', 
                                    xgb_depth = hyperparam[0], xgb_trees = hyperparam[1])
      # single-task LASSO
      elif model_name == 'lasso':
        clf_models = model_training(train_X, train_Y, mode = 'single', model_name = 'lasso', alpha=hyperparam)
      # multi-task ANN
      elif model_name == 'multiann':
        clf_models = model_training(train_X, train_Y, mode = 'multi', model_name = 'ann', nn_size = hyperparam)
      # multi-task LASSO
      elif model_name == 'multilasso':
        clf_models = model_training(train_X, train_Y, mode = 'multi', model_name = 'lasso', alpha=hyperparam)

      # validate
      val_RAE = model_testing(clf_models, val_X, val_Y, normVal['training_Y'], Y_normalized = True)

      average_RAE = np.add(average_RAE, val_RAE)

    # compute average error across the validation sets
    average_RAE = np.divide(average_RAE, len(train_Xs))

    print "Parameter", hyperparam, ": val error", average_RAE, "across", len(train_Xs), "val runs"

    # select the best hyperparameter combination

    # single-task models: simple value comparison
    if model_name == 'ann' or model_name == 'xgb' or model_name == 'lasso':
      for i in range(train_Ys[0].shape[1]):
        if average_RAE[i] < best_RAE[i]:
          best_RAE[i] = average_RAE[i]
          best_RAE_param[i] = hyperparam

    # multi-task models: compare the norm of the error vector
    elif model_name == 'multiann' or model_name == 'multilasso':
      average_RAE_norm = LA.norm(average_RAE)
      if average_RAE_norm < LA.norm(best_RAE):
        best_RAE = average_RAE
        for i in range(train_Ys[0].shape[1]):
          best_RAE_param[i] = hyperparam
    
  # use the best hyperparameters to run through the test set
  clf_models_RAE = []

  # retrain the models with all the data except for the test set
  # single-task ANN
  if model_name == 'ann':
    for i in range(train_and_val_Y.shape[1]):
      Y = train_and_val_Y[:, i:i+1]
      model_RAE = model_training(train_and_val_X, Y, mode = 'single', model_name = 'ann', nn_size = best_RAE_param[i])
      clf_models_RAE.append(model_RAE[0])
      print "Best parameter for task ", i, ":", best_RAE_param[i]
  # XGBoost
  elif model_name == 'xgb':
    for i in range(train_and_val_Y.shape[1]):
      Y = train_and_val_Y[:, i:i+1]
      model_RAE = model_training(train_and_val_X, Y, mode = 'single', model_name = 'xgb', 
                                 xgb_depth = best_RAE_param[i][0], xgb_trees = best_RAE_param[i][1])
      clf_models_RAE.append(model_RAE[0])
      print "Best parameter for task ", i, ":", best_RAE_param[i]
  # single-task LASSO
  elif model_name == 'lasso':
    for i in range(train_and_val_Y.shape[1]):
      Y = train_and_val_Y[:, i:i+1]
      model_RAE = model_training(train_and_val_X, Y, mode = 'single', model_name = 'lasso', alpha=best_RAE_param[i])
      clf_models_RAE.append(model_RAE[0])
      print "Best parameter for task ", i, ":", best_RAE_param[i]
  # multi-task ANN
  elif model_name == 'multiann':
    clf_models_RAE = model_training(train_and_val_X, train_and_val_Y, mode = 'multi', 
                                    model_name = 'ann', nn_size = best_RAE_param[0])
    print "Best parameter: ",best_RAE_param[0]
  # multi-task LASSO
  elif model_name == 'multilasso':
    clf_models_RAE = model_training(train_and_val_X, train_and_val_Y, mode = 'multi', model_name = 'lasso', alpha=best_RAE_param[0])
    print "Best parameter: ", best_RAE_param[0]

  return clf_models_RAE

#========== Utility functions ==========#

# generate ANN parameter list
def gen_ANN_param_list(max_size_1D = 200, max_size_2D = 50):
  ann_sizes = []
  for i in range(max_size_1D/10):
    ann_sizes.append(10*(i+1))
  for i in range(max_size_2D/10):
    for j in range(max_size_2D/10):
      ann_sizes.append((10*(i+1), 10*(j+1)))

  return ann_sizes

# generate XGBoost parameter list
def gen_XGB_param_list(max_depth = 3, max_trees = 500):
  xgb_params = []
  for i in range(max_depth):
    for j in range(max_trees/100):
      xgb_params.append((i+1, (j+1)*100))

  return xgb_params

# generate training-validation splits
def train_val_split(X, Y, rounds = 10, val_size = 0.25):

  print "\n---- Training / Validation Split ----\n"

  train_Xs = []
  train_Ys = []
  val_Xs = []
  val_Ys = []
  for n_train_val_pairs in range(rounds):
    train_X, val_X, train_Y, val_Y = train_test_split(X, Y, test_size = val_size)
    print "train ", n_train_val_pairs, ": X", train_X.shape, "Y", train_Y.shape, "val: X", val_X.shape, "Y", val_Y.shape
    train_Xs.append(train_X)
    train_Ys.append(train_Y)
    val_Xs.append(val_X)
    val_Ys.append(val_Y)

  # pack into a dict
  pack = {'train_X': train_Xs, 'train_Y': train_Ys, 'val_X': val_Xs, 'val_Y': val_Ys}

  return pack

# load dataset from pickle files
def load_data(path, Y_start_col, Y_end_col):
  print "\n==== Loading data from Pickle files ====\n"

  # load normalized data from pickle files
  all_data_file_path = path + "/all.pkl"
  if not os.path.exists(all_data_file_path):
    sys.exit("File " + all_data_file_path + " does not exist!\n")
  with open(all_data_file_path, "rb") as f:
    all_data = pickle.load(f)  

  # remove the fields we don't want to predict
  all_data['training_Y'] = all_data['training_Y'][:, Y_start_col:Y_end_col]
  all_data['norm_training_Y'] = all_data['norm_training_Y'][:, Y_start_col:Y_end_col]
  all_data['test_Y'] = all_data['test_Y'][:, Y_start_col:Y_end_col]
  all_data['normVal']['training_Y'] = all_data['normVal']['training_Y'][Y_start_col:Y_end_col]

  print "total training: X", all_data['training_X'].shape, "Y", all_data['norm_training_Y'].shape, ", total test: X", all_data['test_X'].shape, "Y", all_data['test_Y'].shape

  dev_data = []
  for device in [0, 1, 2, 3]:
    dev_data_file_path = path + "/dev" + str(device) + ".pkl"
    if not os.path.exists(dev_data_file_path):
      sys.exit("File " + dev_data_file_path + " does not exist!\n")
    with open(dev_data_file_path, "rb") as f:
      tmp = pickle.load(f)

    # remove the fields we don't want to predict
    tmp['training_Y'] = tmp['training_Y'][:, Y_start_col:Y_end_col]
    tmp['norm_training_Y'] = tmp['norm_training_Y'][:, Y_start_col:Y_end_col]
    tmp['test_Y'] = tmp['test_Y'][:, Y_start_col:Y_end_col]
    tmp['normVal'] = tmp['normVal'][Y_start_col:Y_end_col]
    print "device ", device, "training: X", tmp['training_X'].shape, "Y", tmp['norm_training_Y'].shape, ", test: X", tmp['test_X'].shape, "Y", tmp['test_Y'].shape
    dev_data.append(tmp)
 
  name_file_path = path + "/names.pkl"
  if not os.path.exists(name_file_path):
    sys.exit("File " + name_file_path + " does not exist!\n")
  with open(path + "/names.pkl", "rb") as f:
    names = pickle.load(f)

  # remove the fields we don't want to predict
  names['impl_name'] = names['impl_name'][Y_start_col:Y_end_col]
  print "Predicting: ", names['impl_name']

  return all_data, dev_data, names

# write the result to a csv file
def write_to_csv_resource(path, results):
  
  # header of the file
  header = ['Device', 'Model', 'LUT_RAE', 'FF_RAE', 'DSP_RAE', 'BRAM_RAE']
  data_list = []

  # fill the array
  for key in results:
    # get the result row
    res = results[key]
    
    # process the key to determine the device and model fields
    data_inst = {}
    # if the key ends with a string, then it is the result for a device
    if key[-1].isdigit():
      data_inst['Device'] = key[-1]
      data_inst['Model'] = key[0:-1]
    else:
      data_inst['Device'] = 'average'
      data_inst['Model'] = key

    # fill in the numbers
    data_inst['LUT_RAE']   = res[0]
    data_inst['FF_RAE']    = res[1]
    data_inst['DSP_RAE']   = res[2]
    data_inst['BRAM_RAE']  = res[3]
   
    # add in the data
    data_list.append(data_inst)

  # write out
  with open(path, "wb") as f:
    writer = csv.DictWriter(f, header)
    writer.writeheader()
    for data in data_list:
      writer.writerow(data)

def write_to_csv_timing(path, results):
  
  # header of the file
  header = ['Device', 'Model', 'Timing_Error']
  data_list = []

  # fill the array
  for key in results:
    # get the result row
    res = results[key]
    
    # process the key to determine the device and model fields
    data_inst = {}
    # if the key ends with a string, then it is the result for a device
    if key[-1].isdigit():
      data_inst['Device'] = key[-1]
      data_inst['Model'] = key[0:-1]
    else:
      data_inst['Device'] = 'average'
      data_inst['Model'] = key

    # fill in the numbers
    data_inst['Timing_Error']   = res
   
    # add in the data
    data_list.append(data_inst)

  # write out
  with open(path, "wb") as f:
    writer = csv.DictWriter(f, header)
    writer.writeheader()
    for data in data_list:
      writer.writerow(data)

#========== Use all data to train, test on all data ==========#

def run_all_data(all_data, dev_data, names, FLAGS):

  # run through the total set
  # we use the normalized training Ys to train the models
  # when calculating error rates, we recover the predictions to the scale of original values
  # and compare against non-normalized ground truth results

  ###### training phase

  if FLAGS.train:
    print "\n==== Training models ====\n"

    best_models = {}

    ###### feature selection

    if FLAGS.feature_select:
      # feature selection
      # use normalized results in here
      # since data is normalized, the coefs should be relatively small, use a small threshold
      all_data['training_X'], all_data['test_X'], selected_mask = linearFeatureSelection(all_data['training_X'],
                                                                                         all_data['norm_training_Y'], 
                                                                                         all_data['test_X'], 
                                                                                         names['features_name'],
                                                                                         alpha = 0.001, 
                                                                                         threshold = 0.01); 

      print "after feature selection: total training: X", all_data['training_X'].shape, "Y", all_data['norm_training_Y'].shape, ", total test: X", all_data['test_X'].shape, "Y", all_data['test_Y'].shape
      selected_feature_names = names['features_name'][selected_mask]

      # also filter features for device-specific test sets
      for device in [0, 1, 2, 3]:
        dev_data[device]['test_X'] = dev_data[device]['test_X'][:, selected_mask]

      best_models['selected_mask'] = selected_mask
   
    ###### generate training and validation splits

    all_train_val_pairs = train_val_split(all_data['training_X'], all_data['norm_training_Y'], rounds = 10, val_size = 0.25)

    ###### sweep the model parameters on the whole training set

    model_params = {}
    # generate ANN param list
    model_params['ann'] = gen_ANN_param_list(max_size_1D = 200, max_size_2D = 50)
    model_params['multiann'] = model_params['ann']
    # generate XGBoost param list
    model_params['xgb'] = gen_XGB_param_list(max_depth = 5, max_trees = 500)
    # generate Lasso param list
    model_params['lasso'] = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
    model_params['multilasso'] = model_params['lasso']

    # sweep results for all models
    # record results in a dictionary
    model_list = ['lasso', 'ann', 'xgb', 'multiann', 'multilasso'] 
    # single-task: ann, xgboost, lasso; multi-task: ann, lasso
    for model in model_list:
      # sweep parameters to get the best models, test on the overall test set
      clf_models_RAE = sweep_params(model_params[model], all_train_val_pairs, all_data, model)
      best_models[model] = clf_models_RAE

      # uncomment this to see XGBoost feature importance for each resource
      """
      if model == 'xgb':
        for i in range(len(clf_models_RAE)):
          print "XGB feature importance for Target ", i
          score = clf[i].get_booster().get_fscore()
          sorted_score = sorted(score.iteritems(), key = lambda (k, v): (v, k))
          for key, value in sorted_score:
            fea_id = int(key.replace('f', ''))
            print selected_feature_names[fea_id], value
          print "\n"
      """

    ###### dump out the trained models

    with open(FLAGS.store_model_path, "wb") as f:
      pickle.dump(best_models, f)

  ###### test phase
  
  if FLAGS.test:
    print "\n==== Testing models ====\n"

    ###### load models

    with open(FLAGS.pretrained_model_path, "rb") as f:
      models = pickle.load(f)
    # models should be stored in a dictionary, the same format as we stored above
    if not isinstance(models, dict):
      sys.exit("The pretrained models should be stored in a dictionary. Please check the readme file for format. ")

    ###### feature selection

    if FLAGS.feature_select:
      selected_mask = models['selected_mask']
      # filter features for device-specific test sets
      for device in [0, 1, 2, 3]:
        dev_data[device]['test_X'] = dev_data[device]['test_X'][:, selected_mask]
      # filter for whole test sets
      all_data['test_X'] = all_data['test_X'][:, selected_mask]

      del models['selected_mask']

    ###### test on whole test set and device-specific test sets

    all_results = {}
    for m in models:
      # test
      all_test_RAE = model_testing(models[m], all_data['test_X'], all_data['test_Y'], 
                                   all_data['normVal']['training_Y'], Y_normalized = False)
      all_results[m] = np.array(all_test_RAE)
      # print result
      print m, "on whole test set: ", all_test_RAE

      # test the best model on individual device test sets
      for device in [0, 1, 2, 3]:
        dev_test_RAE = model_testing(models[m], dev_data[device]['test_X'], 
                                     dev_data[device]['test_Y'], all_data['normVal']['training_Y'], Y_normalized = False)
        print m, "on test", device, ":", dev_test_RAE
        all_results[m+str(device)] = np.array(dev_test_RAE)

    ###### write the results into a csv file

    write_to_csv_resource("./resource.csv", all_results)

#========== predict whether the design meets timing or not ==========#

def predict_timing(all_data, dev_data, names, FLAGS):

  ###### generate new training/test Y

  # shortcuts
  target_cp_std = all_data['normVal']['features'][12]['std']
  target_cp_mean = all_data['normVal']['features'][12]['mean']
  # generate new training Y 
  new_training_Y_all = (all_data['training_X'][:, 12] * target_cp_std + target_cp_mean) > all_data['training_Y'][:, 0]
  new_training_Y_all = new_training_Y_all.reshape((-1, 1))
  # new test Y
  new_test_Y_all = (all_data['test_X'][:, 12] * target_cp_std + target_cp_mean) > all_data['test_Y'][:, 0]
  new_test_Y_all = new_test_Y_all.reshape((-1, 1))
  # same for the device-specific datasets
  dev_new_training_Y = []
  dev_new_test_Y = []
  for device in [0, 1, 2, 3]:
    new_training_Y = (dev_data[device]['training_X'][:, 12] * target_cp_std + target_cp_mean) > dev_data[device]['training_Y'][:, 0]
    new_test_Y = (dev_data[device]['test_X'][:, 12] * target_cp_std + target_cp_mean) > dev_data[device]['test_Y'][:, 0]
    new_training_Y = new_training_Y.reshape((-1, 1))
    new_test_Y = new_test_Y.reshape((-1, 1))
    dev_new_training_Y.append(new_training_Y)
    dev_new_test_Y.append(new_test_Y)

  ###### training phase

  if FLAGS.train:
    print "\n==== Training models ====\n"

    best_models = {}

    ###### feature selection

    if FLAGS.feature_select:
      # l1 feature selection
      all_data['training_X'], all_data['test_X'], selected_mask = linearFeatureSelection(all_data['training_X'],
                                                                                         new_training_Y_all, 
                                                                                         all_data['test_X'], 
                                                                                         names['features_name'],
                                                                                         task = 'classification', 
                                                                                         threshold = 0.01)

      # also filter features for device-specific test sets
      for device in [0, 1, 2, 3]:
        dev_data[device]['test_X'] = dev_data[device]['test_X'][:, selected_mask]

      print "after feature selection: total training: X", all_data['training_X'].shape, "Y", new_training_Y_all.shape, ", total test: X", all_data['test_X'].shape, "Y", new_test_Y.shape
      selected_feature_names = names['features_name'][selected_mask]

      best_models['selected_mask'] = selected_mask

    ###### training-validation split

    train_val_pairs = train_val_split(all_data['training_X'], new_training_Y_all, rounds = 10, val_size = 0.25)

    ###### sweep model parameters

    model_params = {}

    # generate XGBoost param list
    model_params['xgb'] = gen_XGB_param_list(max_depth = 5, max_trees = 500)
    # select the best ANN parameters
    model_params['ann'] = gen_ANN_param_list(max_size_1D = 200, max_size_2D = 50)

    for model in ['xgb', 'ann']:
      print model, ":"
      # select the best parameters
      best_error = 999999.9
      best_param = []
      for params in model_params[model]:
        error = 0.0
        for r in range(10):

          if model == 'xgb':
            clf = xgb.XGBClassifier(max_depth=params[0], n_estimators=params[1])
          elif model == 'ann':
            clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes = params, random_state = 0)

          clf.fit(train_val_pairs['train_X'][r], train_val_pairs['train_Y'][r].reshape(train_val_pairs['train_Y'][r].shape[0],))
          res = clf.predict(train_val_pairs['val_X'][r])
          res = res.reshape((-1, 1))
          error = error + (np.sum(res[:, 0] != train_val_pairs['val_Y'][r][:, 0]) / float(res.shape[0]))

        error = error / 10.0
        print "parameter", params, ", val error =", error
        if error < best_error:
          best_error = error
          best_param = params

      print "Best parameter:", best_param
      if model == 'xgb':
        clf = xgb.XGBClassifier(max_depth=best_param[0], n_estimators=best_param[1])
      elif model == 'ann':
        clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes = best_param, random_state = 0)
      clf.fit(all_data['training_X'], new_training_Y_all.reshape(new_training_Y_all.shape[0],))
      best_models[model] = clf

      # uncomment this to see XGBoost feature importance for timing classification
      """
      if model == 'xgb':
        print "XGB feature importance for timing classification:"
        score = clf.get_booster().get_fscore()
        sorted_score = sorted(score.iteritems(), key = lambda (k, v): (v, k))
        for key, value in sorted_score:
          fea_id = int(key.replace('f', ''))
          print selected_feature_names[fea_id], value
        print "\n"
      """

    ###### dump out the trained models 
    with open(FLAGS.store_model_path, "wb") as f:
      pickle.dump(best_models, f)

  ###### test phase

  if FLAGS.test:
    print "\n==== Testing models ====\n"

    ###### load models
  
    if not os.path.exists(FLAGS.pretrained_model_path):
      sys.exit("Pretrained model file " + FLAGS.pretrained_model_path + " does not exist!")

    with open(FLAGS.pretrained_model_path, "rb") as f:
      models = pickle.load(f)
    # models should be stored in a dictionary, the same format as we stored above
    if not isinstance(models, dict):
      sys.exit("The pretrained models should be stored in a dictionary. Please check the readme file for format. ")

    ###### feature selection

    if FLAGS.feature_select:
      selected_mask = models['selected_mask']
      # filter features for device-specific test sets
      for device in [0, 1, 2, 3]:
        dev_data[device]['test_X'] = dev_data[device]['test_X'][:, selected_mask]
      # filter for whole test sets
      all_data['test_X'] = all_data['test_X'][:, selected_mask]

      del models['selected_mask']
    ###### test on whole test set and device-specific test sets

    all_results = {}
    for m in models:
      # test
      all_res = models[m].predict(all_data['test_X'])
      all_res = all_res.reshape((-1, 1))
      test_error = np.sum(all_res[:, 0] != new_test_Y_all[:, 0]) / float(all_res.shape[0])
      all_results[m] = test_error
      # print result
      print m, "on whole test set: ", test_error

      # test the best model on individual device test sets
      for device in [0, 1, 2, 3]:
        dev_res = models[m].predict(dev_data[device]['test_X'])
        dev_res = dev_res.reshape((-1, 1))
        test_error = np.sum(dev_res[:, 0] != dev_new_test_Y[device][:, 0]) / float(dev_res.shape[0])
        print m, "on device ", device, ": ", test_error
        all_results[m+str(device)] = test_error

    ###### write the results into a csv file

    write_to_csv_timing("./timing.csv", all_results)

#========== Main function ==========#

if __name__ == "__main__":

  # fix the random seed so that the results would be reproduceable
  np.random.seed(seed = 6)        

  FLAGS, unparsed = parser.parse_known_args()

  dataPath = FLAGS.data_dir

  if not os.path.exists(dataPath):
    sys.exit("Input data directory does not exist!")

  if FLAGS.predict_area:
    # load data from pickle file
    all_data, dev_data, names = load_data(dataPath, 2, 6)
    # run on the whole dataset
    run_all_data(all_data, dev_data, names, FLAGS)

  if FLAGS.classify_timing:
    # load data from pickle file
    all_data, dev_data, names = load_data(dataPath, 0, 1)
    # predict timing
    predict_timing(all_data, dev_data, names, FLAGS)
