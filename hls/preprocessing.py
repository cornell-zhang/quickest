# basics
import os
import sys
import numpy as np

# train_test_split
from sklearn.model_selection import train_test_split

# pickle
import cPickle as pickle

# argparse
import argparse

# setup parser
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type = str, default = './data/',
                    help = 'Directory to the input data.')
parser.add_argument('--output_dir', type = str, default = './out/',
                    help = 'Directory to the output data.')

#================================ Split data =================================#

# group the columns according to their catagories
# target device information, features, lsynth result, impl result
# returns a dictionary, string index, contains the corresponding columns as numpy arrays
def group_cols(data):

  # resource and cp after impl
  # cp, slice, LUT, FF, DSP, BRAM, SRL
  impl = data[:, -7:]
  # resource and cp after logic synthesis
  # cp, LUT, FF, DSP, BRAM, SRL
  lsynth = data[:, -13:-7]
  # features
  features = data[:, 2:-13]
  # device family information
  devices = data[:, 0:2]

  data_pack = {'impl': impl, 
               'lsynth': lsynth, 
               'features': features, 
               'devices': devices}

  return data_pack

# group the names of the columns in the same way
def group_names(names):

  impl_name = names[-7:]
  lsynth_name = names[-13:-7]
  features_name = names[2:-13]
  devices_name = names[0:2]

  name_pack = {'impl_name': impl_name, 
               'lsynth_name': lsynth_name, 
               'features_name': features_name,
               'devices_name': devices_name}

  return name_pack

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

# main script: load data, split according to device family, and save them into files
# sys.argv[1] is the output directory
if __name__ == "__main__":

  # fix the random seed so that the results would be reproduceable
  # comment out this line to generate different datasets
  np.random.seed(seed = 6)        

  # parse arguments and set up paths
  FLAGS, unparsed = parser.parse_known_args()
  folderPath = FLAGS.data_dir
  outputPath = FLAGS.output_dir
  dataFile   = 'data.csv'
  nameFile   = 'feature_names.txt'

  if not os.path.exists(folderPath):
    sys.exit('Input directory does not exist!')
  if not os.path.exists(outputPath):
    os.makedirs(outputPath)

  # load raw data and the names of all columns
  all_raw_data = np.loadtxt(folderPath + dataFile, delimiter=',', skiprows=1)
  all_col_names = np.loadtxt(folderPath + nameFile, dtype = str, delimiter=',')

  # group the columns and names
  
  # names
  namepack = group_names(all_col_names) 

  # feature and result columns
  all_datapack = group_cols(all_raw_data)
  
  # prepare a dictionary for normalization values
  all_normVal = {}

  # normalize features
  all_datapack['features'], all_normVal['features'] = normalize_data(all_datapack['features'])

  print "all data: features: ", all_datapack['features'].shape, " results: ", all_datapack['impl'].shape

  # split according to different device families
  # still keep the original one
  
  # 0: virtex-7, 1: zynq, 2: artix, 3: kintex
  features = []          # feature matrices
  impl_data = []         # result matrices
  for device in [0, 1, 2, 3]:
    tmp_X = all_datapack['features'][all_datapack['devices'][:, 0] == device]
    tmp_Y = all_datapack['impl'][all_datapack['devices'][:, 0] == device]
    print "device ", device, ": ", tmp_X.shape, tmp_Y.shape

    features.append(tmp_X)
    impl_data.append(tmp_Y)

  # split training and test for each device
  trainings_X = []       # training feature matrices
  trainings_Y = []       # training result matrices
  norm_trainings_Y = []  # normalized training result matrices
  tests_X = []           # test feature matrices
  tests_Y = []           # test result matrices
  Y_normVals = []        # result normalization values for each device
  for device in [0, 1, 2, 3]:
    # split
    training_X, test_X, training_Y, test_Y = train_test_split(features[device], impl_data[device], test_size = 0.2)
    print "device ", device, ": training ", training_X.shape, training_Y.shape, ", test ", test_X.shape, test_Y.shape
    
    # normalize training Ys
    norm_training_Y, normVal = normalize_data(training_Y)

    # append to lists
    trainings_X.append(training_X)
    trainings_Y.append(training_Y)
    norm_trainings_Y.append(norm_training_Y)
    tests_X.append(test_X)
    tests_Y.append(test_Y)
    Y_normVals.append(normVal)

  # merge the training for all devices into a large training set
  # same with test
  all_training_X = trainings_X[0]
  all_training_Y = trainings_Y[0]
  all_test_X = tests_X[0]
  all_test_Y = tests_Y[0]

  for device in [1, 2, 3]:
    all_training_X = np.vstack((all_training_X, trainings_X[device]))
    all_training_Y = np.vstack((all_training_Y, trainings_Y[device]))
    all_test_X = np.vstack((all_test_X, tests_X[device]))
    all_test_Y = np.vstack((all_test_Y, tests_Y[device]))

  print "total training: ", all_training_X.shape, all_training_Y.shape, ", total test: ", all_test_X.shape, all_test_Y.shape

  # normalize the training results
  norm_all_training_Y, all_normVal['training_Y'] = normalize_data(all_training_Y)

  # write processed datasets into files

  # dataset containing all devices
  all_data = {'training_X': all_training_X, 'training_Y': all_training_Y, 'norm_training_Y': norm_all_training_Y, 'test_X': all_test_X, 'test_Y': all_test_Y, 'normVal': all_normVal}
  with open( outputPath + "/all.pkl", "wb" ) as f:
    pickle.dump(all_data, f)

  # device-specific dataset
  for device in [0, 1, 2, 3]:
    dev_data = {'training_X': trainings_X[device], 
                'training_Y': trainings_Y[device], 
                'norm_training_Y': norm_trainings_Y[device], 
                'test_X': tests_X[device], 
                'test_Y': tests_Y[device], 
                'normVal': Y_normVals[device]}    
    with open( outputPath + "/dev" + str(device) + ".pkl", "wb" ) as f:
      pickle.dump(dev_data, f )

  with open( outputPath + "/names.pkl", "wb") as f:
    pickle.dump(namepack, f)
