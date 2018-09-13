# -*- coding: utf-8 -*-
# This file is used to preprocess the raw data.
import os
import argparse
import pickle
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data/data.csv', 
                    help='Directory or file of the input data. \
                    String. Default: ./data/data.csv')

parser.add_argument('--feature_col', type=int, default=236,  # 87， 92, 236
                    help='The index (start from 1) of the last feature column.\
                    The first 2 columns are design index and device index respectively. \
                    Integer. Default: 236')

parser.add_argument('--test_seed', type=int, default=0, 
                    help='The seed used for selecting the test id. \
                    Integer. Default: 0')

parser.add_argument('--cluster_k', type=int, default=8, 
                    help='How many clusters will be grouped when patitioning the training and testing dataset.\
                    Integer. Default: 8')


def load_data(file_name, feature_col, test_seed, split_by='design_sort', # design
              test_ratio=0.25, test_ids=[3], cluster_k=8):
    """
    This function is used to load the data to be preprocessed.
    The format of the data file needs to be:
        1) The first 2 columns should be design index and device index respectively.
        2) The columns from 3 to <feature_col> should be features.
        3) The columns from <feature_col> to the end should be the targets.
        4) If there are k targets, the first k features should be the corresponding HLS result of the k targets.
    --------------
    Prameters:
        file_name: The path of the raw data file.
        feature_col: The column index of the last feature column.
        test_seed: The seed to control the random when select the test data.
        split_by: The strategy to split the data. 
            random - Splitting the data by the id randomly. The ratio of the testing data is given by <ratio>
            design_random - Splitting the data by the design id randomly. The ratio of the testing designs is given by <ratio>
            design_select - Splitting the data by the design id. The testing design id is given by <test_ids>.
            design_sort - Splitting the data by the design id. The design ids are clustered, \
                          sorted by the target values and the splitted.
                          The number of cluster groups are controlled by <cluster_k>.
        test_ratio - Used when <split_by> is "random" or "design_random"
        test_ids - Used when <split_by> is "design_select" 
        cluster_k - Used when <split_by> is "design_sort"
    """
    # load data
    data, names, stas = split_feature_target(file_name=file_name, 
                                             feature_col=feature_col)
    
    # split data
    data, index = split_train_test(data[0], data[1], split_by=split_by, 
                                   test_seed=test_seed,
                                   test_ratio=test_ratio, 
                                   test_ids=test_ids, 
                                   cluster_k=cluster_k)
    
    # return 
    return data, index, names, stas


def split_feature_target(file_name, feature_col,
        is_normalizeX=True, is_normalizeY=True, is_shuffle_design=True):
    """
    Split the data to feature dataset and target dataset. (Split the data by columns)
    """
    
    df = pd.read_csv(file_name, sep=',') 
    
    df_features = df[df.columns[2: feature_col]].copy()
    df_targets = df[df.columns[feature_col::]].copy()
    
    # the standard variance and mean
    mean_features = list(df_features.mean())
    std_features = list(df_features.std() + 1e-6)
    mean_targets = list(df_targets.mean())
    std_targets = list(df_targets.std() + 1e-6)
    
    # normalization
    if is_normalizeX:
        df_features = (df_features - mean_features) / std_features
        
    if is_normalizeY:
        df_targets = (df_targets - mean_targets) / std_targets
    
    # shuffle the design id
    map_design = pd.DataFrame()
    map_design['Value'] = pd.Series(df['Design_Index'].unique()).sort_values()
    map_design['Index'] = map_design['Value']
    map_design = map_design.set_index('Index')
    
    if is_shuffle_design:
        np.random.seed(7)
        map_design['Value'] = np.random.permutation(len(map_design))
        
    # add index
    df_features['Design_Index'] = df['Design_Index'].apply(lambda x:map_design['Value'][x]).copy()
    df_features['Device_Index'] = df['Device_Index'].copy()
    
    df_targets['Design_Index'] = df['Design_Index'].apply(lambda x:map_design['Value'][x]).copy()
    df_targets['Device_Index'] = df['Device_Index'].copy()
    
    # the names
    name_features = df_features.columns.tolist()
    name_targets = df_targets.columns.tolist()
    name_features.remove('Design_Index')
    name_features.remove('Device_Index')
    name_targets.remove('Design_Index')
    name_targets.remove('Device_Index')
    
    # return
    return [df_features, df_targets], [name_features, name_targets],  \
           [mean_features, std_features, mean_targets, std_targets]
    
    
def split_train_test(df_features, df_targets, split_by, test_seed, 
                     test_ratio, test_ids, cluster_k,
                     is_tonumpy=True, is_dropindex=True):
    """
    Split the data to training dataset and testing dataset. (Split the data by rows)
    """
    # set random seed
    np.random.seed(seed=test_seed)
        
    # split data
    if split_by == 'random':
        # select design ID
        data_indexes = [ii for ii in xrange(df_features.shape[0])]
        np.random.shuffle(data_indexes)
        data_indexes = data_indexes[0: int(len(data_indexes) * test_ratio)]
        
        # split dataset
        x_train = df_features[~df_features.index.isin(data_indexes)]
        y_train = df_targets[~df_targets.index.isin(data_indexes)]
        
        x_test = df_features[df_features.index.isin(data_indexes)]
        y_test = df_targets[df_targets.index.isin(data_indexes)]
        
    elif split_by == 'design' or split_by == 'design_random':
        # select design ID
        design_indexes = df_features['Design_Index'].unique().tolist()
        np.random.shuffle(design_indexes)
        design_indexes = design_indexes[0: int(len(design_indexes) * test_ratio)]
        
        # split dataset
        x_train = df_features[~df_features['Design_Index'].isin(design_indexes)]
        y_train = df_targets[~df_targets['Design_Index'].isin(design_indexes)]
        
        x_test = df_features[df_features['Design_Index'].isin(design_indexes)]
        y_test = df_targets[df_targets['Design_Index'].isin(design_indexes)]
    
    elif split_by == 'design_select':
        # split dataset
        x_train = df_features[~df_features['Design_Index'].isin(test_ids)]
        y_train = df_targets[~df_targets['Design_Index'].isin(test_ids)]
        
        x_test = df_features[df_features['Design_Index'].isin(test_ids)]
        y_test = df_targets[df_targets['Design_Index'].isin(test_ids)]
    
    elif split_by == 'device_select':
        # split dataset
        x_train = df_features[~df_features['Device_Index'].isin(test_ids)]
        y_train = df_targets[~df_targets['Device_Index'].isin(test_ids)]
        
        x_test = df_features[df_features['Device_Index'].isin(test_ids)]
        y_test = df_targets[df_targets['Device_Index'].isin(test_ids)]
    
    elif split_by == 'design_sort':
        # cluster
        x_cluster = df_targets.groupby(['Design_Index']).mean()
        x_cluster = x_cluster.drop(['Device_Index'], axis=1)
        
        model = KMeans(n_clusters=cluster_k) 
        model.fit(x_cluster)
        
        # extract test ids
        y_cluster = pd.DataFrame(index=x_cluster.index)
        y_cluster['Cluster'] = model.labels_
        y_cluster['Target'] = x_cluster.mean(axis=1)
        
        sort_ids = y_cluster.sort_values(['Cluster', 'Target']).index
        test_ids = sort_ids[test_seed % cluster_k: -1: int(1 / test_ratio)].tolist()
        
        # split dataset
        x_train = df_features[~df_features['Design_Index'].isin(test_ids)]
        y_train = df_targets[~df_targets['Design_Index'].isin(test_ids)]
        
        x_test = df_features[df_features['Design_Index'].isin(test_ids)]
        y_test = df_targets[df_targets['Design_Index'].isin(test_ids)]
        
    # pick the index
    design_index_train = x_train['Design_Index']
    design_index_test = x_test['Design_Index']
    device_index_train = x_train['Device_Index']
    device_index_test = x_test['Device_Index']
        
    # drop index
    if is_dropindex:
        x_train = x_train.drop(['Design_Index', 'Device_Index'], axis=1)
        y_train = y_train.drop(['Design_Index', 'Device_Index'], axis=1)
        x_test = x_test.drop(['Design_Index', 'Device_Index'], axis=1)
        y_test = y_test.drop(['Design_Index', 'Device_Index'], axis=1)
    
    # to numpy
    if is_tonumpy:
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        design_index_train = np.array(design_index_train)
        design_index_test = np.array(design_index_test)
        device_index_train = np.array(device_index_train)
        device_index_test = np.array(device_index_test)
    
    # return
    return [x_train, y_train, x_test, y_test], [design_index_train, design_index_test, device_index_train, device_index_test]
    

if __name__ == '__main__':
    # parser
    FLAGS, unparsed = parser.parse_known_args()
    
    # print info
    print "\n========== Start preprocessing ==========\n"
    
#    # fix the random seed
#    np.random.seed(seed = 6)
    
    # file names
    if os.path.isdir(FLAGS.data_dir):
        file_dir = FLAGS.data_dir
        file_name = 'data.csv'
    elif os.path.isfile(FLAGS.data_dir):
        file_dir, file_name = os.path.split(FLAGS.data_dir)
    else:
        print "File not found. The default data path (./data/data.csv) is used."
        file_dir = './data/'
        file_name = 'data.csv'
    
    # file path
    file_load = os.path.join(file_dir, file_name)
    file_save_train = os.path.join(file_dir, os.path.splitext(file_name)[0] + '_train.pkl')
    file_save_test = os.path.join(file_dir, os.path.splitext(file_name)[0] + '_test.pkl')
    
    print "Load data from", file_load
    
    # get data
    data, index, names, stas = load_data(file_name=file_load, 
                                         feature_col=FLAGS.feature_col,
                                         test_seed=FLAGS.test_seed,
                                         cluster_k=FLAGS.cluster_k)
    
    x_train = data[0]
    y_train = data[1]
    x_test = data[2]
    y_test = data[3]
    
    design_index_train = index[0]
    design_index_test = index[1]
    device_index_train = index[2]
    device_index_test = index[3]
    
    name_features = names[0]
    name_targets = names[1]
    
    mean_features = stas[0]
    std_features = stas[1]
    mean_targets = stas[2]
    std_targets = stas[3]
    
    # save file
    with open(file_save_train, "wb") as f:
        pickle.dump({'x': x_train, 'y': y_train, 
                     'desii': design_index_train, 'devii': device_index_train,
                     'fname': name_features, 'tname': name_targets,
                     'fmean': mean_features, 'tmean': mean_targets,
                     'fstd': std_features, 'tstd': std_targets}, f)
        
    with open(file_save_test, "wb") as f:
        pickle.dump({'x': x_test, 'y': y_test, 
                     'desii': design_index_test, 'devii': device_index_test,
                     'fname': name_features, 'tname': name_targets,
                     'fmean': mean_features, 'tmean': mean_targets,
                     'fstd': std_features, 'tstd': std_targets}, f)
        
    print "Save training data to", file_save_train
    print "Save testing data to", file_save_test
        
    print "\n========== End ==========\n"
    