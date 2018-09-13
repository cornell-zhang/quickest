
import csv
import os
import datetime
import sys
import argparse
# import pandas as pd

from funcs import Make, MakeF, CheckFiles, ExtractData, WhatFeatures

parser = argparse.ArgumentParser()
# parser.add_argument('--cp', type=int, default=2)
parser.add_argument('--source_dir', type=str, default='.')
parser.add_argument('--list_file', type=str, default='./folderlist.csv')
parser.add_argument('--search', type=str, default='r')
parser.add_argument('--func', type=str, default='l')

# LogFile = None
Design_Index = 0
    
def excute_funcs(func, ns=[2]):
    global ResultFile
    global Design_Index
    
    # add index
    Design_Index += 1
    
    # function
    if func.find('l') > -1:
        ResultFile.write(os.path.abspath('.') + '\n')
        print os.path.abspath('.')
    # function
    if func.find('m') > -1:
        Make(ns, ResultFile)
    # function
    if func.find('f') > -1:
        MakeF(ResultFile)
    # function
    if func.find('c') > -1:
        CheckFiles(ResultFile)
    # function
    if func.find('d') > -1:
        ExtractData(Design_Index, ResultFile)
    # function
    if func.find('w') > -1:
        WhatFeatures()
     

# go to folders by recusion
def recusionFolder(source_dir, func, ns=[2]):
    os.chdir(source_dir)
    
    # search flags
    makeflag1, makeflag2 = False, False
    for x in os.listdir('.'):
        if x[len(x) - 2: len(x)] == '.c': makeflag1 = True
        if x == 'Makefile': makeflag2 = True

    if makeflag1 and makeflag2:
        # excute functions
        excute_funcs(func, ns=ns)
    else:
        # recusion
        for x in os.listdir('.'):
            if os.path.isdir(x):
                recusionFolder(x, func, ns=ns)
    
    os.chdir('..')


# go to folder by given list
def listFolder(list_file, func, ns=[2]):
    # load folder list
    list_dir = []
    with open(list_file, 'r') as f:
        for line in f.readlines():
            if line[-1] == '\n':
                line = line[0: -1]
            list_dir.append(line)
    # list_dir = pd.read_csv(list_file, header=-1, names=['folder'])
    
    for to_dir in list_dir:
        os.chdir(to_dir)
        # excute functions
        excute_funcs(func, ns=ns)


if __name__ == "__main__":
    global ResultFile
    
    FLAGS, unparsed = parser.parse_known_args()

    ResultFile = open("Result_" + FLAGS.func + ".csv", "w")

    if FLAGS.search == 'r':
        print "Start! Search file by recursion. Root folder is", FLAGS.source_dir	
        recusionFolder(FLAGS.source_dir, FLAGS.func, ns=[1, 15]) # [2, 5, 10, 20]
    elif FLAGS.search == 'l':
        print "Start! Search file by list. List file is", FLAGS.list_file	
        listFolder(FLAGS.list_file, FLAGS.func, ns=[1, 15]) # [2, 5, 10, 20]

    # close log file
    ResultFile.close()
