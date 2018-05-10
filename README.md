# DABoost

## Publication
-------------------------------------------------------------------------------------------
If you use DABoost in your research, please cite [our FCCM'18 paper][1]:
```
  @article{dai-hls-qor-fccm2018,
    title   = "{Fast and Accurate Estimation of Quality of Results in 
                High-Level Synthesis with Machine Learning}",
    author  = {Steve Dai and Yuan Zhou and Hang Zhang and Ecenur Ustun and 
               Evangeline F.Y. Young and Zhiru Zhang},
    journal = {Int'l Symp. on Field-Programmable Custom Computing Machines
               (FCCM)},
    month   = {May},
    year    = {2018},
  }
```
[1]: http://www.csl.cornell.edu/~zhiruz/pdfs/hls-qor-fccm2018.pdf 

## Usage

DABoost is organized into directories for different estimation tasks.

### HLS

[hls](./hls) directory currently supports resource estimation and timing 
classification for HLS. The python files for these features are in `[path to top directory]/hls`. Usage of the main script is as follows:

```
usage: main.py [-h] [--data_dir DATA_DIR] [--predict_area] [--classify_timing]
               [--train] [--store_model_path STORE_MODEL_PATH] [--test]
               [--pretrained_model_path PRETRAINED_MODEL_PATH]
               [--feature_select]

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Directory to the input dataset.
  --predict_area        Run area estimation.
  --classify_timing     Run timing classification.
  --train               Run training for the specified task.
  --store_model_path STORE_MODEL_PATH
                        Path to store the trained models
  --test                Run test for the specified task. Should use with the '
                        --pretrained_model_path' option.
  --pretrained_model_path PRETRAINED_MODEL_PATH
                        Path to pretrained models. The models should be dumped
                        in pickle files.
  --feature_select      Use feature selection. If this option is used with '--
                        train', the generated model would contain a mask
                        indicating the selected features. If used with '--
                        test', then the produced model must provide the mask
                        as well.
```
If you have some pretrained models and want to test them on the dataset, please make sure:
1. Your models are dumped out using `pickle` and stored in a `.pkl` file;
2. Your `.pkl` file should contain a dictionary indexed by string, where each string is mapped to a (list of) models. Below is an example:
```python
  models = {
    'xgb': [xgb_model_0, xgb_model_1, xgb_model_2, xgb_model_3],
    'ann': [ann_model_0, ann_model_1, ann_model_2, ann_model_3]
  }
```
3. If feature selection has been applied when training your models, please provide a mask containing the column indices of the selected features. The mask should be stored in the `selected_mask` field of the dictionary. Below is an example:
```python
  models = {
    'xgb': [xgb_model_0, xgb_model_1, xgb_model_2, xgb_model_3],
    'ann': [ann_model_0, ann_model_1, ann_model_2, ann_model_3]
    'selected_mask': [0 1 2 4 8 10]
    # use the 0th, 1st, 2nd, 4th, 8th, and the 10th column of the features
  }
```

Usage of the data preprocessing script is as follows:

```
python preprocessing.py --data_dir [raw data directory containing 
                                    data.csv and feature_names.txt]
                        --output_dir [output data directory containing 
                                      .pkl files for main.py]
```

