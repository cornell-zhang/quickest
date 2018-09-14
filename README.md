# QuickEst

## Publication
--------------------------------------------------------------------------
If you use QuickEst in your research, please cite our preliminary work 
published in FCCM'18.

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
## Usage

QuickEst is organized into directories for different estimation tasks.

### HLS

[hls](./hls) directory currently supports resource estimation for HLS. 
The python files for these features are in *[path to top directory]/hls*.

## Input data format

```
The data file is CSV file. The columns are separated by ",". The rows are separated by "\n".
The following formats should also be satisfied:
    1) The first 2 columns should be design index and device index respectively.
    2) The columns from 3 to <feature_col> should be features.
    3) The columns from <feature_col> to the end should be the targets.
    4) If there are k targets, the first k features should be the corresponding HLS result of the k targets.
```


## Data preprocessing
python preprocess.py [-h] [--data_dir DATA_DIR] [-c FEATURE_COL]
                     [--test_seed TEST_SEED] [--cluster_k CLUSTER_K]

```
optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Directory or file of the input data. String. Default:
                        ./data/data.csv
  -c FEATURE_COL, --feature_col FEATURE_COL
                        The index (start from 1) of the last feature column.
                        The first 2 columns are design index and device index
                        respectively. Integer. Default: 236
  --test_seed TEST_SEED
                        The seed used for selecting the test id. Integer.
                        Default: 0
  --cluster_k CLUSTER_K
                        How many clusters will be grouped when patitioning the
                        training and testing dataset. Integer. Default: 8
```

## Model training
python train.py [-h] [--data_dir DATA_DIR] [--params_dir PARAMS_DIR]
                [--models_dir MODELS_DIR] [-t]
                [--validation_ratio VALIDATION_RATIO] [-m MODEL_TRAIN]
                [-s MODEL_FSEL] [-a MODEL_ASSEMBLE]

```
optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Directory or file of the training dataset. String.
                        Default: ./data/data_train.pkl
  --params_dir PARAMS_DIR
                        Directory or file to load and save the parameters.
                        String. Default: ./saves/train/params.pkl
  --models_dir MODELS_DIR
                        Directory or file to save the trained model. String.
                        Default: ./saves/train/models.pkl
  -t, --tune_parameter  Whether to tune parameters or not. Boolean. Default:
                        true
  --validation_ratio VALIDATION_RATIO
                        The ratio of the training data to do validation.
                        Float. Default: 0.25
  -m MODEL_TRAIN, --model_train MODEL_TRAIN
                        The model to be trained. Empty means not training
                        models. Value from "", "xgb"(default), "lasso"
  -s MODEL_FSEL, --model_fsel MODEL_FSEL
                        The model used to select features. Empty means not
                        selecting features. Value from "", "xgb",
                        "lasso"(default)
  -a MODEL_ASSEMBLE, --model_assemble MODEL_ASSEMBLE
                        Strategy used to assemble the trained models. Empty
                        means not training models. Value from ""(default),
                        "xgb+lasso+equal_weights", "xgb+lasso+learn_weights"
```

## Model testing
python test.py [-h] [--data_dir DATA_DIR] [--models_dir MODELS_DIR]
               [--save_result_dir SAVE_RESULT_DIR]

```
optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Directory or file of the testing dataset. String.
                        Default: ./data/data_test.pkl
  --models_dir MODELS_DIR
                        Directory or file of the pre-trained models. String.
                        Default: ./train/models.pkl
  --save_result_dir SAVE_RESULT_DIR
                        Directory to save the result. Input folder or file
                        name. String. Default: ./saves/test/
```

## Model analyzing
usage: analyze.py [-h] [--train_data_dir TRAIN_DATA_DIR]
                  [--test_data_dir TEST_DATA_DIR] [--model_dir MODEL_DIR]
                  [--param_dir PARAM_DIR] [--result_dir RESULT_DIR]
                  [--save_result_dir SAVE_RESULT_DIR] [-f FUNC]

```
optional arguments:
  -h, --help            show this help message and exit
  --train_data_dir TRAIN_DATA_DIR
                        File of the training dataset. String. Default:
                        ./data/data_train.pkl
  --test_data_dir TEST_DATA_DIR
                        File of the testing dataset. String. Default:
                        ./data/data_test.pkl
  --model_dir MODEL_DIR
                        File of the pre-trained models. String. Default:
                        ./save/train/models.pkl
  --param_dir PARAM_DIR
                        File of the pre-tuned params. String. Default:
                        ./save/train/params.pkl
  --result_dir RESULT_DIR
                        File of the testing results. String. Default:
                        ./save/test/results.pkl
  --save_result_dir SAVE_RESULT_DIR
                        Directory to save the analyzing results. String.
                        Default: ./save/analysis/
  -f FUNC, --func FUNC  Select the analysis function. Value from "fi" or
                        "feature_importance", "schls" or "score_hls", "sc" or
                        "score" (default), "lc" or "learning_curve", "re" or
                        "result_error", "red" or "result_error_design", "rt"
                        or "result_truth", "rtd" or "result_truth_design",
                        "rp" or "result_predict", "rpd" or
                        "result_predict_design".
                        
function explaination:
fi - calculate feature importance
sc - calculate the scores of model results
schls - calculate the scores of HLS results
lc - plot the learning curve
re - show the result errors
red - show the result errors grouped by the design ids
rt - show the result ground truth of the testing data
rtd - show the result truth of the testing data grouped by the design ids
rp - show the result prediction of the testing data
rpd - show the result prediction grouped by the design ids
```
