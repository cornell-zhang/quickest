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

To preprocess the raw .csv data and generate training and testing datasets
in .pkl, run *preprocess.py* with the relevant options:

```
python preprocess.py  [-h] [--data_dir DATA_DIR]

optional arguments:
  -h, --help           show this help message and exit
  --data_dir DATA_DIR  Directory to the input dataset.
```

To train the models for resource estimation with or without feature 
selection using a .pkl training dataset, run *train.py* with the relevant 
options:

```
python train.py [-h] [--data_dir DATA_DIR] 
		[--save_model_dir SAVE_MODEL_DIR] [--feature_select]

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Directory to the training dataset.
  --save_model_dir SAVE_MODEL_DIR
                        Directory to save the trained model. Input folder or
                        file name.
  --feature_select      Use feature selection.
```

To test the trained models for resource estimation using a .pkl testing
dataset, run *test.py* with the relevant options:

```
python test.py [-h] [--data_dir DATA_DIR] [--model_dir MODEL_DIR] 
	       [--save_result_dir SAVE_RESULT_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Directory to the testing dataset.
  --model_dir MODEL_DIR
                        Directory to the pre-trained models.
  --save_result_dir SAVE_RESULT_DIR
                        Directory to save the result. Input folder or file
                        name.
```

 

